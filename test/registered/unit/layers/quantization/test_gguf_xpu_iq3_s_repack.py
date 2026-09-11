import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType
from gguf import dequantize

import sglang.srt.layers.quantization.gguf as gguf_xpu
from sglang.srt.layers.quantization.gguf import (
    _q5q6_col_perm_elems,
    _xpu_dequant_iq3_s,
    _xpu_dequant_rep,
    _xpu_dequant_rep_to_fp16,
    _xpu_group_shards,
    _xpu_groups_m_ok,
    _xpu_perm_rep_rows,
    _xpu_prepare_shard,
    _xpu_rep_gemv_into,
    _xpu_rep_gemv_m_into,
    _xpu_repack_iq3_s,
    _xpu_repack_rows_chunked,
    _xpu_try_merge_shards,
)


_COL_PERM = (3, 8, 128)
_BLOCK_BYTES = 110
_BLOCK_SIZE = 256


def _make_iq3_s_bytes(rows: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260910)
    k = np.prod(_COL_PERM)
    qweight = torch.randint(
        0,
        256,
        (rows, k // _BLOCK_SIZE, _BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    # Avoid random half NaN/Inf while preserving random quant payload/scales.
    d = torch.tensor([0.00390625], dtype=torch.float16).view(torch.uint8)
    qweight[:, :, 0:2] = d
    return qweight.view(rows, -1).contiguous()


def _reference(qweight: torch.Tensor) -> torch.Tensor:
    raw = np.array(qweight.numpy(), copy=True)
    return torch.from_numpy(dequantize(raw, WeightType.IQ3_S))


def test_iq3_s_canonical_dequant_matches_gguf_reference():
    qweight = _make_iq3_s_bytes()
    qs, qh, signs, scale = _xpu_repack_iq3_s(qweight)
    actual = _xpu_dequant_iq3_s(qs, qh, signs, scale, torch.float32)
    expected = _reference(qweight)

    assert qs.shape == (qweight.shape[0], actual.shape[1] // 4)
    assert qh.shape == (qweight.shape[0], actual.shape[1] // 32)
    assert scale.shape == (qweight.shape[0], actual.shape[1] // 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-5)


def test_iq3_s_col_perm_matches_dense_reference():
    qweight = _make_iq3_s_bytes()
    qs, qh, signs, scale = _xpu_repack_iq3_s(qweight)
    pqs, pqh, psigns, pscale = _xpu_repack_iq3_s(qweight, col_perm=_COL_PERM)

    expected = _q5q6_col_perm_elems(
        _xpu_dequant_iq3_s(qs, qh, signs, scale, torch.float32), _COL_PERM
    )
    actual = _xpu_dequant_iq3_s(pqs, pqh, psigns, pscale, torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_iq3_s_row_chunking_matches_whole_repack():
    qweight = _make_iq3_s_bytes(rows=5)
    expected = _xpu_repack_iq3_s(qweight, col_perm=_COL_PERM)
    actual = _xpu_repack_rows_chunked(
        _xpu_repack_iq3_s,
        qweight,
        chunk_rows=2,
        col_perm=_COL_PERM,
    )
    for chunked, whole in zip(actual, expected):
        torch.testing.assert_close(chunked, whole, rtol=0, atol=0)


@pytest.mark.parametrize(
    "col_perm,match",
    [
        ((3, 8, 127), "divisible by 32"),
        ((3, 7, 128), "does not match K"),
    ],
)
def test_iq3_s_col_perm_rejects_invalid_shape(col_perm, match):
    with pytest.raises(ValueError, match=match):
        _xpu_repack_iq3_s(
            _make_iq3_s_bytes(rows=1), col_perm=col_perm
        )


def test_iq3_s_repack_rejects_partial_block():
    with pytest.raises(ValueError, match="invalid IQ3_S"):
        _xpu_repack_iq3_s(torch.zeros((1, 109), dtype=torch.uint8))


@pytest.mark.parametrize('index', [0, 255, 256, 511])
@pytest.mark.parametrize('nibble', [0, 15])
def test_iq3_s_grid_high_sign_and_scale_boundaries(index, nibble):
    from gguf.quants import IQ3_S

    raw = torch.zeros((1, 110), dtype=torch.uint8)
    raw[:, :2] = torch.tensor([0.00390625], dtype=torch.float16).view(torch.uint8)
    raw[:, 2:66] = index & 255
    raw[:, 66:74] = 255 if index >= 256 else 0
    # Every bit position occurs with both signs, including bit 7.
    raw[:, 74:106] = torch.arange(32, dtype=torch.uint8) * 7
    raw[:, 106:110] = nibble | (nibble << 4)
    rep = _xpu_repack_iq3_s(raw)
    actual = _xpu_dequant_iq3_s(*rep, torch.float32)
    torch.testing.assert_close(actual, _reference(raw), rtol=0, atol=0)
    IQ3_S.init_grid()
    sign = 1 - 2 * ((raw[:, 74:106].int().unsqueeze(-1) >> torch.arange(8)) & 1)
    expected = torch.from_numpy(IQ3_S.grid.reshape(512, 4)[index]).repeat(64)
    expected = expected * sign.reshape(256) * (0.00390625 * (1 + 2*nibble))
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
def test_prepare_iq3_s_native_fallback_and_dense(monkeypatch):
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_iq3_s", lambda *args: None)
    monkeypatch.delenv("SGLANG_GGUF_XPU_FORCE_DEQUANT", raising=False)
    monkeypatch.delenv("SGLANG_GGUF_XPU_NO_IQ3S", raising=False)
    raw = _make_iq3_s_bytes()
    rep = _xpu_prepare_shard(raw, int(WeightType.IQ3_S), torch.float16)
    assert rep[0] == "iq3_s"
    dense = _xpu_dequant_iq3_s(*rep[1:], torch.float16)
    for reconstruct in (_xpu_dequant_rep, _xpu_dequant_rep_to_fp16):
        torch.testing.assert_close(reconstruct(rep, torch.float16), dense, rtol=0, atol=0)
    monkeypatch.setenv("SGLANG_GGUF_XPU_NO_IQ3S", "1")
    fallback = _xpu_prepare_shard(raw, int(WeightType.IQ3_S), torch.float16)
    assert fallback[0] == "fp16"
    torch.testing.assert_close(fallback[1], _reference(raw).half(), rtol=0, atol=0)
    monkeypatch.delenv("SGLANG_GGUF_XPU_NO_IQ3S")
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_iq3_s", None)
    assert _xpu_prepare_shard(raw, int(WeightType.IQ3_S), torch.float16)[0] == "fp16"


def test_iq3_s_row_permute_merge_and_mixed_group():
    rep = ("iq3_s",) + _xpu_repack_iq3_s(_make_iq3_s_bytes(rows=4))
    perm = torch.tensor([3, 1, 0, 2])
    for actual, original in zip(_xpu_perm_rep_rows(rep, perm)[1:], rep[1:]):
        torch.testing.assert_close(actual, original[perm], rtol=0, atol=0)
    a = ("iq3_s",) + tuple(t[:1] for t in rep[1:])
    b = ("iq3_s",) + tuple(t[1:] for t in rep[1:])
    merged, sizes = _xpu_try_merge_shards({0: a, 1: b}, [0, 1])
    assert sizes == [1, 3]
    for actual, original in zip(merged[1:], rep[1:]):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    iq4 = ("iq4", torch.zeros((2, 1536), dtype=torch.uint8),
           torch.ones((2, 96), dtype=torch.float16))
    groups = _xpu_group_shards({0: a, 1: b, 2: iq4}, [0, 1, 2])
    assert [(r[0], size) for r, size in groups] == [("iq3_s", 4), ("iq4", 2)]


def test_iq3_s_group_output_slice_and_moe_dispatch(monkeypatch):
    calls = []
    def fake_kernel(x, qs, qh, signs, scale, out):
        calls.append((x.shape[0], out.stride(0)))
        out.fill_(7)
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_iq3_s", fake_kernel)
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_iq3_s_m", fake_kernel)
    rep = ("iq3_s",) + _xpu_repack_iq3_s(_make_iq3_s_bytes(rows=4))
    assert _xpu_groups_m_ok([(rep, 4)])
    storage = torch.zeros((3, 10), dtype=torch.float16)
    _xpu_rep_gemv_into(torch.zeros((1, 3072)), rep, storage[:1, 2:6])
    assert _xpu_rep_gemv_m_into(torch.zeros((3, 3072)), rep, storage[:, 2:6])
    assert calls == [(1, 10), (3, 10)]
    assert torch.all(storage[:, 2:6] == 7)
    assert torch.all(storage[:, :2] == 0) and torch.all(storage[:, 6:] == 0)
    assert torch.all(gguf_xpu._xpu_rep_gemv(torch.zeros((1, 3072)), rep) == 7)
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_iq3_s_m", None)
    assert not _xpu_groups_m_ok([(rep, 4)])
    assert not _xpu_rep_gemv_m_into(torch.zeros((3, 3072)), rep, storage[:, 2:6])


@pytest.mark.parametrize("m", [1, 4, 17])
def test_iq3_s_shard_matmul_dispatch(monkeypatch, m):
    rep = ("iq3_s",) + _xpu_repack_iq3_s(_make_iq3_s_bytes(rows=3))
    dense = _xpu_dequant_iq3_s(*rep[1:], torch.float16)
    def fake_kernel(x, qs, qh, signs, scale, out):
        out.copy_(x @ dense.t())
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_iq3_s", fake_kernel)
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_iq3_s_m", fake_kernel)
    monkeypatch.setattr(gguf_xpu, "_NO_KQUANT_M", False)
    x = torch.randn((m, 3072), dtype=torch.float16)
    torch.testing.assert_close(gguf_xpu._xpu_shard_matmul(x, rep), x @ dense.t())
