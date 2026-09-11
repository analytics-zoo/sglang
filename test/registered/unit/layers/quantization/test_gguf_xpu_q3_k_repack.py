import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType
from gguf import dequantize

import sglang.srt.layers.quantization.gguf as gguf_xpu
from sglang.srt.layers.quantization.gguf import (
    _q5q6_col_perm_elems,
    _xpu_dequant_q3_k,
    _xpu_dequant_rep,
    _xpu_dequant_rep_to_fp16,
    _xpu_group_shards,
    _xpu_groups_m_ok,
    _xpu_perm_rep_rows,
    _xpu_prepare_shard,
    _xpu_rep_gemv_into,
    _xpu_rep_gemv_m_into,
    _xpu_repack_q3_k,
    _xpu_repack_rows_chunked,
    _xpu_try_merge_shards,
)


_COL_PERM = (3, 8, 128)
_BLOCK_BYTES = 110
_BLOCK_SIZE = 256


def _make_q3_k_bytes(rows: int = 3) -> torch.Tensor:
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
    qweight[:, :, 108:110] = d
    return qweight.view(rows, -1).contiguous()


def _reference(qweight: torch.Tensor) -> torch.Tensor:
    raw = np.array(qweight.numpy(), copy=True)
    return torch.from_numpy(dequantize(raw, WeightType.Q3_K))


def test_q3_k_canonical_dequant_matches_gguf_reference():
    qweight = _make_q3_k_bytes()
    ql, qh, scale = _xpu_repack_q3_k(qweight)
    actual = _xpu_dequant_q3_k(ql, qh, scale, torch.float32)
    expected = _reference(qweight)

    assert ql.shape == (qweight.shape[0], actual.shape[1] // 4)
    assert qh.shape == (qweight.shape[0], actual.shape[1] // 8)
    assert scale.shape == (qweight.shape[0], actual.shape[1] // 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-5)


def test_q3_k_col_perm_matches_dense_reference():
    qweight = _make_q3_k_bytes()
    ql, qh, scale = _xpu_repack_q3_k(qweight)
    pql, pqh, pscale = _xpu_repack_q3_k(qweight, col_perm=_COL_PERM)

    expected = _q5q6_col_perm_elems(
        _xpu_dequant_q3_k(ql, qh, scale, torch.float32), _COL_PERM
    )
    actual = _xpu_dequant_q3_k(pql, pqh, pscale, torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_q3_k_row_chunking_matches_whole_repack():
    qweight = _make_q3_k_bytes(rows=5)
    expected = _xpu_repack_q3_k(qweight, col_perm=_COL_PERM)
    actual = _xpu_repack_rows_chunked(
        _xpu_repack_q3_k,
        qweight,
        chunk_rows=2,
        col_perm=_COL_PERM,
    )
    for chunked, whole in zip(actual, expected):
        torch.testing.assert_close(chunked, whole, rtol=0, atol=0)


@pytest.mark.parametrize(
    "col_perm,match",
    [
        ((3, 8, 127), "divisible by 16"),
        ((3, 7, 128), "does not match K"),
    ],
)
def test_q3_k_col_perm_rejects_invalid_shape(col_perm, match):
    with pytest.raises(ValueError, match=match):
        _xpu_repack_q3_k(
            _make_q3_k_bytes(rows=1), col_perm=col_perm
        )


def test_q3_k_repack_rejects_partial_block():
    with pytest.raises(ValueError, match="invalid Q3_K"):
        _xpu_repack_q3_k(torch.zeros((1, 109), dtype=torch.uint8))


def test_prepare_q3_k_native_and_type_specific_fallback(monkeypatch):
    qweight = _make_q3_k_bytes()
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_q3_k", lambda *args: None)
    monkeypatch.delenv("SGLANG_GGUF_XPU_FORCE_DEQUANT", raising=False)
    monkeypatch.delenv("SGLANG_GGUF_XPU_NO_Q3K", raising=False)

    rep = _xpu_prepare_shard(qweight, int(WeightType.Q3_K), torch.float16)
    assert rep[0] == "q3_k"
    expected = _xpu_repack_q3_k(qweight)
    for actual_tensor, expected_tensor in zip(rep[1:], expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    dense = _xpu_dequant_q3_k(*expected, torch.float16)
    torch.testing.assert_close(
        _xpu_dequant_rep_to_fp16(rep, torch.float16), dense, rtol=0, atol=0
    )
    torch.testing.assert_close(
        _xpu_dequant_rep(rep, torch.float16), dense, rtol=0, atol=0
    )

    monkeypatch.setenv("SGLANG_GGUF_XPU_NO_Q3K", "1")
    fallback = _xpu_prepare_shard(
        qweight, int(WeightType.Q3_K), torch.float16
    )
    assert fallback[0] == "fp16"
    torch.testing.assert_close(
        fallback[1], _reference(qweight).to(torch.float16), rtol=0, atol=0
    )


def test_q3_k_row_permute_and_same_kind_merge():
    ql = torch.arange(32, dtype=torch.uint8).reshape(4, 8)
    qh = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
    scale = torch.arange(8, dtype=torch.float16).reshape(4, 2)
    rep = ("q3_k", ql, qh, scale)
    perm = torch.tensor([3, 1, 0, 2])
    permuted = _xpu_perm_rep_rows(rep, perm)
    for actual, expected in zip(permuted[1:], (ql[perm], qh[perm], scale[perm])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    merged, sizes = _xpu_try_merge_shards(
        {
            0: ("q3_k", ql[:1], qh[:1], scale[:1]),
            1: ("q3_k", ql[1:], qh[1:], scale[1:]),
        },
        [0, 1],
    )
    assert sizes == [1, 3]
    for actual, expected in zip(merged[1:], (ql, qh, scale)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_q3_k_participates_in_mixed_kind_group():
    q3_a = (
        "q3_k",
        torch.zeros((1, 8), dtype=torch.uint8),
        torch.zeros((1, 4), dtype=torch.uint8),
        torch.ones((1, 2), dtype=torch.float16),
    )
    q3_b = (
        "q3_k",
        torch.ones((2, 8), dtype=torch.uint8),
        torch.ones((2, 4), dtype=torch.uint8),
        torch.ones((2, 2), dtype=torch.float16),
    )
    iq4 = (
        "iq4",
        torch.zeros((3, 16), dtype=torch.uint8),
        torch.ones((3, 1), dtype=torch.float16),
    )
    groups = _xpu_group_shards({0: q3_a, 1: q3_b, 2: iq4}, [0, 1, 2])
    assert [rep[0] for rep, _ in groups] == ["q3_k", "iq4"]
    assert [size for _, size in groups] == [3, 3]


def test_q3_k_group_output_slice_dispatch(monkeypatch):
    calls = []

    def fake_q3(input, ql, qh, scale, output):
        calls.append((input.shape[0], output.stride(0)))
        output.fill_(7)

    monkeypatch.setattr(gguf_xpu, "esimd_gemv_q3_k", fake_q3)
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_q3_k_m", fake_q3)
    rep = (
        "q3_k",
        torch.zeros((4, 8), dtype=torch.uint8),
        torch.zeros((4, 4), dtype=torch.uint8),
        torch.ones((4, 2), dtype=torch.float16),
    )
    assert _xpu_groups_m_ok([(rep, 4)])

    storage = torch.zeros((3, 10), dtype=torch.float16)
    _xpu_rep_gemv_into(torch.zeros((1, 32)), rep, storage[:1, 2:6])
    assert _xpu_rep_gemv_m_into(torch.zeros((3, 32)), rep, storage[:, 2:6])
    assert calls == [(1, 10), (3, 10)]
    assert torch.all(storage[:, 2:6] == 7)
    assert torch.all(storage[:, :2] == 0)
    assert torch.all(storage[:, 6:] == 0)
