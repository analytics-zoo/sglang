import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType
from gguf import dequantize

import sglang.srt.layers.quantization.gguf as gguf_xpu
from sglang.srt.layers.quantization.gguf import (
    _pack_nibble_interleaved,
    _q5q6_col_perm_elems,
    _xpu_dequant_iq4,
    _xpu_group_shards,
    _xpu_perm_rep_rows,
    _xpu_prepare_shard,
    _xpu_repack_iq4_nl,
    _xpu_repack_iq4_xs,
    _xpu_repack_rows_chunked,
    _xpu_try_merge_shards,
)


_COL_PERM = (3, 8, 128)


def _make_iq4_bytes(kind: str, rows: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260910)
    k = np.prod(_COL_PERM)
    block_bytes = 18 if kind == "nl" else 136
    block_size = 32 if kind == "nl" else 256
    qweight = torch.randint(
        0,
        256,
        (rows, k // block_size, block_bytes),
        dtype=torch.uint8,
        generator=generator,
    )
    # Random half bit patterns include NaN/Inf.  Keep the quant payload random
    # but install a finite, exactly representable scale in every block.
    d = torch.tensor([0.00390625], dtype=torch.float16).view(torch.uint8)
    qweight[:, :, :2] = d
    return qweight.view(rows, -1).contiguous()


def _reference(qweight: torch.Tensor, qtype: WeightType) -> torch.Tensor:
    raw = np.array(qweight.numpy(), copy=True)
    return torch.from_numpy(dequantize(raw, qtype))


@pytest.mark.parametrize(
    "kind,qtype,repack,max_abs",
    [
        ("nl", WeightType.IQ4_NL, _xpu_repack_iq4_nl, 0.0),
        ("xs", WeightType.IQ4_XS, _xpu_repack_iq4_xs, 0.0),
    ],
)
def test_iq4_canonical_dequant_matches_gguf_reference(
    kind, qtype, repack, max_abs
):
    qweight = _make_iq4_bytes(kind)
    packed, scale = repack(qweight)
    actual = _xpu_dequant_iq4(packed, scale, torch.float32)
    expected = _reference(qweight, qtype)

    assert packed.shape == (qweight.shape[0], actual.shape[1] // 2)
    assert scale.shape == (qweight.shape[0], actual.shape[1] // 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=max_abs)


@pytest.mark.parametrize(
    "kind,repack",
    [("nl", _xpu_repack_iq4_nl), ("xs", _xpu_repack_iq4_xs)],
)
def test_iq4_col_perm_matches_dense_reference(kind, repack):
    qweight = _make_iq4_bytes(kind)
    packed, scale = repack(qweight)
    perm_packed, perm_scale = repack(qweight, col_perm=_COL_PERM)

    lo = packed & 0x0F
    hi = packed >> 4
    indices = torch.stack((lo, hi), dim=2).view(qweight.shape[0], -1)
    expected_indices = _q5q6_col_perm_elems(indices, _COL_PERM)
    torch.testing.assert_close(
        perm_packed,
        _pack_nibble_interleaved(expected_indices),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        perm_scale,
        _q5q6_col_perm_elems(scale, (3, 8, 4)),
        rtol=0,
        atol=0,
    )

    expected_dense = _q5q6_col_perm_elems(
        _xpu_dequant_iq4(packed, scale, torch.float32), _COL_PERM
    )
    actual_dense = _xpu_dequant_iq4(
        perm_packed, perm_scale, torch.float32
    )
    torch.testing.assert_close(actual_dense, expected_dense, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind,repack",
    [("nl", _xpu_repack_iq4_nl), ("xs", _xpu_repack_iq4_xs)],
)
def test_iq4_row_chunking_matches_whole_repack(kind, repack):
    qweight = _make_iq4_bytes(kind, rows=5)
    expected = repack(qweight, col_perm=_COL_PERM)
    actual = _xpu_repack_rows_chunked(
        repack, qweight, chunk_rows=2, col_perm=_COL_PERM
    )
    for chunked, whole in zip(actual, expected):
        torch.testing.assert_close(chunked, whole, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind,repack",
    [("nl", _xpu_repack_iq4_nl), ("xs", _xpu_repack_iq4_xs)],
)
@pytest.mark.parametrize(
    "col_perm,match",
    [
        ((3, 8, 127), "divisible by 32"),
        ((3, 7, 128), "does not match K"),
    ],
)
def test_iq4_col_perm_rejects_invalid_shape(kind, repack, col_perm, match):
    with pytest.raises(ValueError, match=match):
        repack(_make_iq4_bytes(kind, rows=1), col_perm=col_perm)


@pytest.mark.parametrize(
    "kind,repack,width",
    [
        ("nl", _xpu_repack_iq4_nl, 17),
        ("xs", _xpu_repack_iq4_xs, 135),
    ],
)
def test_iq4_repack_rejects_partial_block(kind, repack, width):
    with pytest.raises(ValueError, match=f"invalid IQ4_{kind.upper()}"):
        repack(torch.zeros((1, width), dtype=torch.uint8))


@pytest.mark.parametrize(
    "kind,qtype,repack",
    [
        ("nl", WeightType.IQ4_NL, _xpu_repack_iq4_nl),
        ("xs", WeightType.IQ4_XS, _xpu_repack_iq4_xs),
    ],
)
def test_prepare_iq4_native_and_type_specific_fallback(
    monkeypatch, kind, qtype, repack
):
    qweight = _make_iq4_bytes(kind)
    monkeypatch.setattr(gguf_xpu, "esimd_gemv_iq4", lambda *args: None)
    monkeypatch.delenv("SGLANG_GGUF_XPU_FORCE_DEQUANT", raising=False)
    monkeypatch.delenv("SGLANG_GGUF_XPU_NO_IQ4", raising=False)

    rep = _xpu_prepare_shard(qweight, int(qtype), torch.float16)
    assert rep[0] == "iq4"
    expected = repack(qweight)
    for actual_tensor, expected_tensor in zip(rep[1:], expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    monkeypatch.setenv("SGLANG_GGUF_XPU_NO_IQ4", "1")
    fallback = _xpu_prepare_shard(qweight, int(qtype), torch.float16)
    assert fallback[0] == "fp16"
    torch.testing.assert_close(
        fallback[1], _reference(qweight, qtype).to(torch.float16), rtol=0, atol=0
    )


def test_iq4_row_permute_and_same_kind_merge():
    packed = torch.arange(24, dtype=torch.uint8).reshape(4, 6)
    scale = torch.arange(4, dtype=torch.float16).reshape(4, 1)
    rep = ("iq4", packed, scale)
    perm = torch.tensor([3, 1, 0, 2])
    permuted = _xpu_perm_rep_rows(rep, perm)
    torch.testing.assert_close(permuted[1], packed[perm], rtol=0, atol=0)
    torch.testing.assert_close(permuted[2], scale[perm], rtol=0, atol=0)

    merged, sizes = _xpu_try_merge_shards(
        {0: ("iq4", packed[:1], scale[:1]),
         1: ("iq4", packed[1:], scale[1:])},
        [0, 1],
    )
    assert sizes == [1, 3]
    torch.testing.assert_close(merged[1], packed, rtol=0, atol=0)
    torch.testing.assert_close(merged[2], scale, rtol=0, atol=0)


def test_iq4_participates_in_mixed_kind_group():
    q4_a = ("q4_k", torch.zeros((1, 8), dtype=torch.uint8),
            torch.ones((1, 1), dtype=torch.float16),
            torch.zeros((1, 1), dtype=torch.float16))
    q4_b = ("q4_k", torch.ones((2, 8), dtype=torch.uint8),
            torch.ones((2, 1), dtype=torch.float16),
            torch.zeros((2, 1), dtype=torch.float16))
    iq4 = ("iq4", torch.zeros((3, 8), dtype=torch.uint8),
           torch.ones((3, 1), dtype=torch.float16))
    groups = _xpu_group_shards({0: q4_a, 1: q4_b, 2: iq4}, [0, 1, 2])
    assert [rep[0] for rep, _ in groups] == ["q4_k", "iq4"]
    assert [size for _, size in groups] == [3, 3]
