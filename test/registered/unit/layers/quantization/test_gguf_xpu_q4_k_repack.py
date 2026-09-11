import pytest
import torch

import sglang.srt.layers.quantization.gguf as gguf_xpu
from sglang.srt.layers.quantization.gguf import (
    _pack_nibble_interleaved,
    _q5q6_col_perm_elems,
    _xpu_dequant_q4_k,
    _xpu_repack_q4_k,
    _xpu_repack_q4_k_chunked,
)


_Q4_K_BYTES = 144
_COL_PERM = (3, 8, 128)


def _make_q4_k_bytes(rows: int = 5) -> torch.Tensor:
    """Create finite Q4_K blocks with the Qwen3.8 TP2-local K=3072."""
    ratio, nk, head_v_dim = _COL_PERM
    k = ratio * nk * head_v_dim
    blocks = k // 256
    generator = torch.Generator().manual_seed(20260910)
    qweight = torch.randint(
        0,
        256,
        (rows, blocks, _Q4_K_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )

    # The first four bytes are the fp16 dall/dmin pair. Random bit patterns can
    # encode NaN/Inf, so use finite values while leaving scales and nibbles
    # exhaustive enough for the layout test.
    dm = torch.tensor([0.125, 0.0625], dtype=torch.float16).view(torch.uint8)
    qweight[:, :, :4] = dm
    return qweight.view(rows, blocks * _Q4_K_BYTES).contiguous()


def _unpack_nibbles(ql: torch.Tensor) -> torch.Tensor:
    even = ql & 0x0F
    odd = (ql >> 4) & 0x0F
    return torch.stack((even, odd), dim=2).view(ql.shape[0], -1)


def test_q4_k_default_keeps_existing_layout():
    qweight = _make_q4_k_bytes()
    default = _xpu_repack_q4_k(qweight)
    explicit_none = _xpu_repack_q4_k(qweight, col_perm=None)
    for actual, expected in zip(default, explicit_none):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_q4_k_col_perm_matches_dense_reference():
    qweight = _make_q4_k_bytes()
    ql, scale, minv = _xpu_repack_q4_k(qweight)
    pql, pscale, pminv = _xpu_repack_q4_k(qweight, col_perm=_COL_PERM)

    expected_nibbles = _q5q6_col_perm_elems(
        _unpack_nibbles(ql), _COL_PERM
    )
    torch.testing.assert_close(
        pql, _pack_nibble_interleaved(expected_nibbles), rtol=0, atol=0
    )

    ratio, nk, head_v_dim = _COL_PERM
    scale_perm = (ratio, nk, head_v_dim // 32)
    torch.testing.assert_close(
        pscale, _q5q6_col_perm_elems(scale, scale_perm), rtol=0, atol=0
    )
    torch.testing.assert_close(
        pminv, _q5q6_col_perm_elems(minv, scale_perm), rtol=0, atol=0
    )

    dense = _xpu_dequant_q4_k(ql, scale, minv, torch.float32)
    expected_dense = _q5q6_col_perm_elems(dense, _COL_PERM)
    permuted_dense = _xpu_dequant_q4_k(pql, pscale, pminv, torch.float32)
    torch.testing.assert_close(permuted_dense, expected_dense, rtol=0, atol=0)


def test_q4_k_col_perm_chunked_matches_non_chunked():
    qweight = _make_q4_k_bytes()
    expected = _xpu_repack_q4_k(qweight, col_perm=_COL_PERM)
    actual = _xpu_repack_q4_k_chunked(
        qweight, chunk_rows=2, col_perm=_COL_PERM
    )
    for chunked, whole in zip(actual, expected):
        torch.testing.assert_close(chunked, whole, rtol=0, atol=0)


@pytest.mark.parametrize(
    "col_perm, match",
    [
        ((3, 8, 127), "divisible by 32"),
        ((3, 7, 128), "does not match K"),
    ],
)
def test_q4_k_col_perm_rejects_invalid_shape(col_perm, match):
    with pytest.raises(ValueError, match=match):
        _xpu_repack_q4_k(_make_q4_k_bytes(rows=1), col_perm=col_perm)


def test_large_fp16_fallback_does_not_cache_transpose(monkeypatch):
    gguf_xpu._fp16_wt_cache.clear()
    monkeypatch.setattr(gguf_xpu, "_FP16_WT_CACHE_MAX_BYTES", 1)
    x = torch.randn(2, 8, dtype=torch.float16)
    weight = torch.randn(4, 8, dtype=torch.float16)

    actual = gguf_xpu._xpu_shard_matmul(x, ("fp16", weight, None))

    torch.testing.assert_close(actual, torch.mm(x, weight.t()))
    assert not gguf_xpu._fp16_wt_cache


def test_small_fp16_weight_still_caches_transpose(monkeypatch):
    gguf_xpu._fp16_wt_cache.clear()
    monkeypatch.setattr(gguf_xpu, "_FP16_WT_CACHE_MAX_BYTES", 1 << 20)
    x = torch.randn(2, 8, dtype=torch.float16)
    weight = torch.randn(4, 8, dtype=torch.float16)

    actual = gguf_xpu._xpu_shard_matmul(x, ("fp16", weight, None))

    torch.testing.assert_close(actual, torch.mm(x, weight.t()))
    assert id(weight) in gguf_xpu._fp16_wt_cache
    gguf_xpu._fp16_wt_cache.clear()
