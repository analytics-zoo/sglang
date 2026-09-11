import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType
from gguf import dequantize

from sglang.srt.layers.quantization.gguf import (
    _q5q6_col_perm_elems,
    _xpu_dequant_q3_k,
    _xpu_repack_q3_k,
    _xpu_repack_rows_chunked,
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
