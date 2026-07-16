import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import memory_pool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKVBuffer(unittest.TestCase):
    def test_xpu_fused_scatter_rejects_strided_source_rows(self):
        tokens, row_dim = 3, 32
        packed = (
            torch.arange(tokens * row_dim * 3)
            .reshape(tokens, row_dim * 3)
            .to(torch.bfloat16)
        )
        k = packed[:, row_dim : 2 * row_dim]
        v = packed[:, 2 * row_dim :]
        indices = torch.tensor([4, 7, 9], dtype=torch.int64)
        k_cache = torch.zeros(16, row_dim, dtype=packed.dtype)
        v_cache = torch.zeros_like(k_cache)

        def fused_scatter(*_args):
            self.fail("Fused scatter must not receive strided source rows")

        with patch.multiple(
            memory_pool,
            _is_cpu=False,
            _is_cuda=False,
            _is_hip=False,
            _is_xpu=True,
            _esimd_kv_scatter=fused_scatter,
        ):
            memory_pool._set_kv_buffer_impl(
                k,
                v,
                k_cache,
                v_cache,
                indices,
                row_dim=row_dim,
                store_dtype=packed.dtype,
                device_module=torch,
                size_limit=len(k_cache),
            )

        torch.testing.assert_close(k_cache[indices], k)
        torch.testing.assert_close(v_cache[indices], v)


if __name__ == "__main__":
    unittest.main()
