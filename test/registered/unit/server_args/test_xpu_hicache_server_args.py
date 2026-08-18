"""
Unit tests for the XPU HiCache validation guard in ServerArgs.

The guard (ServerArgs._reject_unsupported_xpu_hicache) rejects HiCache mem layouts
and IO backends that have not been validated on Intel XPU, and honours
SGLANG_XPU_ALLOW_UNVALIDATED_HICACHE=1 as an escape hatch so an unvalidated-but-
implemented path can still be measured without patching this file.

No XPU hardware is required: is_xpu() is patched, so this runs on CPU CI runners.

Usage:
python3 -m unittest test_xpu_hicache_server_args -v
"""

import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def _args(layout: str, io_backend: str) -> ServerArgs:
    """Build only the two fields the guard reads.

    The guard runs at the end of __post_init__, which also resolves the device, the
    ports and the model config -- none of which exist on a CPU-only runner. Bypassing
    __init__ keeps this a unit test of the guard rather than an integration test of
    ServerArgs, so it cannot start passing (or failing) because of an unrelated
    __post_init__ change.
    """
    server_args = ServerArgs.__new__(ServerArgs)
    server_args.hicache_mem_layout = layout
    server_args.hicache_io_backend = io_backend
    return server_args


class TestXPUHiCacheGuard(CustomTestCase):
    """Testcase: HiCache flag validation on Intel XPU.

    Cover scenarios:
    1. Validated layout/backend pairs are accepted
    2. Unvalidated layouts are rejected, naming the resolved pair
    3. io_backend=direct is rejected as unvalidated, not as unimplemented
    4. The escape hatch downgrades every rejection to a warning
    5. Non-XPU devices are never touched by the guard

    [Test Category] HiCache
    [Test Target] ServerArgs._reject_unsupported_xpu_hicache
    """

    def test_validated_pairs_are_accepted(self):
        with patch("sglang.srt.server_args.is_xpu", return_value=True):
            for layout in ("layer_first", "page_first"):
                _args(layout, "kernel")._reject_unsupported_xpu_hicache()

    def test_unvalidated_layout_is_rejected(self):
        with patch("sglang.srt.server_args.is_xpu", return_value=True):
            for layout in ("page_first_direct", "page_first_kv_split", "page_head"):
                with self.assertRaises(ValueError) as ctx:
                    _args(layout, "kernel")._reject_unsupported_xpu_hicache()
                # The message must report the RESOLVED pair. hicache_mem_layout and
                # hicache_io_backend rewrite each other before the guard runs, so a
                # message naming only one of them can describe a flag the user never
                # passed.
                self.assertIn("resolved:", str(ctx.exception))
                self.assertIn("not validated", str(ctx.exception))

    def test_direct_io_backend_is_rejected_as_unvalidated(self):
        with patch("sglang.srt.server_args.is_xpu", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                _args("layer_first", "direct")._reject_unsupported_xpu_hicache()
            msg = str(ctx.exception)
            self.assertIn("not validated on XPU", msg)
            # The direct-family ops DO exist on XPU as page-by-page torch copy_
            # fallbacks (sgl_kernel/kvcacheio.py _transfer_page_direct). Asserting the
            # wording keeps the guard from regressing back to claiming they are
            # missing, which is what an earlier revision of it wrongly said.
            self.assertIn("implemented", msg)

    def test_escape_hatch_downgrades_to_warning(self):
        env = {"SGLANG_XPU_ALLOW_UNVALIDATED_HICACHE": "1"}
        with patch("sglang.srt.server_args.is_xpu", return_value=True), patch.dict(
            "os.environ", env
        ), patch("sglang.srt.server_args.logger") as mock_logger:
            # Both rejection sites must honour the hatch, so exercise both.
            _args("page_first_direct", "direct")._reject_unsupported_xpu_hicache()
            self.assertEqual(mock_logger.warning.call_count, 2)

    def test_guard_is_inert_off_xpu(self):
        with patch("sglang.srt.server_args.is_xpu", return_value=False):
            # A CUDA user must never see an XPU message, even on a pair that XPU rejects.
            _args("page_first_direct", "direct")._reject_unsupported_xpu_hicache()


if __name__ == "__main__":
    unittest.main()
