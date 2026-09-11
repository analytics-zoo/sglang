"""CPU regression for snapshot state written by the actual XPU decode wrapper.

Only the device kernel and backend copy are replaced. The model wrapper is
extracted from its source so this test does not import model/kernel dependencies.
The legacy-conv case checks Python writeback ordering, not device dtype support.
"""

import ast
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


_UNSET = object()


def _load_decode_wrapper():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/models/qwen3_5.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5GatedDeltaNet"
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_forward_xpu_esimd_gdn_decode"
    )
    namespace = {"torch": torch, "Optional": Optional, "ForwardBatch": Any}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


def _load_metadata_producer():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MambaAttnBackendBase"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_metadata"
    )
    namespace = {
        "torch": torch,
        "ForwardBatch": Any,
        "ForwardMetadata": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


def _load_forward_metadata_class():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/layers/attention/mamba/mamba2_metadata.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ForwardMetadata"
    )
    namespace = {"dataclass": dataclass, "Optional": Optional, "torch": torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[cls.name]


class TestQwen35XpuGdnTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decode = staticmethod(_load_decode_wrapper())
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _run_case(
        self,
        *,
        fused_norm,
        legacy_conv=False,
        mask=(True, False),
        metadata_flag=_UNSET,
    ):
        n, h, hv, k, v, slots = 2, 8, 24, 128, 128, 6
        dim = 2 * h * k + hv * v
        conv_dtype = torch.float32 if legacy_conv else torch.float16
        pool_conv = torch.full((slots, dim, 3), -10, dtype=conv_dtype)
        pool_ssm = torch.full((slots, hv, v, k), -20, dtype=torch.float16)
        initial_conv, initial_ssm = pool_conv.clone(), pool_ssm.clone()
        indices = torch.tensor([3, 1], dtype=torch.int32)
        batch = SimpleNamespace(
            batch_size=n,
            mamba_track_mask=None if mask is None else torch.tensor(mask),
            mamba_track_indices=torch.tensor([4, 5]),
        )
        pattern = (torch.arange(dim * 3).reshape(dim, 3) % 13).to(conv_dtype) / 8
        expected_conv = [pattern + 2, pattern + 4]
        expected_ssm_values = [7, 9]

        track_calls = []

        def track(forward_batch, conv, ssm, cache_indices):
            track_calls.append(None)
            # Checking identity prevents a temporary transposed copy from
            # receiving snapshots that never reach the real cache pool.
            self.assertIs(conv, pool_conv)
            self.assertIs(ssm, pool_ssm)
            self.assertIs(forward_batch, batch)
            if forward_batch.mamba_track_mask is None:
                return
            for row, enabled in enumerate(forward_batch.mamba_track_mask.tolist()):
                if enabled:
                    src = int(cache_indices[row])
                    dst = int(forward_batch.mamba_track_indices[row])
                    conv[dst].copy_(conv[src])
                    ssm[dst].copy_(ssm[src])

        metadata = SimpleNamespace(mamba_cache_indices=indices)
        if metadata_flag is not _UNSET:
            metadata.has_mamba_track_mask = metadata_flag
        backend = SimpleNamespace(
            forward_metadata=metadata,
            req_to_token_pool=SimpleNamespace(
                mamba2_layer_cache=lambda layer: SimpleNamespace(
                    conv=(pool_conv,), temporal=pool_ssm
                )
            ),
            _track_mamba_state_decode=track,
        )

        def kernel(qkvz, conv, weights, bias, conv_indices, a_log, dt_bias,
                   ba, ssm, ssm_indices, out, zout, *shape):
            for row, slot in enumerate(conv_indices.tolist()):
                value = expected_conv[row]
                conv[slot].copy_(value.transpose(0, 1) if legacy_conv else value)
                ssm[int(ssm_indices[row])].fill_(expected_ssm_values[row])
            out.fill_(1)
            zout.zero_()

        # Snapshot values must already be correct before either norm branch
        # returns; this also detects accidentally placing the hook after the
        # fused projection's early return.
        def check_snapshots():
            for row, source in enumerate(indices.tolist()):
                torch.testing.assert_close(pool_conv[source], expected_conv[row], rtol=0, atol=0)
                self.assertTrue(torch.all(pool_ssm[source] == expected_ssm_values[row]))
                dest = int(batch.mamba_track_indices[row])
                enabled = (
                    metadata_flag is not False
                    and mask is not None
                    and mask[row]
                )
                expected_c = expected_conv[row] if enabled else initial_conv[dest]
                expected_s = pool_ssm[source] if enabled else initial_ssm[dest]
                torch.testing.assert_close(pool_conv[dest], expected_c, rtol=0, atol=0)
                torch.testing.assert_close(pool_ssm[dest], expected_s, rtol=0, atol=0)
            # Unrelated slots must retain their original values too.
            torch.testing.assert_close(pool_conv[[0, 2]], initial_conv[[0, 2]], rtol=0, atol=0)
            torch.testing.assert_close(pool_ssm[[0, 2]], initial_ssm[[0, 2]], rtol=0, atol=0)

        def norm_projection(core, z):
            check_snapshots()
            return core.reshape(n, -1) if fused_norm else None

        model = SimpleNamespace(
            layer_id=0,
            num_k_heads=2 * h,
            num_v_heads=2 * hv,
            attn_tp_size=2,
            head_k_dim=k,
            head_v_dim=v,
            conv1d=SimpleNamespace(weight=torch.zeros(dim, 1, 4, dtype=torch.float16)),
            A_log=torch.zeros(hv, dtype=torch.float16),
            dt_bias=torch.zeros(hv, dtype=torch.float16),
            _esimd_norm_out_proj=norm_projection,
            norm=lambda core, z: core,
            out_proj=lambda core: (core, None),
        )
        extension = ModuleType("custom_esimd_kernels_sglang")
        extension.esimd_gdn_conv_fused_seq = kernel
        context = ModuleType("sglang.srt.model_executor.forward_context")
        context.get_attn_backend = lambda: SimpleNamespace(linear_attn_backend=backend)
        with patch.dict(sys.modules, {
            extension.__name__: extension,
            context.__name__: context,
        }):
            result = self.decode(
                model,
                torch.zeros(n, dim + hv * v, dtype=torch.float16),
                torch.zeros(n, 2 * hv, dtype=torch.float16),
                batch,
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (n, hv * v))
        self.assertEqual(0 if metadata_flag is False else 1, len(track_calls))
        check_snapshots()

    def test_snapshot_reads_updated_native_pool_before_both_norm_paths(self):
        for fused_norm in (True, False):
            with self.subTest(fused_norm=fused_norm):
                self._run_case(fused_norm=fused_norm)

    def test_snapshot_follows_legacy_conv_writeback_to_real_pool(self):
        for fused_norm in (True, False):
            with self.subTest(fused_norm=fused_norm):
                self._run_case(fused_norm=fused_norm, legacy_conv=True)

    def test_false_and_absent_masks_leave_tracking_slots_untouched(self):
        for fused_norm in (True, False):
            for mask in ((False, False), None):
                with self.subTest(fused_norm=fused_norm, mask=mask):
                    self._run_case(fused_norm=fused_norm, mask=mask)

    def test_decode_tracking_skips_only_an_explicit_false_metadata_flag(self):
        cases = (
            (False, (False, False)),
            (True, (True, False)),
            (None, (True, False)),
            (_UNSET, (True, False)),
        )
        for metadata_flag, mask in cases:
            with self.subTest(metadata_flag=metadata_flag):
                self._run_case(
                    fused_norm=True,
                    mask=mask,
                    metadata_flag=metadata_flag,
                )

    def test_forward_metadata_default_is_unknown(self):
        forward_metadata = _load_forward_metadata_class()
        metadata = forward_metadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([1], dtype=torch.int32),
        )
        self.assertIsNone(metadata.has_mamba_track_mask)

    def test_metadata_producer_marks_false_and_mixed_masks_explicitly(self):
        producer = _load_metadata_producer()
        backend = SimpleNamespace(
            device="cpu",
            req_to_token_pool=SimpleNamespace(
                get_mamba_indices=lambda _: torch.tensor([3, 1], dtype=torch.int32)
            ),
        )
        forward_mode = SimpleNamespace(
            is_decode_or_idle=lambda: True,
            is_extend=lambda **_: False,
        )

        for mask, expected in ((False, False), (True, True)):
            batch = SimpleNamespace(
                batch_size=2,
                req_pool_indices=torch.tensor([0, 1]),
                forward_mode=forward_mode,
                mamba_track_mask=torch.tensor([mask, False]),
            )
            with self.subTest(mask=mask):
                self.assertIs(producer(backend, batch).has_mamba_track_mask, expected)


if __name__ == "__main__":
    unittest.main()
