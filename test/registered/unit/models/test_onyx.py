import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.configs.model_config import (
    get_hybrid_layer_ids,
    is_hybrid_swa_model,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _xpu_fp8_esimd_shape_qualified,
    _xpu_fp8_w8a16_shape_qualified,
)
from sglang.srt.models.onyx import (
    ONYX_QUERY_SCALE_FACTOR,
    OnyxLogitsProcessor,
    OnyxNormalizedEmbedding,
    OnyxOffsetRMSNorm,
    OnyxRMSNorm,
    OnyxScalelessRMSNorm,
    OnyxForCausalLM,
    get_onyx_layer_types,
    get_onyx_query_scale,
    get_onyx_sliding_window,
    onyx_layer_uses_rope,
)
from sglang.srt.models.onyx_vision import OnyxVisionEncoder
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_config(num_hidden_layers=8):
    pattern = [2048, 2048, 2048, 0]
    count_backward_offset = len(pattern) - num_hidden_layers % len(pattern)
    windows = [
        pattern[(layer_id + count_backward_offset) % len(pattern)]
        for layer_id in range(num_hidden_layers)
    ]
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        head_dim=128,
        sliding_window=2048,
        sliding_window_pattern=pattern,
        every_n_layers_nope=4,
        layer_types=[
            "sliding_attention" if window > 0 else "full_attention"
            for window in windows
        ],
        no_rope_layers=[
            (num_hidden_layers - layer_id - 1) % 4 != 0
            for layer_id in range(num_hidden_layers)
        ],
        is_hybrid_swa=True,
    )


class TestOnyxArchitectureSemantics(unittest.TestCase):
    def test_vision_encoder_output_shape_and_dtype(self):
        config = SimpleNamespace(
            vision_latent_dim=32,
            vision_heads=4,
            vision_mlp_ratio=2.0,
            vision_patch_temporal=2,
            vision_patch_size=14,
            vision_downsample_factor=2,
            vision_sparse_attention_factor=2,
            vision_pos_emb_grid_h=4,
            vision_pos_emb_grid_w=4,
            vision_layers=3,
        )
        encoder = OnyxVisionEncoder(config)
        output = encoder([torch.randn(3, 56, 56)])
        self.assertEqual(output.shape, (4, 128))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_backward_aligned_hybrid_pattern(self):
        config = make_config(num_hidden_layers=10)
        self.assertEqual(
            get_onyx_layer_types(config),
            [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        self.assertEqual(get_onyx_sliding_window(config, 0), 2047)
        self.assertEqual(get_onyx_sliding_window(config, 1), -1)

    def test_pattern_alignment_uses_nope_frequency(self):
        config = make_config(num_hidden_layers=10)
        config.every_n_layers_nope = 3
        config.sliding_window_pattern = [2048, 2048, 0, 2048]
        config.layer_types = None
        self.assertEqual(
            get_onyx_layer_types(config),
            [
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
        )

    def test_backward_aligned_irope_pattern(self):
        config = make_config(num_hidden_layers=10)
        rope_layers = [
            layer_id
            for layer_id in range(config.num_hidden_layers)
            if onyx_layer_uses_rope(config, layer_id)
        ]
        self.assertEqual(rope_layers, [0, 2, 3, 4, 6, 7, 8])

    def test_hybrid_model_config_integration(self):
        config = make_config()
        self.assertTrue(is_hybrid_swa_model(["OnyxForCausalLM"], config))
        swa_ids, full_ids = get_hybrid_layer_ids(["OnyxForCausalLM"], config)
        self.assertEqual(swa_ids, [0, 1, 2, 4, 5, 6])
        self.assertEqual(full_ids, [3, 7])

    def test_query_scale(self):
        config = make_config()
        self.assertEqual(
            get_onyx_query_scale(config),
            ONYX_QUERY_SCALE_FACTOR / math.sqrt(config.head_dim),
        )

    def test_norm_scale_conventions(self):
        x = torch.tensor([[1.0, 2.0, -3.0, 4.0]], dtype=torch.float32)
        weight = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32)

        offset_norm = OnyxOffsetRMSNorm(4, eps=1e-5)
        offset_norm.weight.data.copy_(weight)
        torch.testing.assert_close(
            offset_norm.forward_native(x),
            F.rms_norm(x, (4,), eps=1e-5) * (1 + weight),
        )

        final_norm = OnyxRMSNorm(4, eps=1e-5)
        final_norm.weight.data.copy_(weight)
        torch.testing.assert_close(
            final_norm.forward_native(x),
            F.rms_norm(x, (4,), eps=1e-5) * weight,
        )

        scaleless_norm = OnyxScalelessRMSNorm(4, eps=1e-5)
        torch.testing.assert_close(
            scaleless_norm.forward_native(x),
            F.rms_norm(x, (4,), eps=1e-5),
        )

    def test_multimodal_text_embedding_is_normalized_before_merge(self):
        embedding = torch.nn.Embedding(3, 4)
        embedding.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 4.0, 6.0, 8.0],
                    [1.0, -1.0, 1.0, -1.0],
                ]
            )
        )
        norm = torch.nn.RMSNorm(4, eps=1e-5, elementwise_affine=False)
        normalized_embedding = OnyxNormalizedEmbedding(embedding, norm)
        output = normalized_embedding(torch.tensor([0, 2]))
        torch.testing.assert_close(
            output,
            F.rms_norm(embedding(torch.tensor([0, 2])), (4,), eps=1e-5),
        )
        self.assertEqual(normalized_embedding.num_embeddings, 3)
        self.assertEqual(normalized_embedding.embedding_dim, 4)
        self.assertIs(normalized_embedding.weight, embedding.weight)

    def test_logits_processor_promotes_lm_head_output_to_fp32(self):
        processor = object.__new__(OnyxLogitsProcessor)
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        lm_head = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        with patch.object(
            LogitsProcessor,
            "_compute_lm_head",
            return_value=torch.randn(2, 8, dtype=torch.bfloat16),
        ):
            logits = processor._compute_lm_head(hidden_states, lm_head)
        self.assertEqual(logits.dtype, torch.float32)

    def test_lm_head_is_excluded_from_online_quantization(self):
        config = SimpleNamespace(
            vocab_size=16,
            hidden_size=8,
            final_logit_softcapping=20.0,
            logits_scaling=1.0,
        )
        quant_config = object()
        captured = {}

        def make_lm_head(*args, **kwargs):
            captured.update(kwargs)
            return torch.nn.Identity()

        with (
            patch(
                "sglang.srt.models.onyx.get_pp_group",
                return_value=SimpleNamespace(is_last_rank=True),
            ),
            patch(
                "sglang.srt.models.onyx.OnyxModel",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "sglang.srt.models.onyx.ParallelLMHead",
                side_effect=make_lm_head,
            ),
            patch(
                "sglang.srt.models.onyx.OnyxLogitsProcessor",
                return_value=SimpleNamespace(),
            ),
        ):
            OnyxForCausalLM(config, quant_config=quant_config)

        self.assertIsNone(captured["quant_config"])

    def test_weight_reload_invalidates_optimization_caches(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(4))
        model.weight._esimd_t = torch.ones(4)
        model.weight._esimd_1d = torch.ones(1)
        model.layer = torch.nn.Module()
        model.layer._post_attn_norm_weight = torch.ones(4)
        model.layer._pre_ffn_norm_weight = torch.ones(4)
        model.layer._post_ffn_norm_weight = torch.ones(4)

        OnyxForCausalLM._invalidate_optimization_caches(model)

        self.assertFalse(hasattr(model.weight, "_esimd_t"))
        self.assertFalse(hasattr(model.weight, "_esimd_1d"))
        self.assertFalse(hasattr(model.layer, "_post_attn_norm_weight"))
        self.assertFalse(hasattr(model.layer, "_pre_ffn_norm_weight"))
        self.assertFalse(hasattr(model.layer, "_post_ffn_norm_weight"))

    def test_fp8_validation_ignores_vision_projection_names(self):
        class FakeFp8LinearMethod:
            pass

        model = torch.nn.Module()
        model.config = SimpleNamespace(num_hidden_layers=0)
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(1, 1, dtype=torch.float16)
        model.model.vision_encoder = torch.nn.Module()
        model.model.vision_encoder.o_proj = torch.nn.Linear(1, 1)
        model.lm_head = torch.nn.Linear(1, 1, dtype=torch.float16)

        with patch(
            "sglang.srt.layers.quantization.fp8.Fp8LinearMethod",
            FakeFp8LinearMethod,
        ):
            OnyxForCausalLM.validate_online_fp8_weights(model)

    def test_fp8_fast_path_shape_qualification(self):
        self.assertTrue(_xpu_fp8_esimd_shape_qualified(1, 9984, 6656))
        self.assertTrue(_xpu_fp8_esimd_shape_qualified(2, 9984, 6656))
        self.assertTrue(_xpu_fp8_esimd_shape_qualified(64, 9984, 6656))
        self.assertTrue(_xpu_fp8_esimd_shape_qualified(32, 6656, 19968))
        self.assertFalse(_xpu_fp8_esimd_shape_qualified(64, 6656, 19968))
        self.assertFalse(_xpu_fp8_w8a16_shape_qualified(128, 6656, 2048))
        self.assertTrue(_xpu_fp8_w8a16_shape_qualified(256, 6656, 2048))
        self.assertTrue(_xpu_fp8_w8a16_shape_qualified(65, 6656, 2304))

    def test_tp2_linear_weight_sharding(self):
        qkv = QKVParallelLinear(
            16,
            4,
            4,
            2,
            bias=False,
            tp_rank=1,
            tp_size=2,
        )
        q_weight = torch.arange(16 * 16).reshape(16, 16).float()
        k_weight = torch.arange(8 * 16).reshape(8, 16).float()
        v_weight = k_weight + 1000
        qkv.weight_loader(qkv.weight, q_weight, "q")
        qkv.weight_loader(qkv.weight, k_weight, "k")
        qkv.weight_loader(qkv.weight, v_weight, "v")
        torch.testing.assert_close(
            qkv.weight,
            torch.cat((q_weight[8:16], k_weight[4:8], v_weight[4:8])),
        )

        output_gate = ColumnParallelLinear(
            16,
            16,
            bias=False,
            tp_rank=1,
            tp_size=2,
        )
        gate_weight = torch.arange(16 * 16).reshape(16, 16).float()
        output_gate.weight_loader(output_gate.weight, gate_weight)
        torch.testing.assert_close(output_gate.weight, gate_weight[8:16])

        gate_up = MergedColumnParallelLinear(
            16,
            [24, 24],
            bias=False,
            tp_rank=1,
            tp_size=2,
        )
        mlp_gate = torch.arange(24 * 16).reshape(24, 16).float()
        mlp_up = mlp_gate + 1000
        gate_up.weight_loader(gate_up.weight, mlp_gate, 0)
        gate_up.weight_loader(gate_up.weight, mlp_up, 1)
        torch.testing.assert_close(
            gate_up.weight,
            torch.cat((mlp_gate[12:24], mlp_up[12:24])),
        )

    def test_hf_split_weight_names_route_to_packed_parameters(self):
        class LoaderHarness(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.start_layer = 0
                self.model.end_layer = 1
                self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
                layer = self.model.layers[0]
                layer.self_attn = torch.nn.Module()
                layer.self_attn.qkv_proj = torch.nn.Module()
                layer.self_attn.qkv_proj.weight = torch.nn.Parameter(torch.zeros(1))
                layer.mlp = torch.nn.Module()
                layer.mlp.gate_up_proj = torch.nn.Module()
                layer.mlp.gate_up_proj.weight = torch.nn.Parameter(torch.zeros(1))
                self.pp_group = SimpleNamespace(
                    is_first_rank=True,
                    is_last_rank=True,
                )
                self.qkv_calls = []
                self.mlp_calls = []
                layer.self_attn.qkv_proj.weight.weight_loader = (
                    lambda _param, _weight, shard_id: self.qkv_calls.append(shard_id)
                )
                layer.mlp.gate_up_proj.weight.weight_loader = (
                    lambda _param, _weight, shard_id: self.mlp_calls.append(shard_id)
                )

        harness = LoaderHarness()
        loaded = OnyxForCausalLM.load_weights(
            harness,
            [
                ("model.layers.0.self_attn.q_proj.weight", torch.ones(1)),
                ("model.layers.0.self_attn.k_proj.weight", torch.ones(1)),
                ("model.layers.0.self_attn.v_proj.weight", torch.ones(1)),
                ("model.layers.0.mlp.gate_proj.weight", torch.ones(1)),
                ("model.layers.0.mlp.up_proj.weight", torch.ones(1)),
                ("model.rotary_emb.freqs", torch.ones(1)),
                ("model.vision_encoder.patch_embedding.weight", torch.ones(1)),
                ("model.vision_adapter.c_fc.weight", torch.ones(1)),
                ("model.vision_projection.weight", torch.ones(1)),
            ],
        )
        self.assertEqual(harness.qkv_calls, ["q", "k", "v"])
        self.assertEqual(harness.mlp_calls, [0, 1])
        self.assertEqual(
            loaded,
            {
                "model.layers.0.self_attn.qkv_proj.weight",
                "model.layers.0.mlp.gate_up_proj.weight",
            },
        )

    def test_vision_qkv_weights_are_not_remapped_to_text_qkv(self):
        class LoaderHarness(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.has_vision = True
                self.model.start_layer = 0
                self.model.end_layer = 1
                self.model.vision_encoder = torch.nn.Module()
                self.model.vision_encoder.transformer = torch.nn.ModuleList(
                    [torch.nn.Module()]
                )
                block = self.model.vision_encoder.transformer[0]
                block.attn = torch.nn.Module()
                block.attn.q_proj = torch.nn.Linear(2, 2, bias=False)
                self.pp_group = SimpleNamespace(
                    is_first_rank=True,
                    is_last_rank=True,
                )

        harness = LoaderHarness()
        weight = torch.arange(4, dtype=torch.float32).view(2, 2)
        loaded = OnyxForCausalLM.load_weights(
            harness,
            [
                (
                    "model.vision_encoder.transformer.0.attn.q_proj.weight",
                    weight,
                )
            ],
        )
        torch.testing.assert_close(
            harness.model.vision_encoder.transformer[0].attn.q_proj.weight,
            weight,
        )
        self.assertEqual(
            loaded,
            {"model.vision_encoder.transformer.0.attn.q_proj.weight"},
        )


if __name__ == "__main__":
    unittest.main()
