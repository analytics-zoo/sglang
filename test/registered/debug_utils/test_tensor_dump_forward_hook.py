import unittest

import torch
from torch import nn

from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import LinearBase
from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import add_prefix
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=9,
    stage="base-b",
    runner_config="1-gpu-small",
    disabled="Test uses pytest-style function without TestCase class - see #17145",
)
register_amd_ci(
    est_time=15,
    suite="stage-b-test-1-gpu-small-amd",
    disabled="Test uses pytest-style function without TestCase class - see #17145",
)

TEST_HIDDEN_SIZE = 32


class SimpleModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = TEST_HIDDEN_SIZE
        self.rms_norm_eps = 1e-5
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.hidden_size,
            hidden_act="silu",
            quant_config=None,
            prefix=add_prefix("mlp", ""),
        )
        self.layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class MockCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleModel()

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model(hidden_states)


class KeywordBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor, *, scale: torch.Tensor):
        return self.proj(hidden_states) * scale


class AllIOMockCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = KeywordBlock()
        self.lm_head = nn.Linear(4, 8, bias=False)

    def forward(self, hidden_states: torch.Tensor, *, scale: torch.Tensor):
        hidden_states = self.model(hidden_states, scale=scale)
        return self.lm_head(hidden_states)


def init_weights(module):
    if isinstance(module, LinearBase):
        torch.nn.init.uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, RMSNorm):
        torch.nn.init.ones_(module.weight)


def test_model_forward_dump(tmp_path):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:2646",
    )
    initialize_model_parallel()
    model = MockCausalLM()
    model.apply(init_weights)
    model = model.cuda().bfloat16()
    dumper = register_forward_hook_for_model(
        model, tmp_path / "sglang_dump", [0], 0, 0, 0
    )

    dir_path = dumper.get_dump_dir()
    inp = torch.randn(4, TEST_HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01
    result = model(inp.cuda())
    data = torch.load(f"{dir_path}/Pass00000.pt")
    assert "model.layernorm" in data
    assert "model.mlp.down_proj" in data
    assert torch.allclose(
        data["model.mlp.down_proj"], result.cpu(), rtol=1e-5, atol=1e-5
    )


def test_model_all_io_dump(tmp_path):
    model = AllIOMockCausalLM()
    dumper = register_forward_hook_for_model(
        model,
        tmp_path / "all_io_dump",
        dump_layers=None,
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        mode="all_io",
    )

    hidden_states = torch.randn(2, 4)
    scale = torch.tensor(2.0)
    # ModelRunner calls the root forward method directly rather than nn.Module.__call__.
    result = model.forward(hidden_states, scale=scale)
    data = torch.load(f"{dumper.get_dump_dir()}/Pass00000.pt", weights_only=False)

    assert torch.equal(data["__root__.input.args.0"], hidden_states)
    assert torch.equal(data["model.input.kwargs.scale"], scale)
    assert torch.equal(data["__root__.output"], result)

    keys = list(data)
    assert keys.index("model.input.args.0") < keys.index("model.proj.input.args.0")
    assert keys.index("model.proj.output") < keys.index("model.output")
    assert keys.index("lm_head.output") < keys.index("__root__.output")


def test_model_all_io_dump_last_token_only(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSOR_DUMP_LAST_TOKEN_ONLY", "1")
    model = AllIOMockCausalLM()
    dumper = register_forward_hook_for_model(
        model,
        tmp_path / "last_token_dump",
        dump_layers=None,
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        mode="all_io",
    )

    hidden_states = torch.randn(3, 4)
    result = model.forward(hidden_states, scale=torch.tensor(2.0))
    data = torch.load(f"{dumper.get_dump_dir()}/Pass00000.pt", weights_only=False)

    assert torch.equal(data["__root__.input.args.0"], hidden_states[-1:])
    assert torch.equal(data["__root__.output"], result[-1:])


if __name__ == "__main__":
    unittest.main()
