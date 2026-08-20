# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/xpu_communicator.py

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_xpu


def _xpu_all_reduce_inplace(x: torch.Tensor, group) -> torch.Tensor:
    """In-place all-reduce returning the same buffer. Used as the eager break
    function under XPU breakable-graph capture (the returned buffer's stable
    storage is what the next captured segment reads)."""
    dist.all_reduce(x, group=group)
    return x


class XpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not is_xpu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # oneCCL collectives cannot be captured into an XPUGraph (they replay
        # stale on the 2nd+ replay -> garbled TP>1 decode). When an XPU breakable
        # graph is capturing, run the collective EAGER as a break point between
        # captured compute segments; otherwise call it directly.
        from sglang.srt.model_executor.breakable_cuda_graph.xpu_breakable_graph import (
            is_xpu_breakable_capturing,
            run_eager_between_segments,
        )

        if is_xpu_breakable_capturing():
            return run_eager_between_segments(_xpu_all_reduce_inplace, x, self.group)
        dist.all_reduce(x, group=self.group)
        return x

    def gather(
        self, input_: torch.Tensor, rank_in_group: int, dst: int = 0, dim: int = -1
    ):
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.group
        )
        if rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor
