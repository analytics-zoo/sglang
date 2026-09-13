# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""XPU graph runner for Frozen-KV MTP speculative decoding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.hardware_backend.xpu.graph_runner.xpu_graph_runner import (
    register_fake_ops,
)
from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPCudaGraphRunner,
)
from sglang.srt.utils import register_xpu_device_properties_for_dynamo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker


class XpuFrozenKVMTPGraphRunner(FrozenKVMTPCudaGraphRunner):
    """XPU graph runner for the Frozen-KV MTP recurrent draft-loop step.

    Inherits FrozenKVMTPCudaGraphRunner and applies XPU-specific setup:
    - Register fake ops for torch.compile tracing
    - Apply XPU dynamo config (suppress_errors for IGC compiler crashes)
    - Register XPU device properties for dynamo
    """

    @staticmethod
    def _apply_xpu_compile_config() -> None:
        """Apply XPU-specific torch.compile / dynamo settings.

        Called before super().__init__() so that the settings are in place.
        The critical flag is suppress_errors: when the Intel IGC compiler
        crashes with SIGFPE on certain reduction kernels (ocloc -device bmg
        returns exit code 245), dynamo falls back to eager for that subgraph
        instead of propagating the crash.
        """
        import torch._dynamo.config

        torch._dynamo.config.suppress_errors = True

    def __init__(self, frozen_kv_mtp_worker: FrozenKVMTPDraftWorker):
        # Apply XPU-specific setup before parent init
        register_fake_ops()
        self._apply_xpu_compile_config()
        register_xpu_device_properties_for_dynamo()

        super().__init__(frozen_kv_mtp_worker)

    def _init_profile_context_and_memory_record(self):
        """XPU-specific profiling context."""
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        )
        torch.xpu.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        """XPU-specific profile post-processing."""
        torch.xpu.memory._dump_snapshot("xpu_frozen_kv_mtp_graph_runner_memory.pickle")
        torch.xpu.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by XPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_xpu_time_total"
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total"
            )
            + "\n\nMemory Usage is saved to xpu_frozen_kv_mtp_graph_runner_memory.pickle\n"
        )
        logger.info(log_message)
