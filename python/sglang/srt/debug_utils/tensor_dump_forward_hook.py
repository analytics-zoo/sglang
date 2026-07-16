"""
This file provides a function `register_forward_hook_for_model` that registers a forward hook on every operator of the model.
After registration, during model inference, all tensors generated throughout the forward pass will be recorded.

Usage:
Specify the output directory for dumping tensors using the argument `--debug-tensor-dump-output-folder`.
A separate directory will be created for each GPU rank, named in the format `f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{pid}"`.
Each complete forward pass of the model generates a `.pt` file named `f"Pass{pass_num}.pt"`, which can be loaded using `torch.load`.
The file contains a series of key-value pairs, where the keys correspond to operator names in the model
(similar to those in model.safetensors.index.json), and the values are the outputs produced by the respective operators.

Set `TENSOR_DUMP_MODE=all_io` to record inputs and outputs for every module,
including parent modules, and flush after the root model returns. This mode is
intended for single-request numerical tracing rather than performance runs.
"""

import logging
import os
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)

_LEAF_OUTPUTS_MODE = "leaf_outputs"
_ALL_IO_MODE = "all_io"
_VALID_DUMP_MODES = {_LEAF_OUTPUTS_MODE, _ALL_IO_MODE}


class TensorDumper:
    def __init__(
        self,
        dump_dir: str,
        dump_layers: Optional[List[int]],
        tp_size: int,
        tp_rank: int,
        pp_rank: int,
        mode: Optional[str] = None,
    ):
        self._mode = mode or os.getenv("TENSOR_DUMP_MODE", _LEAF_OUTPUTS_MODE)
        if self._mode not in _VALID_DUMP_MODES:
            raise ValueError(
                f"Invalid tensor dump mode {self._mode!r}; "
                f"expected one of {sorted(_VALID_DUMP_MODES)}"
            )
        self._last_token_only = os.getenv("TENSOR_DUMP_LAST_TOKEN_ONLY") == "1"
        self._dump_layers = dump_layers
        self._forward_pass_id = 0
        self._pid = os.getpid()
        self._current_tensors: Dict[str, Any] = {}
        self._module_call_counts: Dict[str, int] = {}
        self._active_calls: Dict[int, List[str]] = defaultdict(list)
        self._base_dir = Path(dump_dir)
        rank = tp_size * pp_rank + tp_rank
        self._process_dir = (
            self._base_dir / f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{self._pid}"
        )
        self._process_dir.mkdir(parents=True, exist_ok=True)

    def get_dump_dir(self):
        return str(self._process_dir)

    def _snapshot_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._last_token_only and tensor.ndim >= 2 and tensor.shape[0] > 1:
            tensor = tensor[-1:]
        return tensor.detach().cpu()

    def add_tensor(self, name, tensor_item):
        if isinstance(tensor_item, (tuple, list)):
            tensors = [
                self._snapshot_tensor(tensor)
                for tensor in tensor_item
                if isinstance(tensor, torch.Tensor)
            ]
            if len(tensors) == 1:
                self._current_tensors[name] = tensors[0]
            elif len(tensors) > 1:
                self._current_tensors[name] = tensors
        elif isinstance(tensor_item, torch.Tensor):
            self._current_tensors[name] = self._snapshot_tensor(tensor_item)
        elif isinstance(tensor_item, LogitsProcessorOutput):
            self._current_tensors[name] = self._snapshot_tensor(
                tensor_item.next_token_logits
            )
        elif isinstance(tensor_item, ForwardBatch):
            self._current_tensors[name + ".forward_batch_info.input_ids"] = (
                self._snapshot_tensor(tensor_item.input_ids)
            )
            self._current_tensors[name + ".forward_batch_info.seq_lens"] = (
                self._snapshot_tensor(tensor_item.seq_lens)
            )
            self._current_tensors[name + ".forward_batch_info.positions"] = (
                self._snapshot_tensor(tensor_item.positions)
            )
        elif isinstance(tensor_item, PPProxyTensors):
            for tensor_name in tensor_item.tensors.keys():
                self._current_tensors[name + ".pp_proxy_tensors." + tensor_name] = (
                    self._snapshot_tensor(tensor_item.tensors[tensor_name])
                )
        else:
            logger.warning(f"Unsupported type: {type(tensor_item)}: {tensor_item}")

    def _add_tensor_tree(self, name: str, item: Any) -> None:
        if item is None:
            return
        if isinstance(item, torch.Tensor):
            self._current_tensors[name] = self._snapshot_tensor(item)
            return
        if isinstance(item, LogitsProcessorOutput):
            self._add_tensor_tree(
                name + ".next_token_logits", item.next_token_logits
            )
            self._add_tensor_tree(name + ".hidden_states", item.hidden_states)
            return
        if isinstance(item, ForwardBatch):
            for field_name in ("input_ids", "seq_lens", "positions"):
                self._add_tensor_tree(
                    f"{name}.forward_batch_info.{field_name}",
                    getattr(item, field_name),
                )
            return
        if isinstance(item, PPProxyTensors):
            for tensor_name, tensor in item.tensors.items():
                self._add_tensor_tree(
                    f"{name}.pp_proxy_tensors.{tensor_name}", tensor
                )
            return
        if isinstance(item, dict):
            for key, value in item.items():
                self._add_tensor_tree(f"{name}.{key}", value)
            return
        if isinstance(item, (tuple, list)):
            for index, value in enumerate(item):
                self._add_tensor_tree(f"{name}.{index}", value)

    def dump_current_tensors(self):
        if len(self._current_tensors) == 0:
            return
        tensor_file_for_pass = self._process_dir / f"Pass{self._forward_pass_id:05d}.pt"
        logger.info(
            f"Dump {self._forward_pass_id:05d}th pass to {tensor_file_for_pass}"
        )
        torch.save(self._current_tensors, str(tensor_file_for_pass))
        self._current_tensors = {}
        self._module_call_counts = {}
        self._active_calls.clear()
        self._forward_pass_id += 1

    def _next_call_name(self, tensor_name: str) -> str:
        call_index = self._module_call_counts.get(tensor_name, 0)
        self._module_call_counts[tensor_name] = call_index + 1
        if call_index == 0:
            return tensor_name
        return f"{tensor_name}.call{call_index}"

    def _all_io_pre_hook(self, tensor_name: str):
        def inner_pre_hook(module, args, kwargs):
            call_name = self._next_call_name(tensor_name)
            self._active_calls[id(module)].append(call_name)
            for index, item in enumerate(args):
                self._add_tensor_tree(f"{call_name}.input.args.{index}", item)
            for key, item in kwargs.items():
                self._add_tensor_tree(f"{call_name}.input.kwargs.{key}", item)

        return inner_pre_hook

    def _all_io_post_hook(self, tensor_name: str, do_dump: bool):
        def inner_post_hook(module, args, kwargs, output):
            active_calls = self._active_calls[id(module)]
            if not active_calls:
                raise RuntimeError(
                    f"Tensor dump post-hook for {tensor_name!r} has no matching pre-hook"
                )
            call_name = active_calls.pop()
            self._add_tensor_tree(f"{call_name}.output", output)
            if do_dump:
                self.dump_current_tensors()

        return inner_post_hook

    def _register_all_io_hooks(self, module, tensor_name: str, do_dump: bool) -> None:
        module.register_forward_pre_hook(
            self._all_io_pre_hook(tensor_name), with_kwargs=True
        )
        module.register_forward_hook(
            self._all_io_post_hook(tensor_name, do_dump), with_kwargs=True
        )

    def _wrap_root_forward(self, model) -> None:
        original_forward = model.forward

        @wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            call_name = self._next_call_name("__root__")
            for index, item in enumerate(args):
                self._add_tensor_tree(f"{call_name}.input.args.{index}", item)
            for key, item in kwargs.items():
                self._add_tensor_tree(f"{call_name}.input.kwargs.{key}", item)
            output = original_forward(*args, **kwargs)
            self._add_tensor_tree(f"{call_name}.output", output)
            self.dump_current_tensors()
            return output

        # ModelRunner invokes model.forward directly, bypassing root module hooks.
        model.forward = wrapped_forward

    def _add_hook_recursive(
        self, model, prefix, top_level_module_name, layers_module_name
    ):
        model_top_level_module_matched = False
        layers_prefix = top_level_module_name + "." + layers_module_name
        for name, module in model._modules.items():
            top_level_model = False
            if len(prefix) == 0:
                cur_name = name
                if cur_name == top_level_module_name:
                    model_top_level_module_matched = True
                    top_level_model = True
            else:
                cur_name = prefix + "." + name
            if (
                self._dump_layers is not None
                and name.isdigit()
                and prefix == layers_prefix
            ):
                # If we only need n layers, skip the reset layers.
                # Most models' layout is like model.layers.0.
                cur_layer = int(name)
                if cur_layer not in self._dump_layers:
                    continue
            if module is not None:
                _, sub_count = self._add_hook_recursive(
                    module, cur_name, top_level_module_name, layers_module_name
                )
                if self._mode == _ALL_IO_MODE:
                    self._register_all_io_hooks(module, cur_name, do_dump=False)
                elif sub_count == 0 or top_level_model:
                    # Avoid duplicated output hooks, e.g. self_attn may contain:
                    # self_attn.qkv_proj, self_attn.attn & self_attn.o_proj.
                    # Therefore, we do not need to add output hooks for self_attn,
                    # since the output of self_attn should be the same to self_attn.o_proj.
                    module.register_forward_hook(
                        self._dump_hook(cur_name, top_level_model)
                    )
        return model_top_level_module_matched, len(model._modules.items())

    def _dump_hook(self, tensor_name, do_dump):
        def inner_dump_hook(module, input, output):
            if do_dump:
                # This is the top-level model, so we will record the input for it.
                for item in input:
                    if isinstance(item, ForwardBatch):
                        self.add_tensor(tensor_name, item)
                self.dump_current_tensors()
            if output is not None:
                self.add_tensor(tensor_name, output)

        return inner_dump_hook


def register_forward_hook_for_model(
    model,
    dump_dir: str,
    dump_layers: Optional[List[int]],
    tp_size: int,
    tp_rank: int,
    pp_rank: int,
    mode: Optional[str] = None,
):
    tensor_dumper = TensorDumper(
        dump_dir, dump_layers, tp_size, tp_rank, pp_rank, mode=mode
    )
    # Most models have the layerout like:
    # XxxxForCausalLM
    #     (model): XxxxModel
    #         (layers): ModuleList
    # If the model is not constructed with this layout,
    # environment variable can be used to specify the module names.
    top_level_module_name = os.getenv("TENSOR_DUMP_TOP_LEVEL_MODULE_NAME", "model")
    layers_module_name = os.getenv("TENSOR_DUMP_LAYERS_MODULE_NAME", "layers")
    model_top_level_module_matched, _ = tensor_dumper._add_hook_recursive(
        model, "", top_level_module_name, layers_module_name
    )
    assert (
        model_top_level_module_matched
    ), f"model should have a module named {top_level_module_name}"
    if tensor_dumper._mode == _ALL_IO_MODE:
        tensor_dumper._wrap_root_forward(model)
    return tensor_dumper
