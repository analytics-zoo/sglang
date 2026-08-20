# Copyright 2023-2026 SGLang Team
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
"""XPU breakable graph: capture a decode forward as a sequence of
``torch.xpu.XPUGraph`` segments separated by EAGER break points, sharing one
graph mempool so pool-allocated intermediates keep stable addresses across
segments and replays.

Why this exists
---------------
On Intel XPU, a oneCCL collective (``dist.all_reduce``) captured inside a single
monolithic ``torch.xpu.XPUGraph`` does NOT replay correctly: the second and later
replays return stale / lagged reduction results (verified: the collective's data
movement is not re-driven per replay). In TP>1 gemma4 decode there are ~120
all-reduces per step (after every o_proj + down_proj), so a full-graph capture
produces correct output only on the first replay (token 0) and garbles every
token after.

The CUDA-side ``breakable_cuda_graph.py`` solves the analogous "op that can't be
captured" problem, but it depends on ``cuda.bindings.runtime`` (absent on XPU) and
CUDA side-stream capture-status tracking. This module is a minimal XPU-native
re-implementation: no cuda-python, no side-stream hook (the gemma4 decode forward
is single-stream), just segment capture around eager break points.

Mechanism
---------
During capture, ``break_graph()`` (via the ``eager_on_graph`` wrapper) ends the
current XPUGraph segment, runs the wrapped function EAGER once (so its outputs hold
real data + stable pool addresses), records a ``replay_fn`` that re-runs it eager,
and begins a fresh segment. At replay time, segments and break_fns are interleaved:
``seg[0].replay(); break_fn[0](); seg[1].replay(); ...``. The collective therefore
executes eagerly between captured compute segments — validated correct on XPU.
"""

import logging
from contextvars import ContextVar
from typing import Any, Callable, List, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "xpu_break_graph",
    "run_eager_between_segments",
    "XpuBreakableGraph",
    "XpuBreakableGraphCapture",
    "is_xpu_breakable_capturing",
]

# Active capture context for the currently-capturing thread. Set only while an
# XpuBreakableGraphCapture is open; break points read it to split the graph.
_current_capture_var: ContextVar["Optional[XpuBreakableGraphCapture]"] = ContextVar(
    "xpu_bcg_current_capture", default=None
)


def is_xpu_breakable_capturing() -> bool:
    return _current_capture_var.get() is not None


def _copy_output(dst: Any, src: Any) -> Any:
    """Copy src into dst in-place where possible (tensor / obj-with-tensors /
    dict). Returns dst on success else src. Mirrors the CUDA version so a break
    function's eager output lands in the stable buffer the captured graph reads."""
    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)
        return dst
    if hasattr(dst, "__dict__") and hasattr(src, "__dict__"):
        for key, src_val in src.__dict__.items():
            dst_val = getattr(dst, key, None)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                setattr(dst, key, src_val)
        return dst
    if isinstance(dst, dict) and isinstance(src, dict):
        for key, src_val in src.items():
            dst_val = dst.get(key)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                dst[key] = src_val
        return dst
    return src


def eager_on_xpu_graph(enable: bool):
    """Decorator: when an XpuBreakableGraphCapture is active, run ``inner`` as an
    eager break point (split the surrounding XPUGraph around it). Otherwise call
    ``inner`` normally (eager execution / non-breakable replay)."""

    def decorator(inner: Callable):
        if not enable:
            return inner

        def wrapper(*args, **kwargs):
            capture = _current_capture_var.get()
            if capture is None:
                # Not capturing a breakable graph: just run eagerly. (This is the
                # path taken during warmup and during replay's own break_fns.)
                return inner(*args, **kwargs)

            logger.debug("XPU break graph at: %s", getattr(inner, "__name__", inner))

            # 1. End the segment that captured everything up to here.
            capture._end_current_segment()

            # 2. Run the break function EAGER now so it produces real data in
            #    its (pool-stable) output buffers.
            output = inner(*args, **kwargs)

            # 3. Record a replay function that re-runs it eager and copies the
            #    result back into the same output buffer the next segment reads.
            captured_inner = inner
            captured_args = args
            captured_kwargs = kwargs
            captured_output = output

            def replay_fn():
                new_out = captured_inner(*captured_args, **captured_kwargs)
                return _copy_output(captured_output, new_out)

            capture.graph._break_fns.append(replay_fn)

            # 4. Begin a fresh segment for the remainder of the forward.
            capture._begin_new_segment()
            return output

        return wrapper

    return decorator


class XpuBreakableGraph:
    """Holds one ``torch.xpu.XPUGraph`` per segment plus an eager break function
    between consecutive segments. Duck-types the ``.replay()`` interface the
    graph runner calls, so it drops in where a single XPUGraph was used."""

    def __init__(self) -> None:
        self._segments: List["torch.xpu.XPUGraph"] = []
        self._break_fns: List[Callable[[], Any]] = []
        # Stream the segments were captured on. Replay must interleave segment
        # replays and eager break functions (collectives) on ONE stream so their
        # ordering is preserved — otherwise a break's collective can race the
        # adjacent segment (read-before-write) and corrupt decode.
        self._capture_stream: Optional["torch.xpu.Stream"] = None

    def replay(self) -> None:
        # Replay on the CURRENT stream (the one the runner/caller uses), NOT the
        # capture side-stream. The break functions are oneCCL collectives that
        # must run on the same stream on every rank to stay in lockstep; forcing
        # them onto a private capture stream desynchronizes the two TP ranks and
        # deadlocks. XPUGraph.replay() records/replays its own internal deps, so
        # the segment kernels remain correctly ordered on whatever stream we use.
        for i, seg in enumerate(self._segments):
            seg.replay()
            if i < len(self._break_fns):
                self._break_fns[i]()

    # torch.cuda.CUDAGraph exposes reset(); provide a no-op-ish equivalent so the
    # runner's cleanup paths don't blow up if they call it.
    def reset(self) -> None:
        for seg in self._segments:
            try:
                seg.reset()
            except Exception:
                pass
        self._segments.clear()
        self._break_fns.clear()


class XpuBreakableGraphCapture:
    """Context manager capturing the enclosed forward as ``XPUGraph`` segments
    split at ``xpu_break_graph()`` points, all sharing ``pool`` so intermediates
    keep stable addresses across segments/replays.

    Signature mirrors the ``torch.cuda.graph`` wrapper (``cuda_graph=``, ``pool=``,
    ``stream=``) so it slots into the graph runner's ``graph_ctx(...)`` call site.
    """

    def __init__(
        self,
        cuda_graph: XpuBreakableGraph,
        pool=None,
        stream: Optional["torch.xpu.Stream"] = None,
        capture_error_mode: str = "global",  # accepted + ignored (CUDA-only)
    ):
        assert isinstance(
            cuda_graph, XpuBreakableGraph
        ), "cuda_graph must be an XpuBreakableGraph"
        self.graph = cuda_graph
        self._pool = pool
        self._stream = stream
        self._stream_ctx = None
        self._capture_token = None

    def __enter__(self):
        if self._stream is not None:
            self._stream_ctx = torch.xpu.stream(self._stream)
            self._stream_ctx.__enter__()
        # Record the stream segments are captured on so replay can reproduce the
        # exact segment/break interleaving order on the same stream.
        self.graph._capture_stream = self._stream or torch.xpu.current_stream()
        self._capture_token = _current_capture_var.set(self)
        self._begin_new_segment()
        return self

    def __exit__(self, *args: object):
        try:
            self._end_current_segment()
            logger.debug("[XpuBreakableGraph] capture done: %s", self._debug_summary())
        finally:
            if self._capture_token is not None:
                _current_capture_var.reset(self._capture_token)
                self._capture_token = None
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*args)
                self._stream_ctx = None
        return False

    def _begin_new_segment(self) -> None:
        graph = torch.xpu.XPUGraph()
        if self._pool is not None:
            graph.capture_begin(pool=self._pool)
        else:
            graph.capture_begin()
        self.graph._segments.append(graph)

    def _end_current_segment(self) -> None:
        self.graph._segments[-1].capture_end()

    def _debug_summary(self) -> str:
        return (
            f"segments={len(self.graph._segments)} "
            f"break_fns={len(self.graph._break_fns)}"
        )


@eager_on_xpu_graph(True)
def xpu_break_graph() -> None:
    """Insert an XPU graph break. The decorator performs the segment split; the
    body is intentionally empty. Call this right BEFORE an op that must run eager
    inside an otherwise-captured region (e.g. a oneCCL collective)."""
    pass


@eager_on_xpu_graph(True)
def run_eager_between_segments(fn: Callable, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` as an EAGER break point between graph segments.

    Under breakable capture this ends the current segment, runs ``fn`` eager (its
    output buffer becomes the stable hand-off to the next segment), records it as a
    per-step replay function, and starts a new segment. Outside capture it is a
    plain ``fn(*args, **kwargs)`` call. Use for ops that cannot be captured into an
    XPUGraph — notably oneCCL collectives.

    ``fn`` must return the tensor whose (stable) storage the following segment
    reads; for an in-place collective on a pool buffer, return that buffer.
    """
    return fn(*args, **kwargs)
