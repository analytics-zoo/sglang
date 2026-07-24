#!/usr/bin/env bash
# Launch the qualified Onyx TP=2 online-FP8 tool-calling path on Intel XPU/BMG.
#
# Run detached from the host so the service survives terminal teardown:
#   docker exec -d txytest_sgl_bmg bash -lc \
#     'cd /llm/workspace/sgl_gemma/sglang &&
#      bash benchmark/onyx/launch_onyx_fp8_tp2.sh \
#      > /llm/workspace/copilot_workspace/onyx_tool_calling_server.log 2>&1'
#
# Image-capable serving with the BF16 vision tower:
#   ONYX_ENABLE_VISION=1 bash benchmark/onyx/launch_onyx_fp8_tp2.sh
#
# Print the resolved environment and command without importing extensions or
# starting a server:
#   ONYX_DRY_RUN=1 bash benchmark/onyx/launch_onyx_fp8_tp2.sh
#
# Long-context retrieval testing (up to a 32K input plus a short completion,
# with scheduler/KV-cache headroom):
#   ONYX_CONTEXT_LENGTH=33024 ONYX_MAX_TOTAL_TOKENS=33280 \
#   ONYX_ALLOW_LONG_CONTEXT=1 \
#     bash benchmark/onyx/launch_onyx_fp8_tp2.sh
#
# Tool calling uses the tracked Onyx integration template. The server enforces
# one schema-constrained tool call per assistant response regardless of client
# parallel_tool_calls settings.
#
# Additional command-line arguments are appended to launch_server:
#   bash benchmark/onyx/launch_onyx_fp8_tp2.sh --log-level info

set -euo pipefail

ONYX_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ONYX_DEFAULT_SGLANG_ROOT=$(cd -- "${ONYX_SCRIPT_DIR}/../.." && pwd)

ONYX_SGLANG_ROOT=${ONYX_SGLANG_ROOT:-${ONYX_DEFAULT_SGLANG_ROOT}}
ONYX_WORKSPACE_ROOT=${ONYX_WORKSPACE_ROOT:-$(cd -- "${ONYX_SGLANG_ROOT}/.." && pwd)}
ONYX_KERNEL_PYTHON=${ONYX_KERNEL_PYTHON:-${ONYX_WORKSPACE_ROOT}/llm-scaler/sglang/custom-esimd-kernels/python}
ONYX_MODEL_PATH=${ONYX_MODEL_PATH:-/llm/workspace/model/onyx-hf}
ONYX_TOOL_CHAT_TEMPLATE=${ONYX_TOOL_CHAT_TEMPLATE:-${ONYX_SCRIPT_DIR}/onyx_tool_chat_template.jinja}
ONYX_PYTHON=${ONYX_PYTHON:-python3}
ONYX_HOST=${ONYX_HOST:-0.0.0.0}
ONYX_PORT=${ONYX_PORT:-31888}
ONYX_MAX_RUNNING_REQUESTS=${ONYX_MAX_RUNNING_REQUESTS:-1}
ONYX_CONTEXT_LENGTH=${ONYX_CONTEXT_LENGTH:-16384}
ONYX_MAX_TOTAL_TOKENS=${ONYX_MAX_TOTAL_TOKENS:-16384}
ONYX_CHUNKED_PREFILL_SIZE=${ONYX_CHUNKED_PREFILL_SIZE:-1024}
ONYX_SWA_FULL_TOKENS_RATIO=${ONYX_SWA_FULL_TOKENS_RATIO:-0.25}
ONYX_ALLOW_LONG_CONTEXT=${ONYX_ALLOW_LONG_CONTEXT:-0}
ONYX_ENABLE_VISION=${ONYX_ENABLE_VISION:-0}
ONYX_DRY_RUN=${ONYX_DRY_RUN:-0}
ONYX_STRICT_KERNEL_CHECK=${ONYX_STRICT_KERNEL_CHECK:-1}

for value_name in ONYX_CONTEXT_LENGTH ONYX_MAX_TOTAL_TOKENS ONYX_CHUNKED_PREFILL_SIZE ONYX_MAX_RUNNING_REQUESTS; do
  value=${!value_name}
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
done

if [[ ! "${ONYX_SWA_FULL_TOKENS_RATIO}" =~ ^(0(\.[0-9]+)?|1(\.0*)?)$ ]] \
  || [[ "${ONYX_SWA_FULL_TOKENS_RATIO}" =~ ^0(\.0*)?$ ]]; then
  echo "ONYX_SWA_FULL_TOKENS_RATIO must be in (0, 1], got: ${ONYX_SWA_FULL_TOKENS_RATIO}" >&2
  exit 2
fi

case "${ONYX_ALLOW_LONG_CONTEXT}" in
  0|1) ;;
  *)
    echo "ONYX_ALLOW_LONG_CONTEXT must be 0 or 1, got: ${ONYX_ALLOW_LONG_CONTEXT}" >&2
    exit 2
    ;;
esac

if (( ONYX_MAX_TOTAL_TOKENS < ONYX_CONTEXT_LENGTH )); then
  echo "ONYX_MAX_TOTAL_TOKENS must be >= ONYX_CONTEXT_LENGTH" >&2
  exit 2
fi

if (( ONYX_CONTEXT_LENGTH > 16384 )); then
  if [[ "${ONYX_ALLOW_LONG_CONTEXT}" != "1" ]]; then
    echo "ONYX_CONTEXT_LENGTH=${ONYX_CONTEXT_LENGTH} exceeds the model's declared 16384-token limit." >&2
    echo "Set ONYX_ALLOW_LONG_CONTEXT=1 to run an explicit extrapolation test." >&2
    exit 2
  fi
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
fi

if [[ ! -d "${ONYX_MODEL_PATH}" ]]; then
  echo "Onyx model directory does not exist: ${ONYX_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ONYX_TOOL_CHAT_TEMPLATE}" ]]; then
  echo "Onyx tool-calling template does not exist: ${ONYX_TOOL_CHAT_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -d "${ONYX_KERNEL_PYTHON}" ]]; then
  echo "Custom ESIMD kernel Python directory does not exist: ${ONYX_KERNEL_PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${ONYX_SGLANG_ROOT}/python:${ONYX_KERNEL_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export SGLANG_USE_SGL_XPU=1
export SGLANG_XPU_FP8_W8A16_PREFILL=1

case "${ONYX_ENABLE_VISION}" in
  0)
    export SGLANG_SKIP_VISION_GPU=1
    ;;
  1)
    export SGLANG_SKIP_VISION_GPU=0
    export SGLANG_FP8_IGNORED_LAYERS=${SGLANG_FP8_IGNORED_LAYERS:-model.vision_encoder,model.vision_adapter,model.vision_projection,model.perception_emb_norm}
    ;;
  *)
    echo "ONYX_ENABLE_VISION must be 0 or 1, got: ${ONYX_ENABLE_VISION}" >&2
    exit 2
    ;;
esac

ONYX_SERVER_ARGS=(
  -m sglang.launch_server
  --model-path "${ONYX_MODEL_PATH}"
  --chat-template "${ONYX_TOOL_CHAT_TEMPLATE}"
  --trust-remote-code
  --model-impl sglang
  --tool-call-parser onyx
  --grammar-backend xgrammar
  --sampling-backend pytorch

  --device xpu
  --tp 2
  --dtype float16
  --quantization fp8
  --load-format layered_fp8
  --attention-backend intel_xpu
  --page-size 64
  --mem-fraction-static 0.95
  --max-total-tokens "${ONYX_MAX_TOTAL_TOKENS}"
  --context-length "${ONYX_CONTEXT_LENGTH}"
  --swa-full-tokens-ratio "${ONYX_SWA_FULL_TOKENS_RATIO}"
  --chunked-prefill-size "${ONYX_CHUNKED_PREFILL_SIZE}"
  --max-running-requests "${ONYX_MAX_RUNNING_REQUESTS}"
  --disable-cuda-graph
  --disable-custom-all-reduce
  --disable-overlap-schedule
  --skip-server-warmup
  --watchdog-timeout 300
  --random-seed 0
  --enable-cache-report

  --host "${ONYX_HOST}"
  --port "${ONYX_PORT}"
)

if [[ "${ONYX_DRY_RUN}" == "1" ]]; then
  printf 'ZE_AFFINITY_MASK=%q\n' "${ZE_AFFINITY_MASK}"
  printf 'SGLANG_SKIP_VISION_GPU=%q\n' "${SGLANG_SKIP_VISION_GPU}"
  printf 'ONYX_CONTEXT_LENGTH=%q\n' "${ONYX_CONTEXT_LENGTH}"
  printf 'ONYX_MAX_TOTAL_TOKENS=%q\n' "${ONYX_MAX_TOTAL_TOKENS}"
  printf 'ONYX_CHUNKED_PREFILL_SIZE=%q\n' "${ONYX_CHUNKED_PREFILL_SIZE}"
  printf 'ONYX_SWA_FULL_TOKENS_RATIO=%q\n' "${ONYX_SWA_FULL_TOKENS_RATIO}"
  printf 'ONYX_ALLOW_LONG_CONTEXT=%q\n' "${ONYX_ALLOW_LONG_CONTEXT}"
  printf 'ONYX_TOOL_CHAT_TEMPLATE=%q\n' "${ONYX_TOOL_CHAT_TEMPLATE}"
  if [[ -n "${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-}" ]]; then
    printf 'SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=%q\n' "${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN}"
  fi
  if [[ "${ONYX_ENABLE_VISION}" == "1" ]]; then
    printf 'SGLANG_FP8_IGNORED_LAYERS=%q\n' "${SGLANG_FP8_IGNORED_LAYERS}"
  fi
  printf 'PYTHONPATH=%q\n' "${PYTHONPATH}"
  printf 'command:'
  printf ' %q' "${ONYX_PYTHON}" "${ONYX_SERVER_ARGS[@]}" "$@"
  printf '\n'
  exit 0
fi

if [[ "${ONYX_STRICT_KERNEL_CHECK}" == "1" ]]; then
  "${ONYX_PYTHON}" - <<'PY'
import custom_esimd_kernels_sglang as kernels
import torch

required_ops = (
    ("custom_esimd_kernels", "esimd_gemm_fp8_pert"),
    ("custom_esimd_kernels_sglang", "esimd_gemv_fp8_pert_fused2"),
    ("custom_esimd_kernels_sglang", "esimd_kv_scatter"),
    ("custom_esimd_kernels_sglang", "esimd_norm_add_norm"),
    ("custom_esimd_kernels_sglang", "esimd_rmsnorm_residual_scalar"),
    ("custom_esimd_kernels_sglang", "esimd_gemv_fp16"),
    ("custom_esimd_kernels_sglang", "onednn_fp8_gemm_w8a16"),
)
missing_ops = []
missing_xpu_kernels = []
dispatch_check = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
for namespace, name in required_ops:
    qualified_name = f"{namespace}::{name}"
    if not hasattr(getattr(torch.ops, namespace), name):
        missing_ops.append(qualified_name)
    elif dispatch_check is not None and not dispatch_check(qualified_name, "XPU"):
        missing_xpu_kernels.append(qualified_name)

if missing_ops or missing_xpu_kernels:
    extension_errors = getattr(kernels, "_MISSING_EXTS", ())
    details = "\n".join(f"  {name}: {error}" for name, error in extension_errors)
    problems = []
    if missing_ops:
        problems.append("unregistered ops: " + ", ".join(missing_ops))
    if missing_xpu_kernels:
        problems.append("ops without XPU kernels: " + ", ".join(missing_xpu_kernels))
    raise SystemExit(
        "Required Onyx native kernels are unavailable: "
        + "; ".join(problems)
        + ("\nExtension load errors:\n" + details if details else "")
    )

try:
    import sgl_kernel
except Exception as exc:
    raise SystemExit(f"Unable to import sgl_kernel: {exc}") from exc
if not hasattr(sgl_kernel, "fused_qk_norm_rope"):
    raise SystemExit("sgl_kernel.fused_qk_norm_rope is unavailable")
PY
fi

exec "${ONYX_PYTHON}" "${ONYX_SERVER_ARGS[@]}" "$@"
