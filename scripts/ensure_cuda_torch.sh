#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="${1:-.venv/bin/python}"
TORCH_BACKEND="${TORCH_BACKEND:-cu126}"
UV_BIN="${UV_BIN:-uv}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not on PATH; cannot validate GPU allocation."
  exit 1
fi

nvidia-smi || { echo "nvidia-smi failed; no GPU visible in job"; exit 1; }

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python interpreter not found or not executable: ${VENV_PYTHON}"
  exit 1
fi

if "${VENV_PYTHON}" - <<'PY'
import sys
import torch

if torch.cuda.is_available():
    raise SystemExit(0)

if torch.version.cuda is None or "+cpu" in torch.__version__:
    raise SystemExit(1)

raise SystemExit(2)
PY
then
  echo "CUDA preflight passed in existing environment."
else
  status=$?
  if [[ "${status}" -eq 1 ]]; then
    echo "Detected CPU-only torch in ${VENV_PYTHON}; reinstalling CUDA torch (${TORCH_BACKEND})."
    if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
      echo "Required tool '${UV_BIN}' not found on PATH."
      exit 1
    fi
    "${UV_BIN}" pip install \
      --python "${VENV_PYTHON}" \
      --reinstall-package torch \
      --torch-backend "${TORCH_BACKEND}" \
      torch
  else
    echo "CUDA torch build detected but torch.cuda.is_available() is still False."
    echo "This usually indicates a runtime/driver/module mismatch."
    exit 1
  fi
fi

"${VENV_PYTHON}" - <<'PY'
import torch

assert torch.cuda.is_available(), "torch.cuda.is_available() is False after CUDA torch preflight"
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device 0:", torch.cuda.get_device_name(0))
print("Torch version:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
PY
