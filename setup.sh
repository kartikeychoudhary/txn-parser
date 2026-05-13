#!/usr/bin/env bash
# One-shot setup for the transaction-parser pipeline on Linux/WSL.
#
# Installs:
#   1. pip-upgraded Python toolchain
#   2. torch 2.8.0 + cu128 (Blackwell-ready)
#   3. base + training requirements
#   4. huggingface_hub CLI with hf_transfer
#   5. llama-cpp-python built against CUDA (falls back to a prebuilt wheel,
#      then to CPU-only if neither path works)
#   6. node deps for the viewer (if npm is on PATH)
#
# Then downloads the trained models from huggingface.co/kartikey31/txn-parser
# into ./models/.
#
# Re-runnable — every step short-circuits if it's already satisfied.
#
# Usage:
#   bash setup.sh                # full setup (recommended)
#   bash setup.sh --no-models    # skip the HF download (just install deps)
#   bash setup.sh --cpu-only     # skip the CUDA build for llama-cpp-python
#
# Expects to be run from the repo root after `conda activate <env>` or with
# an already-activated Python 3.11 virtualenv.

set -euo pipefail

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
LLAMA_CPP_PREBUILT_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu124"
HF_REPO="kartikey31/txn-parser"
HF_LOCAL_DIR="models"

DOWNLOAD_MODELS=1
CPU_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --no-models)  DOWNLOAD_MODELS=0 ;;
        --cpu-only)   CPU_ONLY=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            echo "unknown flag: $arg (try --help)"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
step()   { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
ok()     { printf "    \033[1;32m✓\033[0m %s\n" "$*"; }
warn()   { printf "    \033[1;33m!\033[0m %s\n" "$*"; }
fail()   { printf "    \033[1;31m✗\033[0m %s\n" "$*"; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

py() { python -c "$1"; }

# --------------------------------------------------------------------------
# Sanity
# --------------------------------------------------------------------------
step "Checking Python"
have python   || fail "python not on PATH — activate your conda env first."
PY_VER=$(py 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ok "python $PY_VER ($(which python))"
[[ "$PY_VER" == "3.11" ]] || warn "Recommended Python is 3.11; you're on $PY_VER."

step "Upgrading pip"
python -m pip install --upgrade pip --quiet
ok "pip $(pip --version | awk '{print $2}')"

# --------------------------------------------------------------------------
# torch (cu128)
# --------------------------------------------------------------------------
step "Installing torch $TORCH_VERSION + cu128"
TORCH_OK=$(py "
try:
    import torch
    v = torch.__version__
    cu = torch.version.cuda
    print('YES' if v.startswith('$TORCH_VERSION') and cu and cu.startswith('12.8') else 'NO', v, cu)
except Exception:
    print('NO none none')
" 2>/dev/null || echo "NO none none")
read -r status v cu <<< "$TORCH_OK"
if [[ "$status" == "YES" ]]; then
    ok "torch $v (cuda $cu) already installed"
else
    pip install \
        "torch==$TORCH_VERSION" \
        "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION" \
        --index-url "$TORCH_INDEX"
    ok "installed torch $TORCH_VERSION + cu128"
fi

# Sanity-check torch sees the GPU. Non-fatal — a CI machine without a GPU
# can still run the rest.
GPU_OK=$(py "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null || echo "NO")
if [[ "$GPU_OK" == "YES" ]]; then
    GPU_NAME=$(py "import torch; print(torch.cuda.get_device_name(0))")
    ok "CUDA visible: $GPU_NAME"
else
    warn "torch.cuda.is_available() is False — training/inference will be CPU-only."
fi

# --------------------------------------------------------------------------
# Project requirements
# --------------------------------------------------------------------------
step "Installing base requirements"
pip install -r requirements.txt --quiet
ok "requirements.txt"

step "Installing training requirements"
pip install -r requirements-train.txt
ok "requirements-train.txt"

step "Installing huggingface_hub CLI + hf_transfer"
pip install -U "huggingface_hub[cli]" hf_transfer --quiet
ok "huggingface_hub $(py 'import huggingface_hub; print(huggingface_hub.__version__)')"

# --------------------------------------------------------------------------
# llama-cpp-python with CUDA
# --------------------------------------------------------------------------
step "Installing llama-cpp-python"
LCPP_STATUS=$(py "
try:
    from llama_cpp import llama_cpp
    print('GPU' if llama_cpp.llama_supports_gpu_offload() else 'CPU')
except Exception:
    print('MISSING')
" 2>/dev/null || echo "MISSING")

if [[ "$LCPP_STATUS" == "GPU" ]]; then
    ok "llama-cpp-python (CUDA) already installed"
elif [[ "$CPU_ONLY" -eq 1 ]]; then
    if [[ "$LCPP_STATUS" == "MISSING" ]]; then
        pip install llama-cpp-python --quiet
    fi
    ok "llama-cpp-python (CPU-only, by request)"
else
    BUILT=0

    # Path A: build from source with CUDA if nvcc is present.
    if have nvcc; then
        NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
        ok "nvcc $NVCC_VER detected — building llama-cpp-python with CUDA"
        if CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
           pip install llama-cpp-python --no-cache-dir --force-reinstall; then
            BUILT=1
            ok "built llama-cpp-python from source (CUDA)"
        else
            warn "source build failed, falling back to prebuilt wheel"
        fi
    else
        warn "nvcc not on PATH — cannot build from source"
    fi

    # Path B: prebuilt CUDA wheel (cu124 wheels run fine against cu128 driver).
    if [[ "$BUILT" -eq 0 ]]; then
        if pip install llama-cpp-python \
               --extra-index-url "$LLAMA_CPP_PREBUILT_INDEX" \
               --force-reinstall --no-cache-dir; then
            BUILT=1
            ok "installed prebuilt CUDA wheel (cu124)"
        else
            warn "prebuilt CUDA wheel install failed; falling back to CPU build"
        fi
    fi

    # Path C: CPU-only as a last resort.
    if [[ "$BUILT" -eq 0 ]]; then
        pip install llama-cpp-python --force-reinstall --no-cache-dir
        warn "llama-cpp-python installed CPU-only — inference will be slow."
    fi

    # Verify the result.
    LCPP_FINAL=$(py "
from llama_cpp import llama_cpp
print('GPU' if llama_cpp.llama_supports_gpu_offload() else 'CPU')
" 2>/dev/null || echo "MISSING")
    if [[ "$LCPP_FINAL" == "GPU" ]]; then
        ok "llama-cpp-python supports GPU offload"
    elif [[ "$LCPP_FINAL" == "CPU" ]]; then
        warn "llama-cpp-python is CPU-only. Rebuild later with: \
CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native' \
pip install llama-cpp-python --no-cache-dir --force-reinstall"
    else
        fail "llama-cpp-python import failed after install"
    fi
fi

# --------------------------------------------------------------------------
# Node deps (viewer)
# --------------------------------------------------------------------------
if have npm; then
    step "Installing viewer Node deps"
    (cd viewer && npm install --silent)
    ok "viewer/node_modules"
else
    warn "npm not on PATH — skipping viewer setup (Stage 2/7 won't run)."
fi

# --------------------------------------------------------------------------
# HF model download
# --------------------------------------------------------------------------
if [[ "$DOWNLOAD_MODELS" -eq 1 ]]; then
    step "Downloading models from huggingface.co/$HF_REPO"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    mkdir -p "$HF_LOCAL_DIR"
    huggingface-cli download "$HF_REPO" \
        --repo-type=model \
        --local-dir "$HF_LOCAL_DIR"
    ok "models downloaded to ./$HF_LOCAL_DIR"
else
    warn "--no-models passed — skipping HF download"
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
step "Setup complete"
cat <<EOF

Next steps:
  1. Drop your DeepSeek key into .env  (cp .env.example .env, then edit)
  2. Stage 2 viewer  : cd viewer && npm start    -> http://localhost:3000
  3. Eval the teacher: python scripts/04_eval.py --model models/teacher/gguf --limit 50
  4. Playground UI   : http://localhost:3000/playground (after npm start)

Verify GPU stack with:
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
  python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"

EOF
