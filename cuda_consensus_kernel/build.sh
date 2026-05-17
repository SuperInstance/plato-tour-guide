#!/bin/bash
# build.sh — Compile CUDA consensus kernels for RTX 4050 (sm_75)
#
# Fortran philosophy: explicit build commands, no cmake, no magic.
# Check CUDA availability at build time. Fall back to CPU if no GPU.
#
# Usage: ./build.sh [--clean]
#
# Output: cuda_consensus_kernel.so  (shared object for Python ctypes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SO_NAME="cuda_consensus_kernel.so"
NVCC="${NVCC:-nvcc}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CLEAN=0
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN=1
fi

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
if [[ $CLEAN -eq 1 ]]; then
    echo "Cleaning build artifacts..."
    rm -f "${SCRIPT_DIR}/${SO_NAME}"
    rm -f "${SCRIPT_DIR}"/*.o
    rm -f "${SCRIPT_DIR}"/*.ptx
    rm -f "${SCRIPT_DIR}"/*.cubin
    echo "Done."
    exit 0
fi

# ---------------------------------------------------------------------------
# CUDA availability check at build time
# ---------------------------------------------------------------------------
echo "[build] Checking CUDA availability..."

if ! command -v nvcc &> /dev/null; then
    echo "[build] ERROR: nvcc not found in PATH."
    echo "[build] Install CUDA Toolkit or add it to PATH."
    echo "[build] Falling back to CPU-only build (Python fallback will be used)."
    echo "[build] CPU fallback code is in cosine_distance_kernel.cu (extern \"C\" void cosine_distance_cpu)."
    exit 0
fi

# Check if a GPU is visible to nvcc
NVCC_CHECK=$(nvcc -V 2>&1 || true)
echo "[build] nvcc: ${NVCC_CHECK}"

# Try to detect GPU
GPU_DETECTED=0
if nvcc -arch=sm_75 -o /dev/null -x cu - <<<'int main(){return 0;}' 2>/dev/null; then
    GPU_DETECTED=1
    echo "[build] GPU (sm_75) detected — compiling CUDA kernels."
else
    echo "[build] No GPU visible to nvcc — compiling in emit-relocatable mode."
    echo "[build] Will produce .so for deployment on RTX 4050."
fi

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
# RTX 4050 = compute capability 7.5 (sm_75)
# -use_fast_math: maps to Fortran FMAD (fused multiply-add)
# -fPIC: position-independent code for .so
# --relocatable-device-code: allows separate compilation

CUDA_FLAGS=(
    -O3                        # Maximum optimization
    -use_fast_math             # Fortran FMAD equivalent
    -arch=sm_75               # RTX 4050 compute capability
    -Xcompiler -fPIC          # Position-independent code
    -lineinfo                 # Line-level profiling
    -std=c++14                # C++14 for __shfl_xor_sync
)

# If no GPU, use emit-relocatable-device-code for deferred final linking
if [[ $GPU_DETECTED -eq 0 ]]; then
    echo "[build] Warning: compiling with --relocatable-device-code (no GPU for final link)."
    CUDA_FLAGS+=(
        --relocatable-device-code=yes
        -rdc=true
    )
fi

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
KERNEL_SOURCES=(
    "${SCRIPT_DIR}/cosine_distance_kernel.cu"
    "${SCRIPT_DIR}/spread_reduce_kernel.cu"
    "${SCRIPT_DIR}/consensus_snap_kernel.cu"
)

# ---------------------------------------------------------------------------
# Compile each kernel to object file, then link into .so
# ---------------------------------------------------------------------------
OBJECT_FILES=()

for src in "${KERNEL_SOURCES[@]}"; do
    base="$(basename "${src}" .cu)"
    obj="${SCRIPT_DIR}/${base}.o"

    echo "[build] Compiling ${src} -> ${obj}"
    nvcc "${CUDA_FLAGS[@]}" -c "${src}" -o "${obj}"

    OBJECT_FILES+=("${obj}")
done

# ---------------------------------------------------------------------------
# Link into shared object
# ---------------------------------------------------------------------------
echo "[build] Linking -> ${SO_NAME}"
nvcc "${CUDA_FLAGS[@]}" \
    -shared \
    "${OBJECT_FILES[@]}" \
    -o "${SCRIPT_DIR}/${SO_NAME}"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
if [[ -f "${SCRIPT_DIR}/${SO_NAME}" ]]; then
    SIZE=$(du -h "${SCRIPT_DIR}/${SO_NAME}" | cut -f1)
    echo "[build] SUCCESS: ${SO_NAME} (${SIZE})"
    echo "[build] For Python ctypes loading, use: ctypes.CDLL('${SO_NAME}')"
else
    echo "[build] ERROR: ${SO_NAME} not created."
    exit 1
fi

# ---------------------------------------------------------------------------
# CUDA availability report
# ---------------------------------------------------------------------------
echo ""
echo "=== Build Summary ==="
echo "  Target:        RTX 4050 (sm_75)"
echo "  Optimization: -O3 -use_fast_math"
echo "  Output:        ${SO_NAME}"
echo "  Kernel sources:"
for src in "${KERNEL_SOURCES[@]}"; do
    echo "    - $(basename "${src}")"
done
echo ""
echo "To run benchmarks:"
echo "  python3 benchmark_cuda.py"
echo ""
echo "To load in Python via ctypes:"
echo "  import ctypes"
echo "  lib = ctypes.CDLL('${SO_NAME}')"
echo "  # Then call lib.cosine_distance_launch(...) via ctypes"
echo ""
echo "CPU fallback is always available (see cosine_distance_kernel.cu cosine_distance_cpu)."