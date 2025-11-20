#!/bin/bash
#
# Compilation script for C++ and CUDA FFT implementations
#

set -e  # Exit on error

echo "========================================="
echo "Compiling FFT Implementations"
echo "========================================="

# Check if make is available
if ! command -v make &> /dev/null; then
    echo "Error: make is not installed"
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")"

echo ""
echo "[1/3] Compiling C++ CPU backend with FFT..."
make clean
make

echo ""
echo "[2/3] Testing if compilation succeeded..."
if [ -f "python/needle/backend_ndarray/ndarray_backend_cpu.cpython"*".so" ]; then
    echo "✓ C++ backend compiled successfully"
else
    echo "✗ C++ backend compilation failed"
    exit 1
fi

echo ""
echo "[3/3] Checking CUDA backend..."
if [ -f "python/needle/backend_ndarray/ndarray_backend_cuda.cpython"*".so" ]; then
    echo "✓ CUDA backend found (already compiled)"
else
    echo "⚠ CUDA backend not found (GPU not available or not compiled)"
fi

echo ""
echo "========================================="
echo "Compilation Complete!"
echo "========================================="
echo ""
echo "Available FFT implementations:"
echo "  - numpy (default):  export NEEDLE_FFT_IMPL=numpy"
echo "  - Python Cooley-Tukey:  export NEEDLE_FFT_IMPL=cooley_tukey"
echo "  - C++ Cooley-Tukey:  Use C++ backend directly"
echo "  - CUDA Cooley-Tukey: Use CUDA backend directly"
echo ""
echo "Run tests with:"
echo "  python test_fft_cpp_cuda.py"
echo ""
