#!/bin/bash

# Wrapper script to run FFT comparison tests with correct environment variables
# This ensures NEEDLE_FFT_IMPL is set BEFORE importing needle

echo "================================================================================"
echo "Frequency-Domain Convolution vs Spatial Convolution on MNIST"
echo "================================================================================"
echo ""
echo "Test Configuration:"
echo "  Epochs: 2"
echo "  Batch size: 32"
echo "  Batches/epoch: 50"
echo ""

# Test 1: Spatial Convolution (Baseline, NO FFT)
echo ""
echo "################################################################################"
echo "# TEST 1: Spatial Convolution (Baseline, NO FFT)"
echo "################################################################################"
echo ""

NEEDLE_FFT_IMPL=numpy /opt/homebrew/bin/python3.10 test_frequency_conv_mnist.py spatial

# Test 2: Frequency Convolution with C++ FFT
echo ""
echo ""
echo "################################################################################"
echo "# TEST 2: Frequency Convolution with C++ FFT"
echo "################################################################################"
echo ""

NEEDLE_FFT_IMPL=cpp /opt/homebrew/bin/python3.10 test_frequency_conv_mnist.py cpp

# Test 3: Frequency Convolution with NumPy FFT
echo ""
echo ""
echo "################################################################################"
echo "# TEST 3: Frequency Convolution with NumPy FFT"
echo "################################################################################"
echo ""

NEEDLE_FFT_IMPL=numpy /opt/homebrew/bin/python3.10 test_frequency_conv_mnist.py numpy

# Test 4: Frequency Convolution with Python Cooley-Tukey
echo ""
echo ""
echo "################################################################################"
echo "# TEST 4: Frequency Convolution with Python Cooley-Tukey"
echo "################################################################################"
echo ""

NEEDLE_FFT_IMPL=cooley_tukey /opt/homebrew/bin/python3.10 test_frequency_conv_mnist.py cooley_tukey

echo ""
echo "================================================================================"
echo "All tests completed!"
echo "================================================================================"
