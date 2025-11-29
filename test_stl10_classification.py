#!/usr/bin/env python3
"""
STL-10 Image Classification: Spatial Conv vs Fast FFT Conv
"""

import sys
import os

os.environ['NEEDLE_FFT_IMPL'] = 'cpp'
sys.path.insert(0, './python')

import numpy as np
import time

import needle
from needle import Tensor
import needle.nn as nn
import needle.ops as ops

from freq_conv_fast import FastFrequencyConv2D


def load_stl10_data(data_dir='./data/stl10', max_images=1000):
    """Load STL-10 training data."""

    # Load images
    img_file = os.path.join(data_dir, 'stl10_binary', 'train_X.bin')
    if not os.path.exists(img_file):
        print(f"❌ STL-10 not found at {img_file}")
        print(f"   Download from: http://ai.stanford.edu/~acoates/stl10/")
        return None, None

    with open(img_file, 'rb') as f:
        images = np.fromfile(f, dtype=np.uint8)

    num_images = len(images) // (96 * 96 * 3)
    images = images.reshape(num_images, 3, 96, 96)
    images = images[:max_images].astype(np.float32) / 255.0

    # Load labels
    label_file = os.path.join(data_dir, 'stl10_binary', 'train_y.bin')
    with open(label_file, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
    labels = labels[:max_images] - 1  # STL-10 labels are 1-10, convert to 0-9

    print(f"✓ Loaded {len(images)} images and labels")
    print(f"  Image shape: {images.shape}")
    print(f"  Label shape: {labels.shape}")

    return images, labels


class SpatialConvModel(nn.Module):
    """Simple CNN with Spatial Convolution."""

    def __init__(self, num_classes=10, device=None):
        super().__init__()
        self.device = device

        # 96×96 → 96×96 → 96×96, then pool
        self.conv1 = nn.Conv(3, 16, 5, device=device)
        self.conv2 = nn.Conv(16, 32, 5, device=device)
        self.fc = nn.Linear(32 * 96 * 96, num_classes, device=device)

    def forward(self, x):
        batch = x.shape[0]

        # Conv layers with ReLU
        x = self.conv1(x)
        x = ops.relu(x)

        x = self.conv2(x)
        x = ops.relu(x)

        # Flatten and classify
        x = ops.reshape(x, (batch, 32 * 96 * 96))
        x = self.fc(x)

        return x


class FFTConvModel(nn.Module):
    """Simple CNN with FFT Convolution."""

    def __init__(self, num_classes=10, device=None):
        super().__init__()
        self.device = device

        # Same architecture as Spatial model
        self.conv1 = FastFrequencyConv2D(3, 16, 5, device=device)
        self.conv2 = FastFrequencyConv2D(16, 32, 5, device=device)
        self.fc = nn.Linear(32 * 96 * 96, num_classes, device=device)

    def forward(self, x):
        batch = x.shape[0]

        # Conv layers with ReLU
        x = self.conv1(x)
        x = ops.relu(x)

        x = self.conv2(x)
        x = ops.relu(x)

        # Flatten and classify
        x = ops.reshape(x, (batch, 32 * 96 * 96))
        x = self.fc(x)

        return x


def train_epoch(model, images, labels, optimizer, batch_size=32):
    """Train for one epoch."""

    num_samples = len(images)
    num_batches = num_samples // batch_size

    total_loss = 0
    correct = 0

    device = model.device

    for i in range(num_batches):
        # Get batch
        start = i * batch_size
        end = start + batch_size

        batch_images = Tensor(images[start:end], device=device)
        batch_labels = Tensor(labels[start:end], device=device)

        # Forward
        logits = model(batch_images)
        loss = nn.SoftmaxLoss()(logits, batch_labels)

        # Backward
        optimizer.reset_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.numpy()
        preds = np.argmax(logits.numpy(), axis=1)
        correct += np.sum(preds == batch_labels.numpy())

    avg_loss = total_loss / num_batches
    accuracy = correct / (num_batches * batch_size)

    return avg_loss, accuracy


def evaluate(model, images, labels, batch_size=32):
    """Evaluate model."""

    num_samples = len(images)
    num_batches = num_samples // batch_size

    correct = 0
    device = model.device

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size

        batch_images = Tensor(images[start:end], device=device)
        batch_labels = Tensor(labels[start:end], device=device)

        logits = model(batch_images)
        preds = np.argmax(logits.numpy(), axis=1)
        correct += np.sum(preds == batch_labels.numpy())

    accuracy = correct / (num_batches * batch_size)
    return accuracy


def main():
    print("="*70)
    print("STL-10 Classification: Spatial Conv vs FFT Conv")
    print("="*70)

    # Load data
    print("\nLoading STL-10 dataset...")
    images, labels = load_stl10_data(max_images=1000)

    if images is None:
        return

    # Split into train/val
    split = 800
    train_images = images[:split]
    train_labels = labels[:split]
    val_images = images[split:]
    val_labels = labels[split:]

    print(f"\nDataset split:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")

    # Training configuration
    num_epochs = 2
    batch_size = 32
    lr = 0.001

    device = needle.cpu_numpy()

    results = {}

    # Test 1: Spatial Conv
    print(f"\n{'='*70}")
    print("1. Training with Spatial Convolution")
    print(f"{'='*70}")

    model_spatial = SpatialConvModel(num_classes=10, device=device)
    optimizer_spatial = needle.optim.Adam(model_spatial.parameters(), lr=lr)

    spatial_times = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        start = time.time()
        loss, acc = train_epoch(model_spatial, train_images, train_labels,
                               optimizer_spatial, batch_size)
        epoch_time = time.time() - start
        spatial_times.append(epoch_time)

        val_acc = evaluate(model_spatial, val_images, val_labels, batch_size)

        print(f"  Train Loss: {loss:.4f}, Train Acc: {acc*100:.2f}%")
        print(f"  Val Acc: {val_acc*100:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

    results['spatial'] = {
        'avg_time': np.mean(spatial_times),
        'final_val_acc': val_acc
    }

    # Test 2: FFT Conv
    print(f"\n{'='*70}")
    print("2. Training with FFT Convolution")
    print(f"{'='*70}")

    try:
        model_fft = FFTConvModel(num_classes=10, device=device)
        optimizer_fft = needle.optim.Adam(model_fft.parameters(), lr=lr)

        fft_times = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            start = time.time()
            loss, acc = train_epoch(model_fft, train_images, train_labels,
                                   optimizer_fft, batch_size)
            epoch_time = time.time() - start
            fft_times.append(epoch_time)

            val_acc = evaluate(model_fft, val_images, val_labels, batch_size)

            print(f"  Train Loss: {loss:.4f}, Train Acc: {acc*100:.2f}%")
            print(f"  Val Acc: {val_acc*100:.2f}%")
            print(f"  Time: {epoch_time:.1f}s")

        results['fft'] = {
            'avg_time': np.mean(fft_times),
            'final_val_acc': val_acc
        }

    except Exception as e:
        print(f"\n❌ FFT training failed: {e}")
        import traceback
        traceback.print_exc()
        results['fft'] = None

    # Summary
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Method':<20} | {'Avg Time/Epoch':<15} | {'Val Accuracy':<12}")
    print("-"*55)

    spatial_time = results['spatial']['avg_time']
    spatial_acc = results['spatial']['final_val_acc']
    print(f"{'Spatial Conv':<20} | {spatial_time:>13.1f}s | {spatial_acc*100:>10.2f}%")

    if results['fft'] is not None:
        fft_time = results['fft']['avg_time']
        fft_acc = results['fft']['final_val_acc']
        print(f"{'FFT Conv':<20} | {fft_time:>13.1f}s | {fft_acc*100:>10.2f}%")

        speedup = spatial_time / fft_time
        print(f"\nSpeedup: {speedup:.2f}x ({'FFT faster' if speedup > 1 else 'Spatial faster'})")

        if speedup > 1:
            print(f"✓ FFT convolution is {speedup:.2f}x faster!")
        elif speedup > 0.5:
            print(f"⚠ FFT is competitive (only {1/speedup:.2f}x slower)")
        else:
            print(f"⚠ FFT is {1/speedup:.2f}x slower")
            print(f"  Try larger images or GPU for better performance")

    print(f"\n{'='*70}\n")

    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
