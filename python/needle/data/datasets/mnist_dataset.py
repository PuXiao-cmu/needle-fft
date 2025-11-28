from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        # Read images
        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            assert magic == 2051, f"Invalid magic number {magic} in image file"
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            self.images = self.images.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Read labels
        with gzip.open(label_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            assert magic == 2049, f"Invalid magic number {magic} in label file"
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

        assert len(self.images) == len(self.labels), "Number of images and labels must match"

        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        label = self.labels[index]

        if self.transforms:
            for transform in self.transforms:
                image = transform(image)

        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION