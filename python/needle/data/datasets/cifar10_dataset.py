import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.base_folder = base_folder
        self.train = train
        self.transforms = transforms or []
        self.p = p

        def _load_batch(path: str):
            with open(path, "rb") as f:
                obj = pickle.load(f, encoding="latin1")
            data = obj["data"] if "data" in obj else obj[b"data"]
            labels = obj.get("labels", obj.get(b"labels"))
            data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            labels = np.array(labels, dtype=np.int64)
            return data, labels

        if train:
            xs, ys = [], []
            for i in range(1, 6):
                fp = os.path.join(base_folder, f"data_batch_{i}")
                x, y = _load_batch(fp)
                xs.append(x)
                ys.append(y)
            self.X = np.concatenate(xs, axis=0)  # (50000, 3, 32, 32)
            self.y = np.concatenate(ys, axis=0)  # (50000,)
        else:
            fp = os.path.join(base_folder, "test_batch")
            self.X, self.y = _load_batch(fp)      # (10000, 3, 32, 32), (10000,)

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img = self.X[index]        # (3, 32, 32), float32
        label = int(self.y[index])

        for t in self.transforms:
            img = t(img)
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
