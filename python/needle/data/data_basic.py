import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        n = len(self.dataset)
        bs = 1 if self.batch_size is None else self.batch_size

        if self.shuffle:
            indices = np.arange(n)
            np.random.shuffle(indices)
            self._ordering = np.array_split(indices, range(bs, n, bs))
        else:
            self._ordering = self.ordering

        self._batch_ptr = 0
        return self

    def __next__(self):
        if self._batch_ptr >= len(self._ordering):
            raise StopIteration

        idxs = self._ordering[self._batch_ptr]
        self._batch_ptr += 1

        samples = [self.dataset[int(i)] for i in idxs]
        first = samples[0]

        def to_tensor_batch(items):
            arr = np.stack(items, axis=0)
            return Tensor(arr)

        if isinstance(first, (tuple, list)):
            fields = list(zip(*samples))
            batched = [to_tensor_batch(list(f)) for f in fields]
            return tuple(batched)
        else:
            return to_tensor_batch(samples)

