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
        #zyy: add for this
        device=None,
        dtype="float32",
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.current_batch_index = -1
        # for example a = np.array_split(np.arange(50000), range(100, 50000, 100))
        # will generate a list of ndarrays, each of which have 100 elements.
        if self.shuffle:
            self.ordering = np.random.permutation(len(self.dataset))
            self.ordering = np.array_split(self.ordering,
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        self.current_batch_index += 1
        if self.current_batch_index >= len(self.ordering):
            raise StopIteration

        batch_indices = self.ordering[self.current_batch_index]
        batch = [self.dataset[idx] for idx in batch_indices]  # batch is a list of tuple, (img, label)

        # batch[i][0] is the img
        # batch[i][1] is the label
        img_batch = [batch[i][0] for i in range(len(batch))]
        label_batch = [batch[i][1] for i in range(len(batch))]
        img = Tensor(np.stack(img_batch, axis=0), device=self.device, dtype=self.dtype)  # axis=0 means the number of samples
        label = Tensor(np.stack(label_batch, axis=0), device=self.device, dtype=self.dtype)
        return (img, label)
        ### END YOUR SOLUTION

