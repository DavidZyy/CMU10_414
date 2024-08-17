import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset


"""
A dataset class for CIFAR-10
the codes here is refrenced from torchvision/datasets/cifar.py/CIFAR10
"""
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        train_list = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
        ]

        test_list = [
            "test_batch",
        ]

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
        # raise NotImplementedError()
        self.transforms = transforms

        self.train = train

        if self.train:
            downloaded_list = train_list
        else:
            downloaded_list = test_list

        # self.data: Any = []
        # self.targets = []
        self.X: Any = []
        self.y = []

        for file_name in downloaded_list:
            file_path = os.path.join(base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.X.append(entry["data"])
                if "labels" in entry:
                    self.y.extend(entry["labels"])
                else:
                    self.y.extend(entry["fine_labels"])

        # if train is true, shape is (50000, 3, 32, 32), else (10000, 3, 32, 32), -1 
        # means the num of images
        self.X = np.vstack(self.X).reshape(-1, 3, 32, 32)
        # self.X = self.X / 255.0
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img, target = self.X[index], self.y[index]

        if self.transforms:
            for transform in self.transforms:
                image = transform(image)

        return img, target
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
