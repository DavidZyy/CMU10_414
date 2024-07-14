from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms

        ### BEGIN YOUR SOLUTION
        # Read images
        with gzip.open(image_filename, 'rb') as f:
            # Read header information
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read the image data
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)  # add the C channel of it
            self.images = images / 255

        # Read labels
        with gzip.open(label_filename, 'rb') as f:
            # Read header information
            magic, num = struct.unpack(">II", f.read(8))
            # Read the label data
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
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