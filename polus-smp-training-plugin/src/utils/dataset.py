from pathlib import Path
from typing import Dict
from typing import Union

import numpy
import torch
import torchvision
from bfio import BioReader
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from .helpers import Tiles

__all__ = [
    'Dataset',
]


class LocalNorm(object):
    def __init__(
            self,
            window_size: int = 129,
            max_response: Union[int, float] = 6,
    ):
        assert window_size % 2 == 1, 'window_size must be an odd integer'

        self.window_size: int = window_size
        self.max_response: float = float(max_response)
        self.pad = torchvision.transforms.Pad(window_size // 2 + 1, padding_mode='reflect')
        # Mode can be 'test', 'train' or 'eval'.
        self.mode: str = 'eval'

    def __call__(self, x: Tensor):
        return torch.clip(
            self.local_response(self.pad(x)),
            min=-self.max_response,
            max=self.max_response,
        )

    def image_filter(self, image: Tensor) -> Tensor:
        """ Use a box filter on a stack of images
        This method applies a box filter to an image. The input is assumed to be a
        4D array, and should be pre-padded. The output will be smaller by
        window_size - 1 pixels in both width and height since this filter does not pad
        the input to account for filtering.
        """
        integral_image: Tensor = image.cumsum(dim=-1).cumsum(dim=-2)
        return (
                integral_image[..., :-self.window_size - 1, :-self.window_size - 1]
                + integral_image[..., self.window_size:-1, self.window_size:-1]
                - integral_image[..., self.window_size:-1, :-self.window_size - 1]
                - integral_image[..., :-self.window_size - 1, self.window_size:-1]
        )

    def local_response(self, image: Tensor):
        """ Regional normalization.
        This method normalizes each pixel using the mean and standard deviation of
        all pixels within the window_size. The window_size parameter should be
        2 * radius + 1 of the desired region of pixels to normalize by. The image should
        be padded by window_size // 2 on each side.
        """
        local_mean: Tensor = self.image_filter(image) / (self.window_size ** 2)
        local_mean_square: Tensor = self.image_filter(image.pow(2)) / (self.window_size ** 2)

        # Use absolute difference because sometimes error causes negative values
        local_std = torch.clip(
            (local_mean_square - local_mean.pow(2)).abs().sqrt(),
            min=1e-3,
        )

        min_i, max_i = self.window_size // 2, -self.window_size // 2 - 1
        response = image[..., min_i:max_i, min_i:max_i]

        return (response - local_mean) / local_std


class Dataset(TorchDataset):
    def __init__(self, labels_map: Dict[Path, Path], tile_map: Tiles):
        self.labels_paths: Dict[Path, Path] = labels_map
        self.tiles: Tiles = tile_map
        self.preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            LocalNorm(),
        ])

    def __getitem__(self, index: int):
        image_path, y_min, y_max, x_min, x_max = self.tiles[index]
        label_path = self.labels_paths[image_path]

        # read and preprocess image
        with BioReader(image_path) as reader:
            image_tile = reader[y_min:y_max, x_min:x_max, 0, 0, 0]
        image_tile = numpy.asarray(image_tile, dtype=numpy.float32)
        image_tile = self.preprocessing(image_tile).numpy()

        # read and preprocess label
        with BioReader(label_path) as reader:
            label_tile = reader[y_min:y_max, x_min:x_max, 0, 0, 0]
        label_tile = numpy.asarray(label_tile, dtype=numpy.float32)
        # label_tile = numpy.reshape(label_tile, (1, y_max - y_min, x_max - x_min))

        return image_tile, label_tile

    def __len__(self):
        return len(self.tiles)
