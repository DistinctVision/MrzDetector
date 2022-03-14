import math
from typing import Union
import torch
import numpy as np

import cv2

from mrz.mrz_transform_generator import MrzTransformDatasetGenerator
from mrz.mrz_transform_dataset_reader import MrzTransformDatasetReader
from mrz.transform import image_corners_to_homography, denormalize_image_corners


class MrzCollageBuilder:
    def __init__(self,
                 data: Union[MrzTransformDatasetReader, MrzTransformDatasetGenerator],
                 model: torch.nn.Module,
                 size: int = 4,
                 index: int = 0):
        self._data = data
        self._model = model
        self._size = size
        self._index = index
        self._mrz_code_image_size = data.mrz_code_image_size
        self._input_image_size = data.input_image_size
        side = int(math.ceil(math.sqrt(self._size)))
        self._collage_size = (side, int(math.ceil(self._size / side)))
        self._canvas = np.zeros((self._collage_size[0] * self._mrz_code_image_size[0],
                                 self._collage_size[1] * self._mrz_code_image_size[1], 3), np.uint8)
        self._canvas[:, :] = (255, 255, 255)

    def build(self) -> np.ndarray:
        len_data = len(self._data)
        for i in range(self._size):
            offset = ((i % self._collage_size[0]) * self._mrz_code_image_size[0],
                      (i // self._collage_size[1]) * self._mrz_code_image_size[1])
            image, labels = self._data[(i + self._index) % len_data]
            x = self._model(image)
            image_corners = [(x, y) for x, y in zip(x[0:8:1, 1:8:1])]
            image_corners = denormalize_image_corners(image_corners, self._input_image_size)
            homography = image_corners_to_homography(image_corners, self._mrz_code_image_size)
            mzt_code_image = cv2.warpAffine(image, homography, self._mrz_code_image_size)
            self._canvas[offset[1]:offset[1]+self._mrz_code_image_size[1],
                         offset[0]:offset[0]+self._mrz_code_image_size[0]] = mzt_code_image
        return self._canvas
