import os
from typing import Iterable, Union, Tuple, List
from pathlib import Path

import cv2
import numpy as np

from tqdm import tqdm


class MrzTransformDatasetReader(Iterable):
    def __init__(self,
                 dataset_directory_path: Union[Path, str],
                 max_size: int = -1):
        self._image_paths = []
        for file in os.listdir(str(dataset_directory_path)):
            if file.endswith(".jpg") or file.endswith('.png'):
                self._image_paths.append(dataset_directory_path / file)
        if max_size > 0:
            self._image_paths = self._image_paths[:max_size]
        self._read_labels()
        self._iter_index = -1

    def _read_labels(self):
        self._labels = []
        for image_path in tqdm(self._image_paths, 'The labels reading'):
            labels_path = image_path.parent / f'{image_path.stem}.txt'
            with open(labels_path) as labels_file:
                labels_str = labels_file.read().strip()
                labels_data = [float(f) for f in labels_str.split(' ')][:8]
                self._labels.append(labels_data)

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self) -> Tuple[np.ndarray, List[float]]:
        self._iter_index += 1
        if self._iter_index >= len(self._image_paths):
            raise StopIteration
        return self[self._iter_index]

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[float]]:
        image_path = self._image_paths[index]
        image = cv2.imread(str(image_path))
        return image, self._labels[index]
