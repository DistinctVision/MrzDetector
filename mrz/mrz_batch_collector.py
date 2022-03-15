import math
from typing import Tuple, Union, Iterable

import torch

from mrz.mrz_transform_generator import MrzTransformDatasetGenerator
from mrz.mrz_transform_dataset_reader import MrzTransformDatasetReader


class MrzBatchCollector(Iterable):
    def __init__(self,
                 data: Union[MrzTransformDatasetReader, MrzTransformDatasetGenerator],
                 batch_size: int,
                 device: Union[torch.device, str] = None):
        assert 0 < batch_size <= len(data)

        self._data = data
        self._batch_size = batch_size
        self._len = int(math.ceil(len(self._data) / self._batch_size))
        self.device = device

        self._iter_index = -1

    @property
    def batch_size(self):
        return self._batch_size

    def __len__(self):
        return self._len

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._iter_index += 1
        if self._iter_index >= self._len:
            raise StopIteration
        return self[self._iter_index]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        images_list_batch = []
        labels_list_batch = []
        offset_index = self._batch_size * self._iter_index
        for i in range(self._batch_size):
            image, labels = self._data[(offset_index + i) % self._len]
            image = torch.Tensor(image).permute(2, 0, 1)
            labels = torch.Tensor(labels)
            images_list_batch.append(image)
            labels_list_batch.append(labels)
        images_batch = torch.stack(images_list_batch)
        labels_batch = torch.stack(labels_list_batch)
        if self.device:
            images_batch = images_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
        return images_batch, labels_batch
