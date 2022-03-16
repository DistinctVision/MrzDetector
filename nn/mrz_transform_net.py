from typing import Tuple, Union, List
from collections import OrderedDict
from pathlib import Path

import numpy as np

import torch

from nn.blocks import ResLayer, BasicBlock
from nn.weight_init import constant_init, kaiming_init
from mrz.transform import image_corners_to_homography, denormalize_image_corners


class MrzTransformNet(torch.nn.Module):
    # it bases on ResNet18

    @staticmethod
    def build_model(input_image_size: Tuple[int, int], device: Union[torch.device, str] = None):
        base_channels = 64
        stage_blocks = (2, 2, 2, 2)

        strides = (1, 2, 2, 2)

        res_layers = []

        in_block = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=(1, 1),
                                                       in_channels=3, out_channels=base_channels),
                                       torch.nn.BatchNorm2d(num_features=base_channels),
                                       torch.nn.ReLU(inplace=True))

        in_channels = base_channels
        for i, (num_blocks, stride) in enumerate(zip(stage_blocks, strides)):
            out_channels = base_channels * 2 ** i
            res_layer = ResLayer(blockType=BasicBlock,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 num_blocks=num_blocks,
                                 stride=stride,
                                 avg_down=False)
            in_channels = out_channels * BasicBlock.expansion
            res_layers.append(res_layer)

        res_blocks = torch.nn.Sequential(*res_layers)

        out_size = (input_image_size[0], input_image_size[1])
        for stride in strides:
            out_size = (out_size[0] // stride, out_size[1] // stride)

        out_block = torch.nn.Sequential(torch.nn.Conv2d(in_channels, 2, kernel_size=(1, 1)),
                                        torch.nn.BatchNorm2d(2),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Flatten(),
                                        torch.nn.Linear(out_size[0] * out_size[1] * 2, 32),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(32, 8))
        if device:
            in_block.to(device)
            res_blocks.to(device)
            out_block.to(device)
        return torch.nn.Sequential(OrderedDict([('in_block', in_block),
                                                ('res_blocks', res_blocks),
                                                ('out_block', out_block)]))
    
    @staticmethod
    def create(input_image_size: Tuple[int, int], mrz_code_image_size: Tuple[int, int],
               device: Union[torch.device, str]):
        net = MrzTransformNet(input_image_size, mrz_code_image_size)
        net.model = MrzTransformNet.build_model(input_image_size, device)
        net._device = device
        return net

    @staticmethod
    def load(weights_path: Union[Path, str],
             input_image_size: Tuple[int, int],
             mrz_code_image_size: Tuple[int, int],
             device: Union[torch.device, str]):
        net = MrzTransformNet(input_image_size, mrz_code_image_size)
        net.model = MrzTransformNet.build_model(input_image_size, device)
        net.model.load_state_dict(torch.load(weights_path))
        net._device = device
        return net

    def __init__(self, input_image_size: Tuple[int, int], mrz_code_image_size: Tuple[int, int]):
        super(MrzTransformNet, self).__init__()
        self._input_image_size = input_image_size
        self._mrz_code_image_size = mrz_code_image_size
        self._device = None

    @property
    def input_image_size(self) -> Tuple[int, int]:
        return self._input_image_size

    @property
    def mrz_code_image_size(self) -> Tuple[int, int]:
        return self._mrz_code_image_size

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        x = x.permute(2, 0, 1).unsqueeze(0)
        if self._device:
            x = x.to(self._device)

        x = self.model(x)

        x = x.tolist()[0]
        image_corners = [(x, y) for x, y in zip(x[0:8:2], x[1:8:2])]
        image_corners = denormalize_image_corners(image_corners, self._input_image_size)
        matrix = image_corners_to_homography(image_corners, self._mrz_code_image_size)
        return matrix
