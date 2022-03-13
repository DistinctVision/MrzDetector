from typing import Tuple

import torch

from nn.blocks import ResLayer, BasicBlock
from weight_init import constant_init, kaiming_init


class MrzTransformNet(torch.nn.Module):
    # it bases on ResNet18

    @staticmethod
    def build_model(input_shape: Tuple[int, int]):
        # TODO down sampling

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

        res_blocks = torch.nn.Sequential(*stage_blocks)

        out_block = torch.nn.Sequential(torch.nn.Conv2d(in_channels, 2, kernel_size=(1, 1)),
                                        torch.nn.BatchNorm2d(2),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Flatten(),
                                        torch.nn.Linear(input_shape[0] * input_shape[1] * 2,
                                                        input_shape[0] * input_shape[1]),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(input_shape[0] * input_shape[1],
                                                        input_shape[0] * input_shape[1]),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.Linear(input_shape[0] * input_shape[1], 8))
        return torch.nn.Sequential(in_block, res_blocks, out_block)

    def __init__(self, input_shape: Tuple[int, int]):
        super(MrzTransformNet, self).__init__()
        self._input_shape = input_shape
        self._model = MrzTransformNet.build_model(self._input_shape)

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._input_shape

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)
        x = self.res_blocks(x)
        x = self.out_block(x)
        return x


