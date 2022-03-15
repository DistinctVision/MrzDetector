from typing import Union, Tuple, Optional, Type
import torch.nn


class SeparableConv2D(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: int = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device: Union[str, torch.device] = None,
                 dtype: torch.dtype = None) -> None:
        super().__init__()

        self.depthwise_conv = torch.nn.Conv2d(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=in_channels,
                                              bias=bias,
                                              padding_mode=padding_mode,
                                              device=device,
                                              dtype=dtype)

        self.pointwise_conv = torch.nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=(1, 1),
                                              stride=(1, 1),
                                              padding=0,
                                              dilation=(1, 1),
                                              groups=1,
                                              bias=bias,
                                              padding_mode='zeros',
                                              device=device,
                                              dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, stride=stride, dilation=dilation,
                                     kernel_size=(3, 3), padding=dilation)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, stride=(1, 1), dilation=(1, 1),
                                     kernel_size=(3, 3), padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 downsample: Optional[torch.nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = torch.nn.Conv2d(in_channels, width, kernel_size=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(width)
        self.conv2 = torch.nn.Conv2d(width, width, kernel_size=(3, 3), stride=stride, dilation=dilation, groups=groups,
                                     padding=dilation)
        self.bn2 = torch.nn.BatchNorm2d(width)
        self.conv3 = torch.nn.Conv2d(width, out_channels * self.expansion, kernel_size=(1, 1))
        self.bn3 = torch.nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResLayer(torch.nn.Sequential):
    def __init__(self,
                 blockType: Union[Type[BasicBlock], Type[Bottleneck]],
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 avg_down: bool = False):

        downsample = None
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(torch.nn.AvgPool2d(kernel_size=(stride, stride), stride=(stride, stride),
                                                     ceil_mode=True, count_include_pad=False))
            downsample.extend([
                torch.nn.Conv2d(in_channels, out_channels * blockType.expansion,
                                kernel_size=(1, 1), stride=(conv_stride, conv_stride),
                                bias=False),
                torch.nn.BatchNorm2d(out_channels * blockType.expansion)
            ])
            downsample = torch.nn.Sequential(*downsample)

        layers = [blockType(in_channels=in_channels,
                            out_channels=out_channels,
                            stride=stride,
                            downsample=downsample)]
        in_channels = out_channels * blockType.expansion
        for i in range(1, num_blocks):
            layers.append(blockType(in_channels=in_channels,
                                    out_channels=out_channels,
                                    stride=1))
        super(ResLayer, self).__init__(*layers)