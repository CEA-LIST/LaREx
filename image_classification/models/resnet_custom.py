from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch import Tensor
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from dropblock import DropBlock2D
from icecream import ic
from numpy import array
from models.resnet_ash import ash_p
from models.resnet_dice import RouteDICE

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152"
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


class AvgPoolShortCut(nn.Module):
    """
    Strided average pooling as implemented for the DDU paper
    This module replaces the 1x1 convolution down-sampling in resnet layers
    """

    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(x.shape[0], self.out_c - self.in_c, x.shape[2], x.shape[3], device=x.device, )
        x = torch.cat((x, pad), dim=1)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class BasicBlockSN(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = spectral_norm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = spectral_norm(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            input_channels: int = 3,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropblock: bool = False,
            dropblock_prob: float = 0.0,
            dropblock_location: int = 2,
            dropblock_block_size: int = 3,
            dropout: bool = False,
            dropout_prob: float = 0.0,
            activation: str = "relu",
            avg_pool: bool = False,
            ash: bool = False,
            ash_percentile: int = 80,
            dice_precompute: bool = False,
            dice_inference: bool = False,
            dice_p: int = 90,
            dice_info: Union[None, array] = None,
            react_threshold: Union[None, float] = None,
            spectral_norm_only_fc: bool = False,
    ) -> None:
        super().__init__()
        assert activation in ("relu", "leaky")
        assert dropblock_location in (1, 2, 3)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.avg_pool = avg_pool
        self.ash = ash
        self.ash_percentile = ash_percentile
        self.dice_precompute = dice_precompute
        self.dice_inference = dice_inference
        self.dice_p = dice_p
        self.react_threshold = react_threshold
        self.spectral_norm_only_fc = spectral_norm_only_fc
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.input_channels = input_channels
        self.dropblock = dropblock
        self.dropblock_prob = dropblock_prob
        self.dropblock_location = dropblock_location
        self.dropblock_block_size = dropblock_block_size
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        # network layers:
        self.conv1 = nn.Conv2d(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation=activation)
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=2, dilate=replace_stride_with_dilation[0], activation=activation)
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=2, dilate=replace_stride_with_dilation[1], activation=activation)
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=2, dilate=replace_stride_with_dilation[2], activation=activation)

        if self.dropblock:
            self.dropblock2d_layer = DropBlock2D(drop_prob=self.dropblock_prob,
                                                 block_size=self.dropblock_block_size)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.spectral_norm_only_fc:
            if self.dice_inference:
                self.fc = spectral_norm(RouteDICE(512 * block.expansion, num_classes, p=self.dice_p, info=dice_info))
            else:
                self.fc = spectral_norm(nn.Linear(512 * block.expansion, num_classes))
        else:
            if self.dice_inference:
                self.fc = RouteDICE(512 * block.expansion, num_classes, p=self.dice_p, info=dice_info)
            else:
                self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            activation: str = "relu"
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_pool:
                downsample = nn.Sequential(AvgPoolShortCut(stride, block.expansion * planes, self.inplanes))
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer, activation
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=activation
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        # ic(x1.shape)
        if self.dropblock and self.dropblock_location == 1:
            x1 = self.dropblock2d_layer(x1)
        x2 = self.layer2(x1)
        # ic(x2.shape)
        if self.dropblock and self.dropblock_location == 2:
            x2 = self.dropblock2d_layer(x2)
            # ic("x2 drop", x2.shape)
        x3 = self.layer3(x2)
        if self.dropblock and self.dropblock_location == 3:
            x3 = self.dropblock2d_layer(x3)
        # ic(x3.shape)
        x4 = self.layer4(x3)
        # ic(x4.shape)
        if self.dice_precompute:
            return x4
        x_avgpool = self.avgpool(x4)
        if self.react_threshold is not None:
            x_avgpool = x_avgpool.clip(max=self.react_threshold)
        # ic(x_avgpool.shape)
        if self.ash:
            x_avgpool = ash_p(x_avgpool, percentile=self.ash_percentile)
        x_flat = torch.flatten(x_avgpool, 1)
        # ic(x_flat.shape)

        if self.dropout:
            x_flat = self.dropout_layer(x_flat)

        x_out = self.fc(x_flat)

        return x_out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetSN(ResNet):
    def __init__(self,
                 block: Type[Union[BasicBlock, BasicBlockSN, Bottleneck]],
                 layers: List[int],
                 input_channels: int = 3,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 dropblock: bool = False,
                 dropblock_prob: float = 0.0,
                 dropblock_location: int = 2,
                 dropblock_block_size: int = 3,
                 dropout: bool = False,
                 dropout_prob: float = 0.0,
                 activation: str = "relu",
                 avg_pool: bool = False,
                 ash: bool = False,
                 ash_percentile: int = 80,
                 dice_precompute: bool = False,
                 dice_inference: bool = False,
                 dice_p: int = 90,
                 dice_info: Union[None, array] = None,
                 react_threshold: Union[None, int] = None
                 ) -> None:
        super().__init__(block,
                         layers,
                         input_channels,
                         num_classes,
                         zero_init_residual,
                         groups, width_per_group,
                         replace_stride_with_dilation,
                         norm_layer,
                         dropblock,
                         dropblock_prob,
                         dropblock_location,
                         dropblock_block_size,
                         dropout,
                         dropout_prob,
                         activation,
                         avg_pool,
                         ash,
                         ash_percentile,
                         dice_precompute,
                         dice_inference,
                         dice_p,
                         dice_info,
                         react_threshold)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], activation=activation)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], activation=activation)
        self.layer3 = self._make_layer(BasicBlockSN, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], activation=activation)
        self.layer4 = self._make_layer(BasicBlockSN, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], activation=activation)
        if self.dice_inference:
            self.fc = spectral_norm(RouteDICE(512 * block.expansion, num_classes, p=self.dice_p, info=dice_info))
        else:
            self.fc = spectral_norm(nn.Linear(512 * block.expansion, num_classes))



def _resnet(arch_name: str,
            block: Type[Union[BasicBlock, BasicBlockSN, Bottleneck]],
            layers: List[int],
            input_channels: int = 3,
            num_classes: int = 1000,  # ImageNet-1000
            dropblock: bool = False,
            dropblock_prob: float = 0.0,
            dropblock_location: int = 2,
            dropblock_block_size: int = 3,
            dropout: bool = False,
            dropout_prob: float = 0.0,
            spectral_norm: bool = False,
            activation: str = "relu",
            avg_pool: bool = False,
            ash: bool = False,
            ash_percentile: int = 80,
            dice_precompute: bool = False,
            dice_inference: bool = False,
            dice_p: int = 90,
            dice_info: Union[None, array] = None,
            react_threshold: Union[None, float] = None,
            pretrained: bool = False,
            progress: bool = True,
            spectral_norm_only_fc: bool = False,
            batch_norm: bool = True,
            **kwargs):
    if not spectral_norm:
        model = ResNet(block,
                       layers,
                       input_channels=input_channels,
                       num_classes=num_classes,
                       dropblock=dropblock,
                       dropblock_prob=dropblock_prob,
                       dropblock_location=dropblock_location,
                       dropblock_block_size=dropblock_block_size,
                       dropout=dropout,
                       dropout_prob=dropout_prob,
                       activation=activation,
                       avg_pool=avg_pool,
                       ash=ash,
                       ash_percentile=ash_percentile,
                       dice_precompute=dice_precompute,
                       dice_inference=dice_inference,
                       dice_p=dice_p,
                       dice_info=dice_info,
                       react_threshold=react_threshold,
                       spectral_norm_only_fc=spectral_norm_only_fc,
                       norm_layer=None if batch_norm else nn.Identity,
                       **kwargs)
    else:
        model = ResNetSN(block,
                         layers,
                         input_channels=input_channels,
                         num_classes=num_classes,
                         dropblock=dropblock,
                         dropblock_prob=dropblock_prob,
                         dropblock_location=dropblock_location,
                         dropblock_block_size=dropblock_block_size,
                         dropout=dropout,
                         dropout_prob=dropout_prob,
                         activation=activation,
                         avg_pool=avg_pool,
                         ash=ash,
                         ash_percentile=ash_percentile,
                         dice_precompute=dice_precompute,
                         dice_inference=dice_inference,
                         dice_p=dice_p,
                         dice_info=dice_info,
                         react_threshold=react_threshold,
                         norm_layer=None if batch_norm else nn.Identity,
                         **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch_name], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(input_channels=3,
             num_classes=1000,
             dropblock=False,
             dropblock_prob=0.0,
             dropblock_location=2,
             dropblock_block_size=3,
             dropout=False,
             dropout_prob=0.0,
             pretrained=False,
             progress=True,
             spectral_norm=False,
             activation="relu",
             avg_pool=False,
             ash=False,
             ash_percentile=80,
             dice_precompute=False,
             dice_inference=False,
             dice_p=90,
             dice_info=None,
             react_threshold=None,
             spectral_norm_only_fc=False,
             batch_norm=True,
             **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18',
                   BasicBlockSN if spectral_norm else BasicBlock,
                   [2, 2, 2, 2],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropblock_location,
                   dropblock_block_size,
                   dropout,
                   dropout_prob,
                   spectral_norm,
                   activation,
                   avg_pool,
                   ash,
                   ash_percentile,
                   dice_precompute,
                   dice_inference,
                   dice_p,
                   dice_info,
                   react_threshold,
                   pretrained,
                   progress,
                   spectral_norm_only_fc,
                   batch_norm,
                   **kwargs)


def resnet34(input_channels=3,
             num_classes=1000,
             dropblock=False,
             dropblock_prob=0.0,
             dropblock_block_size=3,
             dropout=False,
             dropout_prob=0.0,
             pretrained=False,
             progress=True,
             **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34',
                   BasicBlock,
                   [3, 4, 6, 3],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropblock_block_size,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


def resnet50(input_channels=3,
             num_classes=1000,
             dropblock=False,
             dropblock_prob=0.0,
             dropblock_block_size=3,
             dropout=False,
             dropout_prob=0.0,
             pretrained=False,
             progress=True,
             **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50',
                   Bottleneck,
                   [3, 4, 6, 3],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropblock_block_size,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


def resnet101(input_channels=3,
              num_classes=1000,
              dropblock=False,
              dropblock_prob=0.0,
              dropblock_block_size=3,
              dropout=False,
              dropout_prob=0.0,
              pretrained=False,
              progress=True,
              **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101',
                   Bottleneck,
                   [3, 4, 23, 3],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropblock_block_size,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


def resnet152(input_channels=3,
              num_classes=1000,
              dropblock=False,
              dropblock_prob=0.0,
              dropblock_block_size=3,
              dropout=False,
              dropout_prob=0.0,
              pretrained=False,
              progress=True,
              **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152',
                   Bottleneck,
                   [3, 8, 36, 3],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropblock_block_size,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


if __name__ == "__main__":
    sample = torch.randn(1, 3, 128, 128)
    resnet18_model = resnet18(num_classes=10,
                              dropblock=True,
                              dropblock_prob=0.5,
                              dropblock_block_size=6,
                              dropout=True,
                              dropout_prob=0.3,
                              spectral_norm=True)

    resnet18_model.eval()
    ic(resnet18_model.layer1)
    ic(resnet18_model.layer2)
    ic(resnet18_model.layer3)
    ic(resnet18_model.layer4)
    ic(resnet18_model.dropblock2d_layer.drop_prob)
    ic(resnet18_model.dropblock2d_layer.block_size)
    ic(sample.shape)
    resnet18_model(sample)
