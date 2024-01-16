# MIT License

# Original work Copyright (c) 2018 Joris (https://github.com/jvanvugt/pytorch-unet)
# Modified work Copyright (C) 2022 Canon Medical Systems Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from math import sqrt
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from src.swish import CustomSwish
from src.ws_conv import WNConv2d

import torchvision.transforms as transforms

features = None


def get_groups(channels: int) -> int:
    """
    :param channels:
    :return: return a suitable parameter for number of groups in GroupNormalisation'.
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=6,
            padding=False,
            norm="group",
            up_mode='upconv',
            patch2loc=None):
        """
        A modified U-Net implementation [1].

        [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
            Ronneberger et al., 2015 https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
            norm (str): one of 'batch' and 'group'.
                        'batch' will use BatchNormalization.
                        'group' will use GroupNormalization.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.patch2loc = patch2loc
        if self.patch2loc is not None:
            process_hooks(self.patch2loc)
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward_down(self, x, slice_idxs):
        input = x
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)
        if self.patch2loc:
            features = patch2loc_features(input, self.patch2loc, slice_idxs, x.shape[-2:])
            x = torch.cat([x, features])
        return x, blocks

    def forward_up_without_last(self, x, blocks):
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip)

        return x

    def forward_without_last(self, x, slice_idxs):
        x, blocks = self.forward_down(x, slice_idxs)
        x = self.forward_up_without_last(x, blocks)
        return x

    def forward(self, x, slice_idxs):
        x = self.get_features(x, slice_idxs)
        return self.last(x)

    def get_features(self, x, slice_idxs):
        return self.forward_without_last(x, slice_idxs)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


import numpy as np


def process_hooks(patch2loc: torch.nn.DataParallel):
    patch2loc.module._modules['branch2'][8].register_forward_hook(save_output_feature_hook())


def save_output_feature_hook():
    def hook(model, input, output):
        global features
        features = output.cpu()

    return hook


def patch2loc_features(input: torch.Tensor, patch2loc, slice_idxs, target_shape):
    global features
    patches, kwargs = patch_tensor(input, slice_idxs)
    patch2loc(patches, kwargs)
    features = features.reshape(input.shape[0], -1, features.shape[-1])
    return reshape_with_padding(features, target_shape)


def reshape_with_padding(input: torch.Tensor, target_shape: Tuple[int]):
    # calculating padding:
    padded_input_flat = F.pad(input.flatten(), (0, np.prod(target_shape) - input.numel() % np.prod(target_shape)),
                              mode='constant', value=0)
    return padded_input_flat.reshape((input.shape[0], -1,) + (30, 30,))


def patch_tensor(input_tensor: torch.Tensor, slice_idxs: torch.Tensor) -> torch.Tensor:
    """
    Patch an input tensor into non-overlapping patches along the W and H dimensions.

    Args:
    - input_tensor (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Patches tensor.
    """

    # Calculate patch size as 0.125 times the original shape along W and H dimensions
    patch_size_w = int(input_tensor.shape[2] * 0.125)
    patch_size_h = int(input_tensor.shape[3] * 0.125)

    # Use unfold to create non-overlapping patches along W and H dimensions
    patches = input_tensor.unfold(2, patch_size_w, patch_size_w).unfold(3, patch_size_h, patch_size_h)

    patches = patches.contiguous().view(patches.size(0) * patches.size(2) * patches.size(3), patch_size_w, patch_size_h)
    patches = transforms.Resize((64, 64), antialias=True)(patches).unsqueeze(1)

    kwargs = {}
    kwargs['position'] = torch.repeat_interleave(slice_idxs['slice_idx'], dim=1, repeats=64).flatten()
    return patches, kwargs
