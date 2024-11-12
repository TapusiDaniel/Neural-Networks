import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

def get_conv_weight_and_bias(
        filter_size: Tuple[int, int],
        num_groups: int,
        input_channels: int,
        output_channels: int,
        bias: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input_channels % num_groups == 0, "input channels must be divisible by groups number"
    assert output_channels % num_groups == 0, "output channels must be divisible by groups number"
    
    weight = torch.randn(output_channels, input_channels // num_groups, *filter_size)
    bias_vector = torch.ones(output_channels) if bias else None
    return weight, bias_vector

class MyConvStub:
    def __init__(
            self,
            kernel_size: Tuple[int, int],
            num_groups: int,
            input_channels: int,
            output_channels: int,
            bias: bool,
            stride: int,
            dilation: int,
    ):
        self.weight, self.bias = get_conv_weight_and_bias(
            kernel_size, num_groups, input_channels, output_channels, bias)
        self.groups = num_groups
        self.stride = stride
        self.dilation = dilation
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        kernel_h, kernel_w = self.weight.shape[2:]
        
        # Output dimensions
        out_h = (height - (kernel_h - 1) * self.dilation - 1) // self.stride + 1
        out_w = (width - (kernel_w - 1) * self.dilation - 1) // self.stride + 1
        
        # Output tensor
        output = torch.zeros(batch_size, self.output_channels, out_h, out_w)
        
        channels_per_group = self.input_channels // self.groups
        filters_per_group = self.output_channels // self.groups
        
        # Patch extraction and reshape
        patches = []
        for i in range(0, height - (kernel_h - 1) * self.dilation - 1 + 1, self.stride):
            for j in range(0, width - (kernel_w - 1) * self.dilation - 1 + 1, self.stride):
                patch = x[:, :, 
                        i:i + (kernel_h-1)*self.dilation + 1:self.dilation,
                        j:j + (kernel_w-1)*self.dilation + 1:self.dilation]
                patch = patch.reshape(batch_size, in_channels, -1)
                patches.append(patch)
        
        # Stack all the patches
        patches = torch.stack(patches, dim=-1) 
        patches = patches.reshape(batch_size, in_channels * kernel_h * kernel_w, -1)
        L = patches.shape[-1]
        
        for g in range(self.groups):
            # Weight of the group
            group_weight = self.weight[g*filters_per_group:(g+1)*filters_per_group]
            weight_flat = group_weight.view(filters_per_group, -1)
            
            # Patches of the group
            group_patches = patches[:, (g*channels_per_group*kernel_h*kernel_w):
                                    ((g+1)*channels_per_group*kernel_h*kernel_w)]
            group_patches = group_patches.transpose(1, 2)
            
            # convolution for this group 
            group_output = torch.matmul(group_patches, weight_flat.t())
            group_output = group_output.transpose(1, 2)
            
            # Reshape to match the output size
            output[:, g*filters_per_group:(g+1)*filters_per_group] = group_output.view(
                batch_size, filters_per_group, out_h, out_w)
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output

class MyFilterStub:
    def __init__(
            self,
            filter: torch.Tensor,
            input_channels: int,
    ):
        self.weight = filter
        self.input_channels = input_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        kernel_h, kernel_w = self.weight.shape
        
        # Output dimensions
        out_h = height - kernel_h + 1
        out_w = width - kernel_w + 1
        
        output = torch.zeros(batch_size, channels, out_h, out_w)
        
        # Apply filter
        for i in range(out_h):
            for j in range(out_w):
                patch = x[..., i:i+kernel_h, j:j+kernel_w]
                output[..., i, j] = torch.sum(patch * self.weight, dim=(-1, -2))
                
        return output