import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np
import math


class cdc_vg(nn.Module):
    """Feature enhancement module combining center difference convolution (CDC) and directional cross convolution

    Function: After extracting features using CDC convolution, horizontal and vertical/diagonal cross convolutions are used to capture directional information, followed by feature fusion through residual connections.
    Parameters:
        mid_ch: Number of channels in the intermediate layer
        theta: Weight coefficient for the CDC convolution difference, controlling the strength of feature correction
    """

    def __init__(self, mid_ch, theta=0.7):
        super(cdc_vg, self).__init__()
        # CDC convolution layer (center difference convolution)
        self.cdc = Conv2d_cd(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.cdc_bn = nn.BatchNorm2d(mid_ch)  # Batch normalization
        self.cdc_act = nn.PReLU()  # Activation function

        # Horizontal/vertical cross convolution
        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3,
                                             stride=1, padding=1, bias=False, theta=theta)
        # Diagonal cross convolution
        self.d_conv = Conv2d_Diag_Cross(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3,
                                        stride=1, padding=1, bias=False, theta=theta)
        self.vg_bn = nn.BatchNorm2d(mid_ch)
        self.vg_act = nn.PReLU()

    def forward(self, x):
        # CDC branch processing
        out_0 = self.cdc_act(self.cdc_bn(self.cdc(x)))

        # Directional cross convolution branches
        out1 = self.h_conv(out_0)  # Horizontal/vertical features
        out2 = self.d_conv(out_0)  # Diagonal features
        # Feature fusion + activation
        out = self.vg_act(self.vg_bn(0.5 * out1 + 0.5 * out2))

        # Residual connection
        return out + x


class Conv2d_cd(nn.Module):
    """Center difference convolution layer
    Function: Introduces a center difference term based on standard convolution to enhance gradient feature extraction
    Parameters:
        theta: Weight coefficient for the difference term, controlling the strength of the center difference feature. 
               Center difference: Calculates the difference between the center point and the surrounding area to enhance gradient response.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta  # Difference weight coefficient

    def forward(self, x):
        # Standard convolution output
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:  # Degenerates to standard convolution when theta is 0
            return out_normal
        else:
            # Calculate the difference term: simplified convolution after summing the convolution kernel over the spatial dimensions
            [C_out, C_in, H_k, W_k] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)  # Sum over the spatial dimensions to get [C_out, C_in]
            kernel_diff = kernel_diff[:, :, None, None]  # Expand to 4D tensor
            # Perform a fast 1x1 convolution to approximate the spatial difference
            out_diff = F.conv2d(x, kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)
            # Final output = standard convolution - theta * difference term
            return out_normal - self.theta * out_diff


class Conv2d_Hori_Veri_Cross(nn.Module):
    """Horizontal/Vertical cross convolution layer
    Function: Captures horizontal and vertical gradient features through specific weight arrangements
    Parameters:
        theta: Difference weight coefficient
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_Hori_Veri_Cross, self).__init__()
        # Using a 1x5 convolution kernel, rearranged to a 3x3 cross-shaped kernel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5),
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        # Weight rearrangement: convert 1x5 kernel to a 3x3 cross-shaped kernel
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.zeros(C_out, C_in, 1).to(x.device)
        # Rearranged into cross shape (non-zero weights in center row/column)
        conv_weight = torch.cat([
            tensor_zeros,
            self.conv.weight[:, :, :, 0],  # First column
            tensor_zeros,
            self.conv.weight[:, :, :, 1],  # Second column
            self.conv.weight[:, :, :, 2],  # Center column
            self.conv.weight[:, :, :, 3],  # Fourth column
            tensor_zeros,
            self.conv.weight[:, :, :, 4],  # Fifth column
            tensor_zeros
        ], dim=2).view(C_out, C_in, 3, 3)  # Reshaped to 3x3

        # Perform convolution
        out_normal = F.conv2d(x, conv_weight, self.conv.bias,
                              self.conv.stride, self.conv.padding)

        if self.theta == 0:
            return out_normal
        else:
            # Calculate the difference term (similar to CDC)
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(x, kernel_diff, self.conv.bias,
                                self.conv.stride, padding=0,
                                groups=self.conv.groups)
            return out_normal - self.theta * out_diff


class Conv2d_Diag_Cross(nn.Module):
    """Diagonal cross convolution layer
    Function: Captures diagonal gradient features through specific weight arrangements
    Parameters:
        theta: Difference weight coefficient
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_Diag_Cross, self).__init__()
        # Using a 1x5 convolution kernel, rearranged to a 3x3 diagonal cross-shaped kernel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5),
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        # Weight rearrangement: convert 1x5 kernel to a 3x3 diagonal cross-shaped kernel
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.zeros(C_out, C_in, 1).to(x.device)
        # Rearranged into diagonal cross shape (four diagonals with non-zero weights)
        conv_weight = torch.cat([
            self.conv.weight[:, :, :, 0],  # Top-left
            tensor_zeros,
            self.conv.weight[:, :, :, 1],  # Top-right
            tensor_zeros,
            self.conv.weight[:, :, :, 2],  # Center
            tensor_zeros,
            self.conv.weight[:, :, :, 3],  # Bottom-left
            tensor_zeros,
            self.conv.weight[:, :, :, 4]  # Bottom-right
        ], dim=2).view(C_out, C_in, 3, 3)

        # Perform convolution
        out_normal = F.conv2d(x, conv_weight, self.conv.bias,
                              self.conv.stride, self.conv.padding)

        if self.theta == 0:
            return out_normal
        else:
            # Calculate the difference term
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(x, kernel_diff, self.conv.bias,
                                self.conv.stride, padding=0,
                                groups=self.conv.groups)
            return out_normal - self.theta * out_diff




