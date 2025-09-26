import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def swish(x):  # A variant of the ReLU function
	# return x * torch.sigmoid(x)   # Computationally complex
	return x * F.relu6(x + 3) / 6  # Simpler computation
#

class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size=3,
				 stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True,
				 activation=swish, conv=nn.Conv2d,
				 ):  # 'frelu', nn.ReLU(inplace=False), sinlu
		super(BasicConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation > 1:
				padding = dilation * (kernel_size // 2)  # AtrousConv2d
			elif kernel_size == stride:
				padding = 0
			else:
				padding = kernel_size // 2  # BasicConv2d

		self.c = conv(in_channels, out_channels,
					  kernel_size=kernel_size, stride=stride,
					  padding=padding, dilation=dilation, bias=bias)

		if activation is None:
			self.a = nn.Sequential()
		else:
			self.a = activation

		self.b = nn.BatchNorm2d(out_channels) if bn else nn.Sequential()
		self.o = nn.Dropout2d(p=0.15)  # DisOut(p=.15)#

	def forward(self, x):
		x = self.c(x)
		x = self.b(x)
		x = self.o(x)
		x = self.a(x)
		return x


class BottleNeck(torch.nn.Module):  # Lightweight residual structure for feature extraction.

	MyConv = BasicConv2d
	def __init__(self, in_c, out_c, stride=1, out='dis', **args):
		super(BottleNeck, self).__init__()
		if in_c != out_c:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(out_c)
				)
		else:
			self.shortcut = nn.Sequential()
		self.conv1 = self.MyConv(in_c, out_c, 3, padding=1)
		self.conv2 = self.MyConv(out_c, out_c, 3, padding=1, activation=None)
		# self.o = nn.Dropout2d(p=0.15)#DisOut(p=.15)#
	def forward(self, x):
		out = self.conv2(self.conv1(x))  # Two convolutions
		# out = self.o(out)
		return swish(out + self.shortcut(x))  # Residual connection + Swish activation


class MFISA(nn.Module):  # Match filter inspired space attention, considering variance smoothing supervision
	def __init__(self, in_channels=32):
		super(MFISA, self).__init__()
		mid_channels = in_channels // 2
		self.filt_max = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3, 1, 1),
			nn.Conv2d(mid_channels, 1, 3, 1, 1),
			nn.BatchNorm2d(1)
		)  # filt_max: This part calculates the maximum value of the feature map using two convolution layers and batch normalization to capture the dynamic range of the features
		self.filt_std = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3, 1, 1),
			nn.Conv2d(mid_channels, 1, 3, 1, 1),
			nn.BatchNorm2d(1)
		)  # filt_std: A filter to calculate the standard deviation, used to capture the smoothness or dispersion of the feature map.
		self.filt_out = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3, 1, 1),
			nn.Conv2d(mid_channels, 1, 3, 1, 1),
			nn.BatchNorm2d(1)
		)  # filt_out: This filter combines vp2p (peak-to-peak difference) and vstd (standard deviation) to output the combined features.
		self.filt_att = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3, 1, 1),
			nn.Conv2d(mid_channels, 1, 3, 1, 1),
			nn.Sigmoid()
		)  # filt_att: The attention mechanism part, using two convolution layers and a Sigmoid activation function, generates an attention map to decide which regions of the input feature map to focus on.

	def forward(self, res):
		vmin, _ = torch.min(res, dim=1, keepdim=True)  # Calculate the minimum value along the channel dimension
		vmax, _ = torch.max(res, dim=1, keepdim=True)  # Calculate the maximum value along the channel dimension
		vp2p = vmax - vmin  # Calculate the peak-to-peak difference, measuring the dynamic range of the feature map
		vstd = torch.std(res, dim=1, keepdim=True)  # Calculate the standard deviation along the channel dimension

		att = self.filt_max(vp2p) + self.filt_std(vstd) + self.filt_out(vp2p + vstd)
		# vp2p highlights regions with large dynamic range, the std filter processes vstd to capture texture changes, and the combined filter processes vp2p+vstd to merge both pieces of information
		return self.filt_att(F.leaky_relu(att)) * res  # Attention weights adjust the input features


class MFAU(nn.Module):  # Match Filter Activated Spatial Attention Mechanism for enhancing feature representation
	__name__ = 'mfau'

	def __init__(self, in_channels=256, layers=(256, 512, 512, 1024)):
		super(MFAU, self).__init__()

		self.first = BottleNeck(in_c=in_channels, out_c=layers[0])
		self.pool = nn.Conv2d(layers[0], layers[0], kernel_size=3, stride=2, padding=1)

		self.encoders = nn.ModuleList()
		self.attenten = nn.ModuleList()

		for i in range(len(layers) - 1):
			self.encoders.append(BottleNeck(in_c=layers[i], out_c=layers[i + 1]))
			self.attenten.append(MFISA(layers[i + 1]))

	def forward(self, x0):
		x = self.first(x0)  # First convolution block
		down_activations = []

		for i, down in enumerate(self.encoders):
			down_activations.append(x)  # Record the input of the current layer (before down-sampling)

			# Down-sample and encode x
			x = down(self.pool(x))  # Down-sample and apply encoder operation
			x = self.attenten[i](x)  # Attention mechanism applied

		down_activations.append(x)  # Record the output of the final layer
		return down_activations  # Return features from the encoder stage (including features after residual connections)


def mfau(**args):
	return MFAU(**args)

import torch

# Assuming the MFAU class and related modules are already defined
# Instantiate the MFAU network
model = MFAU(in_channels=3)  # The input channels are 3

# Create a random input tensor of shape (16, 3, 256, 256)
x = torch.randn(16, 256, 64, 64)  # 16 samples, each is a 3-channel image of size 256x256

# Pass the input through the MFAU network and get the output
output = model(x)

# Print the output shape to check the feature map sizes
for i, out in enumerate(output):
    print(f"Output at layer {i} has shape: {out.shape}")


