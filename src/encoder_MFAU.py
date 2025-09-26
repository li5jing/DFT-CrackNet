
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def swish(x):  # 是 ReLU 函数的一个变种
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x + 3) / 6  # 计算简单
#

class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size=3,
				 stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True,
				 activation=swish, conv=nn.Conv2d,
				 ):  # 'frelu',nn.ReLU(inplace=False),sinlu
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


class BottleNeck(torch.nn.Module):#轻量化残差结构，用于特征提取。

	MyConv = BasicConv2d
	def __init__(self, in_c, out_c, stride=1, out='dis', **args):
		super(BottleNeck, self).__init__()
		if in_c!=out_c:
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
		out = self.conv2(self.conv1(x))#两次卷积
		# out = self.o(out)
		return swish(out + self.shortcut(x))#残差连接 + Swish激活


class MFISA(nn.Module):  # Match filter inspired space attention，考虑对方差进行平滑性监督
	def __init__(self, in_channels=32):
		super(MFISA, self).__init__()
		mid_channels = in_channels // 2
		self.filt_max = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3, 1, 1),
			nn.Conv2d(mid_channels, 1, 3, 1, 1),
			nn.BatchNorm2d(1)
		)  # filt_max：这个部分通过两层卷积和批归一化操作来计算特征图的最大值，捕捉特征的动态范围
		self.filt_std = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3, 1, 1),
			nn.Conv2d(mid_channels, 1, 3, 1, 1),
			nn.BatchNorm2d(1)
		)  # filt_std：计算标准差的滤波器，用于捕捉特征图的平滑性或分散度。
		self.filt_out = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3, 1, 1),
			nn.Conv2d(mid_channels, 1, 3, 1, 1),
			nn.BatchNorm2d(1)
		)  # filt_out：该滤波器将 vp2p（峰值到峰值的差）和 vstd（标准差）结合，输出合并的特征。
		self.filt_att = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3, 1, 1),
			nn.Conv2d(mid_channels, 1, 3, 1, 1),
			nn.Sigmoid()
		)  # filt_att：注意力机制部分，使用两层卷积和一个 Sigmoid 激活函数，生成一个注意力图，用来决定应该关注输入特征图的哪些区域。

	def forward(self, res):
		vmin, _ = torch.min(res, dim=1, keepdim=True)  # 计算通道维度上的最小值
		vmax, _ = torch.max(res, dim=1, keepdim=True)  # 计算通道维度上的最大值
		vp2p = vmax - vmin  # 计算峰值到峰值的差值，衡量特征的动态范围
		vstd = torch.std(res, dim=1, keepdim=True)  # 计算通道维度上的标准差

		att = self.filt_max(vp2p) + self.filt_std(vstd) + self.filt_out(vp2p + vstd)
		# vp2p，突出动态范围大的区域，标准差滤波器处理vstd，捕捉纹理变化，综合滤波器，处理vp2p+vstd，融合两者的信息
		return self.filt_att(F.leaky_relu(att)) * res  # 成注意力权重，调整输入特征


class MFAU(nn.Module):  # 匹配滤波器激励的空间注意力机制，用于增强特征的表达能力
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
		x = self.first(x0)  # 第一层卷积块
		down_activations = []

		for i, down in enumerate(self.encoders):
			down_activations.append(x)  # 记录当前层的输入（下采样前的特征）

			# 对x进行下采样和编码

			x = down(self.pool(x))  # 下采样和编码器操作
			x = self.attenten[i](x)  # 注意力机制处理



		down_activations.append(x)  # 记录最后一层的输出
		return down_activations  # 返回编码器阶段的特征（包括残差连接后的特征）


def mfau(**args):
	return MFAU(**args)

import torch

# 假设已经定义了MFAU类和相关模块
# 实例化MFAU网络
model = MFAU(in_channels=3)  # 输入通道数是3

# 创建一个随机的输入张量，形状为 (16, 3, 256, 256)
x = torch.randn(16, 256, 64, 64)  # 16个样本，每个样本是3个通道，大小为256x256的图像

# 将输入传入MFAU网络，获取输出
output = model(x)

# 打印输出的形状，检查输出特征图的尺寸
for i, out in enumerate(output):
    print(f"Output at layer {i} has shape: {out.shape}")



