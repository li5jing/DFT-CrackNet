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
    """中心差分卷积与方向交叉卷积结合的特征增强模块

    功能：通过CDC卷积提取特征后，使用水平和垂直/对角交叉卷积捕捉方向信息，最后通过残差连接融合特征
    参数：
        mid_ch: 中间层通道数
        theta: CDC卷积的差分权重系数，控制特征修正强度
    """

    def __init__(self, mid_ch, theta=0.7):
        super(cdc_vg, self).__init__()
        # CDC卷积层（中心差分卷积）
        self.cdc = Conv2d_cd(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.cdc_bn = nn.BatchNorm2d(mid_ch)  # 批归一化
        self.cdc_act = nn.PReLU()  # 激活函数

        # 水平/垂直方向交叉卷积
        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3,
                                             stride=1, padding=1, bias=False, theta=theta)
        # 对角方向交叉卷积
        self.d_conv = Conv2d_Diag_Cross(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3,
                                        stride=1, padding=1, bias=False, theta=theta)
        self.vg_bn = nn.BatchNorm2d(mid_ch)
        self.vg_act = nn.PReLU()

    def forward(self, x):
        # CDC分支处理
        out_0 = self.cdc_act(self.cdc_bn(self.cdc(x)))

        # 方向交叉卷积分支
        out1 = self.h_conv(out_0)  # 水平/垂直特征
        out2 = self.d_conv(out_0)  # 对角特征
        # 特征融合 + 激活
        out = self.vg_act(self.vg_bn(0.5 * out1 + 0.5 * out2))

        # 残差连接
        return out + x


class Conv2d_cd(nn.Module):
    """中心差分卷积层
    功能：在标准卷积基础上引入中心差分项，增强梯度特征提取能力
    参数        theta: 差分项权重，控制中心差分特征的强度。中心差分：计算中心点与周围区域的差异，增强梯度响应，
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta  # 差分权重系数

    def forward(self, x):
        # 标准卷积输出
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:  # theta为0时退化为标准卷积
            return out_normal
        else:
            # 计算差分项：卷积核在空间维度求和后的简化卷积
            [C_out, C_in, H_k, W_k] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)  # 空间维度求和，得到[C_out, C_in]
            kernel_diff = kernel_diff[:, :, None, None]  # 扩展为4D张量
            # 执行快速1x1卷积近似空间差分
            out_diff = F.conv2d(x, kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)
            # 最终输出 = 标准卷积 - theta * 差分项
            return out_normal - self.theta * out_diff


class Conv2d_Hori_Veri_Cross(nn.Module):
    """水平/垂直方向交叉卷积层
    功能：通过特定权重排列，捕捉水平和垂直方向的梯度特征
    参数：
        theta: 差分项权重系数
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_Hori_Veri_Cross, self).__init__()
        # 使用1x5卷积核，后续重排为3x3的十字形核
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5),
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        # 权重重排：将1x5核转换为3x3十字形核
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.zeros(C_out, C_in, 1).to(x.device)
        # 排列成十字形（中心行/列有非零权重）
        conv_weight = torch.cat([
            tensor_zeros,
            self.conv.weight[:, :, :, 0],  # 第1列
            tensor_zeros,
            self.conv.weight[:, :, :, 1],  # 第2列
            self.conv.weight[:, :, :, 2],  # 中心列
            self.conv.weight[:, :, :, 3],  # 第4列
            tensor_zeros,
            self.conv.weight[:, :, :, 4],  # 第5列
            tensor_zeros
        ], dim=2).view(C_out, C_in, 3, 3)  # 重塑为3x3

        # 执行卷积
        out_normal = F.conv2d(x, conv_weight, self.conv.bias,
                              self.conv.stride, self.conv.padding)

        if self.theta == 0:
            return out_normal
        else:
            # 计算差分项（类似CDC）
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(x, kernel_diff, self.conv.bias,
                                self.conv.stride, padding=0,
                                groups=self.conv.groups)
            return out_normal - self.theta * out_diff


class Conv2d_Diag_Cross(nn.Module):
    """对角方向交叉卷积层
    功能：通过特定权重排列，捕捉对角线方向的梯度特征
    参数：
        theta: 差分项权重系数，
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_Diag_Cross, self).__init__()
        # 使用1x5卷积核，后续重排为3x3对角核
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5),
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        # 权重重排：将1x5核转换为3x3对角交叉形核
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.zeros(C_out, C_in, 1).to(x.device)
        # 排列成对角交叉形（四个对角有非零权重）
        conv_weight = torch.cat([
            self.conv.weight[:, :, :, 0],  # 左上
            tensor_zeros,
            self.conv.weight[:, :, :, 1],  # 右上
            tensor_zeros,
            self.conv.weight[:, :, :, 2],  # 中心
            tensor_zeros,
            self.conv.weight[:, :, :, 3],  # 左下
            tensor_zeros,
            self.conv.weight[:, :, :, 4]  # 右下
        ], dim=2).view(C_out, C_in, 3, 3)

        # 执行卷积
        out_normal = F.conv2d(x, conv_weight, self.conv.bias,
                              self.conv.stride, self.conv.padding)

        if self.theta == 0:
            return out_normal
        else:
            # 计算差分项
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(x, kernel_diff, self.conv.bias,
                                self.conv.stride, padding=0,
                                groups=self.conv.groups)
            return out_normal - self.theta * out_diff


