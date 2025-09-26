import torch
from torch import nn
from einops import rearrange
from math import sqrt
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import torchmetrics
import torchmetrics as Metric
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import config
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from metric import DiceBCELoss, DiceLoss
import torchmetrics
from torchmetrics.classification \
    import BinaryJaccardIndex, BinaryRecall, BinaryAccuracy, \
        BinaryPrecision, BinaryF1Score, Dice
import numpy as np
from MLLA import MLLA
from Hpool import Hpool
from MFAU import Mfau,MFISA

DEVICE = config.DEVICE

# Layer Normalisation
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
# Depth-wise CNN
class DepthWiseConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, padding, stride=1, bias=True):
        super(DepthWiseConv, self).__init__()
        # Depthwise Convolution
        self.DW_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
                                 kernel_size=kernel, stride=stride, 
                                 padding=padding, groups=in_dim, bias=bias)
        # Pointwise Convolution
        self.PW_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                 kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.DW_conv(x)
        x = self.PW_conv(x)

        return x

class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()

        hidden_dim = int((in_dim + out_dim)/2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        output = self.conv_block(x)

        return output

#解码器部分的多尺度边缘精华模块 的实现代码
class ParallelCDCConv(nn.Module):
    """
    并行CDC卷积模块，包含两个分支：
    1. 主分支：标准DoubleConv（卷积-批归一化-ReLU）
    2. 边缘分支：CDC增强后的卷积
    两个分支的特征通过加权融合（可学习权重 alpha）进行合成。
    """

    def __init__(self, in_dim, out_dim, theta=0.7):
        super(ParallelCDCConv, self).__init__()

        hidden_dim = (in_dim + out_dim) // 2  # 隐藏层通道数

        # 主分支：DoubleConv (标准卷积)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        # 边缘分支：CDC增强后投影到 out_dim
        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            cdc_vg(hidden_dim, theta=theta),  # CDC模块（需要定义）
            nn.Conv2d(hidden_dim, out_dim, 3, padding=1),  # 输出投影层
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # alpha 权重用于控制两分支的融合程度

    def forward(self, x):
        """
        前向传播：通过两个分支提取特征，并加权融合
        """
        # 主分支特征
        feat_main = self.double_conv(x)  # 输出形状: (B, out_dim, H, W)

        # 边缘分支特征
        feat_edge = self.edge_branch(x)  # 输出形状: (B, out_dim, H, W)

        # 加权融合两个分支的特征
        out = self.alpha * feat_edge + (1 - self.alpha) * feat_main

        return out


class conv_upsample(nn.Module):
    """
    上采样模块：
    1. 使用ParallelCDCConv提取特征
    2. 使用双线性插值进行上采样
    """
    def __init__(self, scale, in_dim, out_dim=32, use_cdc=True, theta=0.7):
        super(conv_upsample, self).__init__()

        # 使用 ParallelCDCConv 替代原有的 DoubleConv
        self.conv = ParallelCDCConv(in_dim, out_dim, theta=theta)

        # 上采样操作，使用双线性插值
        self.upscale = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        前向传播：提取特征并进行上采样
        """
        # 使用 ParallelCDCConv 提取特征
        feat = self.conv(x)

        # 对提取的特征进行上采样
        output = self.upscale(feat)

        return output


resnet_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)

class ResNetEncoder(nn.Module):
    def __init__(self, encoder=resnet_encoder):
        super(ResNetEncoder, self).__init__()
        self.encoder1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # 64x128x128
        # self.mp = encoder.maxpool
        self.mp = Hpool(
            in_channels=64,  # ResNet 第一层输出通道数
            pool_size=(14, 14),  # 输入特征图尺寸为 112x112 时适用
            norm_layer=nn.BatchNorm2d,  # 与 ResNet 归一化层一致
            up_kwargs={'mode': 'bilinear', 'align_corners': True},  # 上采样参数
            reduction=16  # SE 注意力压缩比
        )

        self.encoder2 = nn.Sequential(
            encoder.layer1,  # 原始残差层
            # MFAU(in_channels=256)
            MFISA(in_channels=256)  # 添加注意力
        )
        self.encoder3 = nn.Sequential(
            encoder.layer2,
            # MFAU(in_channels=512)
            MFISA(512)
        )
        self.encoder4 = nn.Sequential(
            encoder.layer3,
            #MFAU(in_channels=1024)
            MFISA(1024)
        )

        self.encoder5 = encoder.layer4  # 2048x8x8

    def forward(self, x):
        output1 = self.encoder1(x)
        output2 = self.mp(output1)
        output2 = self.encoder2(output2)
        output3 = self.encoder3(output2)
        output4 = self.encoder4(output3)
        output5 = self.encoder5(output4)

        return output1, output2, output3, output4, output5


class DFT_CrackNet(pl.LightningModule):
    def __init__(self, img_size=256, patch_size=4, in_chans=3,
                 embed_dim=128, depths=[2, 4, 6, 2], num_heads=[4, 8, 16, 32],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False, learning_rate=config.LEARNING_RATE,
                 **kwargs):
        super(DFT_CrackNet, self).__init__()

        self.mix_transformer = MLLA(img_size, patch_size, in_chans, embed_dim, depths, num_heads, mlp_ratio, qkv_bias,
                                    drop_rate, drop_path_rate, norm_layer, ape, use_checkpoint, **kwargs)
        self.to_segment_conv = nn.Conv2d(5, 1, 1)

        self.reduce_channels_1 = DoubleConv(64, 64)
        self.reduce_channels_2 = DoubleConv(512, 256)
        self.reduce_channels_3 = DoubleConv(1024, 512)
        self.reduce_channels_4 = DoubleConv(2048, 1024)
        self.reduce_channels_5 = DoubleConv(2048, 1024)

        self.upsampling_1 = conv_upsample(2, 64, 1)
        self.upsampling_2 = conv_upsample(4, 256, 1)
        self.upsampling_3 = conv_upsample(8, 512, 1)
        self.upsampling_4 = conv_upsample(16, 1024, 1)
        self.upsampling_5 = conv_upsample(32, 1024, 1)

        self.cnn_encoder = ResNetEncoder(encoder=resnet_encoder)
        self.weight = 0.7

        # loss function
        self.loss_fn = DiceBCELoss()
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()

        # Overlapped area metrics (Ignore Backgrounds)
        self.jaccard_ind = BinaryJaccardIndex()
        self.dice = Dice()

        # LR
        self.lr = learning_rate

    def forward(self, x):
        mit_1, mit_2, mit_3 = self.mix_transformer(x)

        side_output1, output2, output3, output4, side_output5 = self.cnn_encoder(x)

        side_output2 = torch.concat((mit_1, output2), dim=1)  # (16,64,64,256)
        side_output3 = torch.concat((mit_2, output3), dim=1)  # (16,32,32,512)
        side_output4 = torch.concat((mit_3, output4), dim=1)  # (16,16,16,1024)

        up_side_1 = self.upsampling_1(self.reduce_channels_1(side_output1))
        up_side_2 = self.upsampling_2(self.reduce_channels_2(side_output2))
        up_side_3 = self.upsampling_3(self.reduce_channels_3(side_output3))
        up_side_4 = self.upsampling_4(self.reduce_channels_4(side_output4))
        up_side_5 = self.upsampling_5(self.reduce_channels_5(side_output5))

        to_fused = torch.concat((up_side_1, up_side_2, up_side_3, up_side_4, up_side_5), dim=1)
        to_segment = self.to_segment_conv(to_fused)
        # print("to_segment shape:", to_segment.shape)
        return to_segment, up_side_1, up_side_2, up_side_3, up_side_4, up_side_5

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, pred, y = self._common_step(batch, batch_idx)

        # 计算各种评价指标
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y_int = torch.tensor(y, dtype=torch.int32)
        dice = self.dice(pred, y_int)

        # 日志记录
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_precision': precision,
            'train_recall': re,
            'train_IOU': jaccard,
            'train_dice': dice
        }, on_step=False, on_epoch=True, prog_bar=True)

        # -----------------------------
        # 可视化：每 100 个 batch 可视化输入图像、标签和预测
        # -----------------------------
        if batch_idx % 100 == 0:
            x_vis = x[:8]  # 输入图像
            y_vis = y[:8]  # 标签图像，通常是 [B, 1, H, W]
            pred_vis = pred[:8]  # 模型预测

            # 确保维度是 [B, 1, H, W]
            if y_vis.ndim == 3:
                y_vis = y_vis.unsqueeze(1)
            if pred_vis.ndim == 3:
                pred_vis = pred_vis.unsqueeze(1)
            # 将预测结果转换为 0/1 掩码（如用 sigmoid 输出）
            binary_pred = (pred_vis > 0.5).float()

            # 将标签和预测图扩展到 3 通道（为了与输入图像并排显示）
            y_vis_rgb = y_vis.repeat(1, 3, 1, 1)
            pred_vis_rgb = binary_pred.repeat(1, 3, 1, 1)

            # 拼接：第一行输入图像，第二行 Ground Truth，第三行预测结果
            comparison = torch.cat([x_vis, y_vis_rgb, pred_vis_rgb], dim=0)
            # 拼接成网格
            grid = torchvision.utils.make_grid(comparison, nrow=8)
            # 写入 TensorBoard
            self.logger.experiment.add_image("input_GT_pred", grid, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y = torch.tensor(y, dtype=torch.int32)
        dice = self.dice(pred, y)

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1_score': f1_score,
                       'val_precision': precision, 'val_recall': re, 'val_IOU': jaccard, 'val_dice': dice},
                      on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)

        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y_int = torch.tensor(y, dtype=torch.int32)
        dice = self.dice(pred, y_int)

        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_precision': precision,
            'test_recall': re,
            'test_IOU': jaccard,
            'test_dice': dice
        }, on_step=False, on_epoch=True, prog_bar=False)

        if batch_idx % 10 == 0:
            x_vis = batch[0][:8]
            y_vis = y[:8]
            pred_vis = pred[:8]

            if y_vis.ndim == 3:
                y_vis = y_vis.unsqueeze(1)
            if pred_vis.ndim == 3:
                pred_vis = pred_vis.unsqueeze(1)

            binary_pred = (pred_vis > 0.5).float()
            y_vis_rgb = y_vis.repeat(1, 3, 1, 1)
            pred_vis_rgb = binary_pred.repeat(1, 3, 1, 1)

            comparison = torch.cat([x_vis, y_vis_rgb, pred_vis_rgb], dim=0)
            grid = torchvision.utils.make_grid(comparison, nrow=8)
            self.logger.experiment.add_image("test_input_GT_pred", grid, self.global_step)

        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred_lst = self.forward(x)
        pred = pred_lst[0]
        loss = self.loss_fn(pred, y, weight=0.2)
        # loss_recall = 1-self.recall(pred, y)
        # loss *= loss_recall
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        return loss, pred, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred = self.forward(x)
        preds = torch.sigmoid(pred)
        preds = (preds > 0.5).float()
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule,
                "monitor": "val_loss",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

