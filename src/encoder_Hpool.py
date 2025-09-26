import torch
import torch.nn as nn
import torch.nn.functional as F



class SELayer(nn.Module):
   

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 


class PyramidPooling(nn.Module):
  

    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = in_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=False)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=False)
        )
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), dim=1) 


class StripPooling(nn.Module):


    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((1, pool_size[0])) 
        self.pool_w = nn.AdaptiveAvgPool2d((pool_size[1], 1))  

        inter_channels = in_channels // 4
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=False)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, 1), bias=False),
            norm_layer(inter_channels)
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(1, 0), bias=False),
            norm_layer(inter_channels)
        )
        self.rebuild = nn.Sequential(
            nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False),
            norm_layer(in_channels)
        )
        self._up_kwargs = up_kwargs

    def forward(self, x):
        identity = x
        x = self.conv1x1(x)

   
        h = self.pool_h(x)
        h = F.interpolate(h, size=x.shape[2:], **self._up_kwargs)
        h = self.conv_h(h)

 
        w = self.pool_w(x)
        w = F.interpolate(w, size=x.shape[2:], **self._up_kwargs)
        w = self.conv_w(w)


        combined = torch.cat([h, w], dim=1)
        out = self.rebuild(combined)
        return F.relu(identity + out)  # 非原地激活



class Hpool(nn.Module):
  

    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs, reduction=16):
        super(Hpool, self).__init__()
        # PyramidPooling分支
        self.pyramid = PyramidPooling(in_channels, norm_layer, up_kwargs)
        self.pyramid_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1, bias=False),  # 通道压缩
            norm_layer(in_channels),
            nn.ReLU(inplace=False)
        )

   
        self.strip = StripPooling(in_channels, pool_size, norm_layer, up_kwargs)

     
        self.se = SELayer(2 * in_channels, reduction)

    
        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=False)
        )

       
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        identity = x.clone() 

   
        pyramid_feat = self.pyramid(x)
        pyramid_feat = self.pyramid_conv(pyramid_feat)  

  
        strip_feat = self.strip(x) 

     
        combined = torch.cat([pyramid_feat, strip_feat], dim=1)  
        se_weighted = self.se(combined)

      
        out = self.final_conv(se_weighted)
        out = out + identity 

       
        out = self.downsample(F.relu(out))  # 输出尺寸: (B, C, H/2, W/2)
        return out



if __name__ == "__main__":
  
    in_channels = 64
    pool_size = (14, 14) 
    norm_layer = nn.BatchNorm2d
    up_kwargs = {'mode': 'bilinear', 'align_corners': True}

  
    hpool = Hpool(
        in_channels=in_channels,
        pool_size=pool_size,
        norm_layer=norm_layer,
        up_kwargs=up_kwargs,
        reduction=16
    )

 
    x = torch.randn(2, in_channels, 112, 112)
    out = hpool(x)

    print(f"输入尺寸: {x.shape} -> 输出尺寸: {out.shape}")  # 应为 (2, 64, 56, 56)
