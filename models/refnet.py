import torch
import torchvision
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, channel, out_chan, reduction_ratio =2, ):
        super(SE, self).__init__()
        ### squeeze
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        ### excitation
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

        ### ??
        self.c = nn.Conv2d(channel, out_chan, 1, 1)

    def forward(self, x):
        if x.ndim == 4:
            b = x.size(0)
            c = x.size(1)
        if x.ndim == 5:
            b = x.size(1)
            c = x.size(2)
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).unsqueeze(2).unsqueeze(3)
        return self.c(x*y)

class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels, out_channels):
        """
        :param num_channels: No of input channels

        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.c = nn.Conv3d(num_channels, out_channels, 1, 1)

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return self.c(output_tensor)
    
class RefNet_basic(nn.Module):
    def __init__(self, in_chan, out_chan, se=False, maxpool=False):
        super(RefNet_basic, self).__init__()

        self.conv = nn.Conv3d(in_chan, in_chan, 1, 1)
        self.relu = nn.ReLU()

        self.se = None
        self.maxpool = None

        if se:
            # self.se = torchvision.ops.SqueezeExcitation(in_chan, out_chan)
            self.se = SE(in_chan, out_chan)
            # self.se = SpatialSELayer3D(in_chan, out_chan)
        if maxpool:
            self.maxpool = nn.MaxPool2d(2,2)
            self.conv_2 = nn.Conv2d(in_chan, out_chan, 1, 1)

    def forward(self, X):
        # X = B, C, H, W
        x = X.transpose(0,1)
        x_conv =  self.conv(X)

        t = self.relu(x_conv)
        t = self.conv(t)
        t = self.relu(t)

        y = torch.add(x_conv, t)

        if self.se is not None:
            y = y.transpose(1,0)
            y = self.se(y)
        if self.maxpool is not None:
            y = self.maxpool(y)
            y = y.transpose(1,0)
            y = self.conv_2(y)


        y = y.transpose(1,0)
        
        return y

class RefNet(nn.Module):
    def __init__(self, in_chan):
        super(RefNet, self).__init__()

        self.down_1 = RefNet_basic(in_chan, 32, maxpool=True)
        self.att_1 = RefNet_basic(in_chan, 32, se=True)
        
        self.down_2 = RefNet_basic(32, 64, maxpool=True)
        self.conv_1 = nn.Conv2d(64, 128, 1 , 1)
        self.att_2 = RefNet_basic(32, 64, se=True)

        self.deconv_1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.refbasic_1 = RefNet_basic(64, 64)

        self.deconv_2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)

        self.refbasic_2 = RefNet_basic(32, 32)


    
    def forward(self, X):
        # X : S, B, C, H, W
        # seq_number, batch_size, input_channel, height, width = X.size()
        # X = torch.reshape(X, (-1, input_channel, height, width))
        # X = torch.reshape(X, (seq_number, batch_size, X.size(1), X.size(2), X.size(3)))

        X = X.transpose(0,1)

        res_1 = self.att_1(X) # -> 32
        down_1 = self.down_1(X) # -> 32

        res_2 = self.att_2(down_1) # -> 64
        down_2 = self.down_2(down_1) # -> 64


        down_2 = down_2.transpose(1,0)
        conv_1 = self.conv_1(down_2) # -> 128
        down_2 = down_2.transpose(1,0)
        #conv_1 = torch.add(conv_1, conv_1) # ??

        deconv_1 = self.deconv_1(conv_1) # -> 64
        deconv_1 = deconv_1.transpose(1,0)
        deconv_1 = torch.add(deconv_1, res_2) # -> 64
        deconv_1 = self.refbasic_1(deconv_1) # -> 64

        deconv_2 = self.deconv_2(deconv_1) # -> 32
        deconv_2 = deconv_2.transpose(1,0)
        deconv_2 = torch.add(deconv_2, res_1) # -> 32
        deconv_2 = self.refbasic_2(deconv_2) # -> 32

        return nn.Conv2d(32, 1, 1, 1)(deconv_2) # -> 1
