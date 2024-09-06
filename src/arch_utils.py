# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------

import torchvision.transforms.functional as TF
from torchvision import transforms
from skimage.color import rgb2ycbcr
from skimage.io import imread
from scipy.io import loadmat
import torch.utils.data as Data
import numpy as np
from glob import glob
import torch
import h5py
import os
import json
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import kornia

EPSILON = 1e-6


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def get_coords(batch_size, H, W, fix_axis=False):
    U_coord = torch.arange(start=0, end=W).unsqueeze(0).repeat(H, 1).float()
    V_coord = torch.arange(start=0, end=H).unsqueeze(1).repeat(1, W).float()
    if not fix_axis:
        U_coord = (U_coord - ((W - 1) / 2)) / max(W, H)
        V_coord = (V_coord - ((H - 1) / 2)) / max(W, H)
    coords = torch.stack([U_coord, V_coord], dim=0)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    coords = coords.permute(0, 2, 3, 1).cuda()
    coords[..., 0] /= W - 1
    coords[..., 1] /= H - 1
    coords = (coords - 0.5) * 2
    return coords  # BS,H,W,2

class Plane2Depth_function(nn.Module):
    def __init__(self, upratio=1, max_depth=1.0):
        super(Plane2Depth_function, self).__init__()

        self.reduce_feat = torch.nn.Conv2d(4, out_channels=4, bias=False, kernel_size=1, stride=1, padding=0)

        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)
        self.relu = nn.ReLU()
        self.max_depth = max_depth
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):

        plane_eq = self.reduce_feat(feat)
        # print(plane_eq.shape)

        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)

        p = plane_eq_expanded[:, 0, :, :]
        q = plane_eq_expanded[:, 1, :, :]
        r = plane_eq_expanded[:, 2, :, :]
        s = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        norm_factor = torch.sqrt((p ** 2 + q ** 2 + r ** 2) + EPSILON)
        p = torch.div(p, norm_factor)
        q = torch.div(q, norm_factor)
        r = torch.div(r, norm_factor)
        s = s * norm_factor

        disp = (p * u + q * v + r) * s
        disp = torch.clamp(disp, min=(1 / self.max_depth)).unsqueeze(dim=1)
        # print(disp.shape)

        return (1/disp)

class pqrs2depth(nn.Module):
    def __init__(self, max_depth):
        super(pqrs2depth, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.get_coords = get_coords
        self.max_depth = max_depth

    def forward(self, x, upsample_size = None):

        if upsample_size != None:
            x = F.interpolate(x,(upsample_size[2], upsample_size[3]), mode='bilinear')

        p = x[:, 0, :, :]
        q = x[:, 1, :, :]
        r = x[:, 2, :, :]
        s = x[:, 3, :, :]

        batch_size, H, W = p.size()[0], p.size()[1], p.size()[2]

        coords = self.get_coords(batch_size, H, W)
        U_coord = coords[..., 0]
        V_coord = coords[..., 1]

        U_coord = nn.Parameter(U_coord, requires_grad=False)
        V_coord = nn.Parameter(V_coord, requires_grad=False)

        pu = p * U_coord
        qv = q * V_coord

        depth = 1 / torch.clamp((pu + qv + r) * s, min=(1/self.max_depth))


        return depth.unsqueeze(1)

class parameterized_disparity(nn.Module):
    def __init__(self, max_depth):
        super(parameterized_disparity, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_depth = max_depth
        self.get_coords = get_coords

    def forward(self, x, epoch=0):


        p = x[:, 0, :, :]
        q = x[:, 1, :, :]
        r = x[:, 2, :, :]
        s = x[:, 3, :, :] # * self.max_depth
        #s = x[:, 3, :, :]

        # TODO: refer to dispnetPQRS
        norm_factor = torch.sqrt((p ** 2 + q ** 2 + r ** 2) + EPSILON)
        p = torch.div(p, norm_factor)
        q = torch.div(q, norm_factor)
        r = torch.div(r, norm_factor)
        # s = s * norm_factor
        # print("pqrs_", torch.max(p), torch.min(p), torch.max(q), torch.min(q), \
        #       torch.max(r), torch.min(r), torch.max(s), torch.min(s))
        batch_size, H, W = x.size()[0], x.size()[2], x.size()[3]

        coords = self.get_coords(batch_size, H, W)
        U_coord = coords[..., 0]
        V_coord = coords[..., 1]

        U_coord = nn.Parameter(U_coord, requires_grad=False)
        V_coord = nn.Parameter(V_coord, requires_grad=False)

        pu = p * U_coord
        qv = q * V_coord

        disp = (pu + qv + r) * s
        disp = torch.clamp(disp, min=(1 / self.max_depth))

        return p.unsqueeze(1), q.unsqueeze(1), r.unsqueeze(1), s.unsqueeze(1), disp.unsqueeze(1)



class parameterized_depth(nn.Module):
    def __init__(self, max_depth):
        super(parameterized_depth, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_depth = max_depth
        self.get_coords = get_coords

    def forward(self, x):
        EPSILON = 1e-6

        p = x[:, 0, :, :]
        q = x[:, 1, :, :]
        r = x[:, 2, :, :]
        s = x[:, 3, :, :] # * self.max_depth
        #s = x[:, 3, :, :]

        # print(torch.max(p), torch.min(p), torch.max(q), torch.min(q), \
        #       torch.max(r), torch.min(r), torch.max(s), torch.min(s))

        # TODO: refer to dispnetPQRS
        norm_factor = torch.sqrt((p ** 2 + q ** 2 + r ** 2) + EPSILON)
        p = torch.div(p, norm_factor)
        q = torch.div(q, norm_factor)
        r = torch.div(r, norm_factor)
        s = s * norm_factor

        batch_size, H, W = x.size()[0], x.size()[2], x.size()[3]

        coords = self.get_coords(batch_size, H, W)
        U_coord = coords[..., 0]
        V_coord = coords[..., 1]

        U_coord = nn.Parameter(U_coord, requires_grad=False)
        V_coord = nn.Parameter(V_coord, requires_grad=False)

        pu = p * U_coord
        qv = q * V_coord
        disp =  torch.clamp((pu + qv + r) * s, min=(1 / self.max_depth))
        
        # depth = 1 / torch.clamp((pu + qv + r) * s, min=(1 / self.max_depth))

        return p.unsqueeze(1), q.unsqueeze(1), r.unsqueeze(1), s.unsqueeze(1), disp.unsqueeze(1)


def CA_conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                  padding=((kernel_size - 1) // 2) * dilation, bias=bias))

## Channel Attention Block
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def resample_data(input, s):
    """
        input: torch.floatTensor (N, C, H, W)
        s: int (resample factor)
    """

    assert (not input.size(2) % s and not input.size(3) % s)

    if input.size(1) == 3:
        # bgr2gray (same as opencv conversion matrix)
        input = (0.299 * input[:, 2] + 0.587 * input[:, 1] + 0.114 * input[:, 0]).unsqueeze(1)

    out = torch.cat([input[:, :, i::s, j::s] for i in range(s) for j in range(s)], dim=1)

    """
        out: torch.floatTensor (N, s**2, H/s, W/s)
    """
    return out

class ResBlock(nn.Module):
    def __init__(self, embed_ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_ch, embed_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(embed_ch, embed_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def __call__(self, x):
        res = self.body(x)
        return res + x


class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels

        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_branch = nn.Sequential(nn.ELU(inplace=True),\
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True),\
                                         nn.BatchNorm2d(num_features=self.mid),\
                                         nn.ELU(inplace=True),\
                                         nn.Conv2d(in_channels=self.mid, out_channels= self.mid, kernel_size=3, padding=1, stride=1, bias=True))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.elu(x)

        return x


class FFM(nn.Module):
    # Feature fusion module
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True)

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        if x.size() != high_x.size():
            high_x = torch.nn.functional.interpolate(high_x, (x.size()[2], x.size()[3]))
        x = x + high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, upfactor=2):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.BatchNorm2d(num_features=self.inchannels // 2), \
            nn.ELU(inplace=True), \
            nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True))

    def forward(self, x):
        x = self.adapt_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, inchannels=[256, 512, 1024, 2048], midchannels=[256, 256, 256, 512], upfactors=[2,2,2,2]):
        super(Decoder, self).__init__()

        self.inchannels = inchannels
        self.midchannels = midchannels
        self.upfactors = upfactors


        self.offset_prediction = False

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)

        self.ffm2_depth = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
        self.ffm1_depth = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
        self.ffm0_depth = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])

        self.outconv_depth = AO(inchannels=self.inchannels[0], outchannels=64, upfactor=2)



    def forward(self, image_features):
        # image_features layer1:H/4, W/4, layer2:H/8, W/8, layer3:H/16, W/16, layer4:H/32, W/32
        outputs = {}

        _,_,h,w = image_features[3].size()
        bottleneck = self.conv(image_features[3])
        bottleneck = self.conv1(bottleneck)
        bottleneck = self.upsample(bottleneck)  # layer4-up:H/16, W/16

        ## Depth branch
        outputs["depth_feat", 4] = self.ffm2_depth(image_features[2], bottleneck)   # layer3+4-up:H/8, W/8
        outputs["depth_feat", 3] = self.ffm1_depth(image_features[1], outputs["depth_feat", 4]) # layer2+3-up:H/4, W/4
        outputs["depth_feat", 2] = self.ffm0_depth(image_features[0], outputs["depth_feat", 3]) # layer1+2-up:H/2, W/2

        outputs["depth_feat", 1] = self.outconv_depth(outputs["depth_feat", 2])

        return outputs["depth_feat", 1]