import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from arch_utils import *


class PAI(nn.Module):
    def __init__(self, nf):
        super(PAI, self).__init__()
        self.nf = nf
        self.conv_offset = nn.Sequential(
            nn.Conv2d(in_channels=self.nf, out_channels=self.nf//4, kernel_size=3, padding=1,
                      stride=1, bias=True),
            nn.BatchNorm2d(num_features=self.nf // 4), 
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=self.nf//4, out_channels=3, kernel_size=3, padding=1,
                      stride=1, bias=True))

        self.conv_parameterized_depth = nn.Sequential(
            nn.Conv2d(in_channels=self.nf, out_channels=self.nf//4, kernel_size=3, padding=1,
                      stride=1, bias=True),
            nn.BatchNorm2d(num_features=self.nf // 4), 
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=self.nf//4, out_channels=4, kernel_size=3, padding=1,
                      stride=1, bias=True))

        self.parameterized_depth = parameterized_disparity(1.0)
        self.pqrs2depth = pqrs2depth(1.0)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.offset_threshold = 0.1

        self.conv_depth2feature = nn.Conv2d(in_channels=1, out_channels=self.nf//2, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_depthMapping = nn.Conv2d(in_channels=self.nf, out_channels=self.nf//2, kernel_size=3, padding=1, stride=1, bias=True)

        self.conv_fusion = nn.Conv2d(in_channels=self.nf, out_channels=self.nf//2, kernel_size=3, padding=1, stride=1, bias=True)

        if self.training:
            self.upconv1 = nn.Conv2d(self.nf//2, self.nf//2 * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.nf//2, self.nf//2 * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.fusion2Depth = nn.Conv2d(in_channels=self.nf//2, out_channels=1, kernel_size=3, padding=1, stride=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, depth_feature, rgb_feature):
        depth2offset = self.conv_offset(depth_feature)
        confidence_map = self.sigmoid(depth2offset[:, 0, :, :]).unsqueeze(1)  # BS, 1, H, W
        offset = self.tanh(depth2offset[:, 1:, :, :]) * float(self.offset_threshold)  # BS, 2, H, W

        BS, _, H, W = depth_feature.shape
        coords = get_coords(BS, H, W, fix_axis=True)
        coords = nn.Parameter(coords, requires_grad=False)
        ocoords_orig = nn.Parameter(coords, requires_grad=False)

        offset = offset.permute(0, 2, 3, 1)  # BS, H, W, 2
        ocoords = coords + offset  # BS, H, W, 2
        ocoords = torch.clamp(ocoords, min=-1.0, max=1.0)  # BS, H, W, 2


        if int(3) > 0:  # ITERATIVE_REFINEMENT=3
            for _ in range(0, int(3)):
                du = offset[:, :, :, 0].unsqueeze(1)  # BS, H, W
                dv = offset[:, :, :, 1].unsqueeze(1)  # BS, H, W
                du = du + F.grid_sample(du, ocoords, padding_mode="zeros", align_corners=True)
                dv = dv + F.grid_sample(dv, ocoords, padding_mode="zeros", align_corners=True)
                offset = torch.cat([du, dv], dim=1)  # BS, 2, H, W
                offset = offset.permute(0, 2, 3, 1)  # BS, H, W, 2
                ocoords = ocoords_orig + offset  # BS, H, W, 2
                ocoords = torch.clamp(ocoords, min=-1.0, max=1.0)

        parameterizes = self.conv_parameterized_depth(rgb_feature)
        p1, q1, r1, s1, rgb2depth = self.parameterized_depth(parameterizes)  # BS, 1, H, W
        rgb2depth = 1./rgb2depth
        p1 = F.grid_sample(p1, ocoords, padding_mode="border", align_corners=True)
        q1 = F.grid_sample(q1, ocoords, padding_mode="border", align_corners=True)
        r1 = F.grid_sample(r1, ocoords, padding_mode="border", align_corners=True)
        s1 = F.grid_sample(s1, ocoords, padding_mode="border", align_corners=True)

        depth_offset = self.pqrs2depth(torch.cat([p1, q1, r1, s1], dim=1))
        depth_FromRGB = (1 - confidence_map) * rgb2depth + confidence_map * depth_offset

        feature_FromRGB = self.lrelu(self.conv_depth2feature(depth_FromRGB))
        feature_Fromdepth = self.lrelu(self.conv_depthMapping(depth_feature))

        fusion = self.conv_fusion(torch.cat([feature_Fromdepth, feature_FromRGB], dim=1))
        if self.training:
            depth_FromFusion = self.lrelu(self.pixel_shuffle(self.upconv1(fusion)))
            depth_FromFusion = self.lrelu(self.pixel_shuffle(self.upconv2(depth_FromFusion)))
            depth_FromFusion = self.fusion2Depth(depth_FromFusion)
            return fusion, depth_FromFusion, rgb2depth, depth_offset
        else:
            return fusion




class PGR(nn.Module):
    def __init__(self, arg):
        super(PGR, self).__init__()
        self.nf = arg.embed_ch
        self.back_RBs = arg.back_RBs

        self.conv_depth2Plane = nn.Sequential(
            nn.Conv2d(in_channels=self.nf, out_channels=self.nf//8, kernel_size=3, padding=1,
                      stride=1, bias=True),
            nn.BatchNorm2d(num_features=self.nf // 8), 
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=self.nf//8, out_channels=4, kernel_size=3, padding=1,
                      stride=1, bias=True))
        self.depth_mapping = ResBlock(self.nf)

        self.conv_MP = nn.Conv2d(self.nf//2 + 4, 4, 3, 1, 1, bias=True)

        self.conv_parameterized_depth = nn.Sequential(
            nn.Conv2d(in_channels=self.nf, out_channels=self.nf//4, kernel_size=3, padding=1,
                      stride=1, bias=True),
            nn.BatchNorm2d(num_features=self.nf // 4), 
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=self.nf//4, out_channels=4, kernel_size=3, padding=1,
                      stride=1, bias=True))

        self.parameterized_depth = parameterized_disparity(1.0)

        # attention
        self.Q_1 = nn.Conv2d(4, self.nf, 3, 1, 1, bias=True)
        self.Q_2 = nn.Conv2d(self.nf, self. nf, 3, 1, 1, bias=True)

        self.K_1 = nn.Conv2d(4, self.nf, 3, 1, 1, bias=True)
        self.K_2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        self.V = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        # Rec
        self.fusion = nn.Conv2d(2 * self.nf, self.nf, 1, 1, bias=True)

        RCABlocks = functools.partial(ResidualGroup, CA_conv, self.nf, kernel_size=3, reduction=16,
                                      act=nn.LeakyReLU(), res_scale=1, n_resblocks=arg.n_rcablocks)

        self.recon_trunk = make_layer(RCABlocks, self.back_RBs)
        self.upconv1 = nn.Conv2d(self.nf, self.nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(self.nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, feat_depth, feat_rgb, feat_fusion):
        depth2Plane = self.conv_depth2Plane(feat_depth)
        p0, q0, r0, s0, disp0 = self.parameterized_depth(depth2Plane)
        depth2Plane = torch.cat([p0, q0, r0, s0], dim=1)

        mutualPlane = self.lrelu(self.conv_MP(torch.cat([depth2Plane, feat_fusion], dim=1)))
        
        plane_FromRGB = self.conv_parameterized_depth(feat_rgb)
        p1, q1, r1, s1, _ = self.parameterized_depth(plane_FromRGB)
        plane_FromRGB = torch.cat([p1, q1, r1, s1], dim=1)

        Q = self.Q_2(self.Q_1(mutualPlane))    # [B, 4, H, W]
        K = self.K_2(self.K_1(plane_FromRGB))  # [B, 4, H, W]
        V = self.V(feat_rgb)

        Att_map = torch.sum(Q * K, 1).unsqueeze(1)  # B, 1, H, W
        Att_map = torch.sigmoid(Att_map)            # B, 1, H, W

        RGB2Depth = V * Att_map
        feat_depth = self.depth_mapping(feat_depth)

        Fusion = self.lrelu(self.fusion(torch.cat([RGB2Depth, feat_depth], dim=1)))

        Result = self.recon_trunk(Fusion)
        Result = self.lrelu(self.pixel_shuffle(self.upconv1(Result)))
        Result = self.lrelu(self.pixel_shuffle(self.upconv2(Result)))
        Result = self.lrelu(self.HRconv(Result))
        Result = self.conv_last(Result)

        return Result





class PDSR(nn.Module):
    def __init__(self, arg):
        super(PDSR, self).__init__()
        self.nf = arg.embed_ch
        self.front_RBs = arg.front_RBs
        # self.back_RBs = arg.back_RBs

        RCABlocks = functools.partial(ResidualGroup, CA_conv, self.nf, kernel_size=3, reduction=16,
                                      act=nn.LeakyReLU(), res_scale=1, n_resblocks=arg.n_rcablocks)


        RCABlocks_shared = functools.partial(ResidualGroup, CA_conv, self.nf//2, kernel_size=3, reduction=8,
                                      act=nn.LeakyReLU(), res_scale=1, n_resblocks=arg.n_rcablocks)

        self.conv_depth = nn.Conv2d(16, self.nf, 3, 1, 1, bias=True)
        self.conv_rgb = nn.Conv2d(16, self.nf, 3, 1, 1, bias=True)
        self.conv_shared = nn.Conv2d(16, self.nf//2, 3, 1, 1, bias=True)

        self.feature_extraction_depth = make_layer(RCABlocks, self.front_RBs)
        self.feature_extraction_rgb = make_layer(RCABlocks, self.front_RBs)
        self.feature_extraction_shared = make_layer(RCABlocks_shared, self.front_RBs)

        self.conv_depth_semi = nn.Conv2d(self.nf + self.nf//2, self.nf, 3, 1, 1, bias=True)
        self.conv_rgb_semi = nn.Conv2d(self.nf + self.nf//2, self.nf, 3, 1, 1, bias=True)



        #### features interaction
        self.PAI_1 = PAI(self.nf)
        self.conv_depth_mapping1 = nn.Conv2d(self.nf + self.nf//2, self.nf, 3, 1, 1, bias=True)
        self.conv_rgb_mapping1 = nn.Conv2d(self.nf + self.nf//2, self.nf, 3, 1, 1, bias=True)

        self.PAI_2 = PAI(self.nf)
        self.conv_depth_mapping2 = nn.Conv2d(self.nf + self.nf//2, self.nf, 3, 1, 1, bias=True)
        self.conv_rgb_mapping2 = nn.Conv2d(self.nf + self.nf//2, self.nf, 3, 1, 1, bias=True)

        self.PAI_3 = PAI(self.nf)
        self.conv_depth_mapping3 = nn.Conv2d(self.nf + self.nf//2, self.nf, 3, 1, 1, bias=True)
        self.conv_rgb_mapping3 = nn.Conv2d(self.nf + self.nf//2, self.nf, 3, 1, 1, bias=True)

        #### Reconstruction
        self.PGR = PGR(arg)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        #### features fusion

    def forward(self, depth, rgb):
        depth = resample_data(depth, 4)
        rgb = resample_data(rgb, 4)

        depth_spc = self.feature_extraction_depth(self.lrelu(self.conv_depth(depth)))
        rgb_spc = self.feature_extraction_rgb(self.lrelu(self.conv_rgb(rgb)))

        BS, _, _, _ = depth_spc.shape
        depth_rgb = torch.cat([depth, rgb], dim=0)
        shared = self.feature_extraction_shared(self.lrelu(self.conv_shared(depth_rgb)))

        feat_depth = torch.cat([depth_spc, shared[:BS, ...]], dim=1)
        feat_rgb = torch.cat([rgb_spc, shared[BS:, ...]], dim=1)

        feat_depth = self.lrelu(self.conv_depth_semi(feat_depth))
        feat_rgb = self.lrelu(self.conv_rgb_semi(feat_rgb))

        #### features interaction
        if self.training:
            feat_action1, depth_FromFusion1, rgb2depth1, depth_offset1 = self.PAI_1(feat_depth, feat_rgb)
        else:
            feat_action1 = self.PAI_1(feat_depth, feat_rgb)
        feat_depth_mapping1 = self.lrelu(self.conv_depth_mapping1(torch.cat([feat_action1, feat_depth], dim=1)))
        feat_rgb_mapping1 = self.lrelu(self.conv_rgb_mapping1(torch.cat([feat_action1, feat_rgb], dim=1)))

        if self.training:
            feat_action2, depth_FromFusion2, rgb2depth2, depth_offset2 = self.PAI_2(feat_depth_mapping1, feat_rgb_mapping1)
        else:
            feat_action2 = self.PAI_2(feat_depth_mapping1, feat_rgb_mapping1)
        feat_depth_mapping2 = self.lrelu(self.conv_depth_mapping1(torch.cat([feat_action2, feat_depth_mapping1], dim=1)))
        feat_rgb_mapping2 = self.lrelu(self.conv_rgb_mapping1(torch.cat([feat_action2, feat_rgb_mapping1], dim=1)))

        if self.training:
            feat_action3, depth_FromFusion3, rgb2depth3, depth_offset3 = self.PAI_3(feat_depth_mapping2, feat_rgb_mapping2)
        else:
            feat_action3 = self.PAI_3(feat_depth_mapping2, feat_rgb_mapping2)
        feat_depth_mapping3 = self.lrelu(self.conv_depth_mapping1(torch.cat([feat_action3, feat_depth_mapping2], dim=1)))
        feat_rgb_mapping3 = self.lrelu(self.conv_rgb_mapping1(torch.cat([feat_action3, feat_rgb_mapping2], dim=1)))

        output = self.PGR(feat_depth_mapping3, feat_rgb_mapping3, feat_action3)

        if self.training:
            return output, depth_FromFusion1, depth_FromFusion2, depth_FromFusion3, rgb2depth1, depth_offset1, rgb2depth2, depth_offset2, rgb2depth3, depth_offset3
        else:
            return output

