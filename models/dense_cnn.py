"""
This file contains the Definition of DPNet and UVNet
"""
from __future__ import division

import torch
import torch.nn as nn
import numpy as np

from .resnet import resnet50backbone
from .layers import ConvBottleNeck, HgNet
from .uv_generator import Index_UV_Generator
import os
from utils.objfile import read_obj


# Warp elements in image space to UV space.
def warp_feature(dp_out, feature_map, uv_res):
    """
    C: channel number of the input feature map;  H: height;  W: width

    :param dp_out: IUV image in shape (batch_size, 3, H, W)
    :param feature_map: Local feature map in shape (batch_size, C, H, W)
    :param uv_res: The resolution of the transferred feature map in UV space.

    :return: warped_feature: Feature map in UV space with shape (batch_size, C+3, uv_res, uv_res)
    The x, y cordinates in the image sapce and mask will be added as the last 3 channels
     of the warped feature, so the channel number of warped feature is C+3.
    """

    assert dp_out.shape[0] == feature_map.shape[0]
    assert dp_out.shape[2] == feature_map.shape[2]
    assert dp_out.shape[3] == feature_map.shape[3]

    dp_mask = dp_out[:, 0].unsqueeze(1)     # I channel, confidence of being foreground
    dp_uv = dp_out[:, 1:]                   # UV channels, UV coordinates
    thre = 0.5                              # The threshold of foreground and background.
    B, C, H, W = feature_map.shape
    device = feature_map.device

    # Get the sampling index of every pixel in batch_size dimension.
    index_batch = torch.arange(0, B, device=device, dtype=torch.long)[:, None, None].expand([-1, H, W])
    index_batch = index_batch.contiguous().view(-1).long()

    # Get the sampling index of every pixel in H and W dimension.
    tmp_x = torch.arange(0, W, device=device, dtype=torch.long)
    tmp_y = torch.arange(0, H, device=device, dtype=torch.long)

    y, x = torch.meshgrid(tmp_y, tmp_x)
    y = y.contiguous().view(-1).repeat([B])
    x = x.contiguous().view(-1).repeat([B])

    # Sample the confidence of every pixel,
    # and only preserve the pixels belong to foreground.
    conf = dp_mask[index_batch, 0, y, x].contiguous()
    valid = conf > thre
    index_batch = index_batch[valid]
    x = x[valid]
    y = y[valid]

    # Sample the uv coordinates of foreground pixels
    uv = dp_uv[index_batch, :, y, x].contiguous()
    num_pixel = uv.shape[0]
    # Get the corresponding location in UV space
    uv = uv * (uv_res - 1)
    uv_round = uv.round().long().clamp(min=0, max=uv_res - 1)

    # We first process the transferred feature in shape (batch_size * H * W, C+3),
    # so we need to get the location of each pixel in the two-dimension feature vector.
    index_uv = (uv_round[:, 1] * uv_res + uv_round[:, 0]).detach() + index_batch * uv_res * uv_res

    # Sample the feature of foreground pixels
    sampled_feature = feature_map[index_batch, :, y, x]
    # Scale x,y coordinates to [-1, 1] and
    # concatenated to the end of sampled feature as extra channels.
    y = (2 * y.float() / (H - 1)) - 1
    x = (2 * x.float() / (W - 1)) - 1
    sampled_feature = torch.cat([sampled_feature, x[:, None], y[:, None]], dim=-1)

    # Multiple pixels in image space may be transferred to the same location in the UV space.
    # warped_w is used to record the number of the pixels transferred to every location.
    warped_w = sampled_feature.new_zeros([B * uv_res * uv_res, 1])
    warped_w.index_add_(0, index_uv, sampled_feature.new_ones([num_pixel, 1]))

    # Transfer the sampled feature to UV space.
    # Feature vectors transferred to the sample location will be accumulated.
    warped_feature = sampled_feature.new_zeros([B * uv_res * uv_res, C + 2])
    warped_feature.index_add_(0, index_uv, sampled_feature)

    # Normalize the accumulated feature with the pixel number.
    warped_feature = warped_feature / (warped_w + 1e-8)
    # Concatenate the mask channel at the end.
    warped_feature = torch.cat([warped_feature, (warped_w > 0).float()], dim=-1)
    # Reshape the shape to (batch_size, C+3, uv_res, uv_res)
    warped_feature = warped_feature.reshape(B, uv_res, uv_res, C + 3).permute(0, 3, 1, 2)

    return warped_feature


# DPNet returns densepose result
class DPNet(nn.Module):

    def __init__(self, warp_lv=2, norm_type='BN'):
        super(DPNet, self).__init__()
        nl_layer = nn.ReLU(inplace=True)
        self.warp_lv = warp_lv
        # image encoder
        self.resnet = resnet50backbone(pretrained=True)
        # dense pose line
        dp_layers = []
        #              [224, 112, 56, 28,    14,    7]
        channel_list = [3,   64, 256, 512, 1024, 2048]
        for i in range(warp_lv, 5):
            in_channels = channel_list[i + 1]
            out_channels = channel_list[i]

            dp_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    ConvBottleNeck(in_channels=in_channels, out_channels=out_channels, nl_layer=nl_layer, norm_type=norm_type)
                )
            )

        self.dp_layers = nn.ModuleList(dp_layers)
        self.dp_uv_end = nn.Sequential(ConvBottleNeck(channel_list[warp_lv], 32, nl_layer,  norm_type=norm_type),
                                       nn.Conv2d(32, 2, kernel_size=1),
                                       nn.Sigmoid())

        self.dp_mask_end = nn.Sequential(ConvBottleNeck(channel_list[warp_lv], 32, nl_layer, norm_type=norm_type),
                                         nn.Conv2d(32, 1, kernel_size=1),
                                         nn.Sigmoid())

    def forward(self, image, UV=None):
        codes, features = self.resnet(image)
        # output densepose results
        dp_feature = features[-1]
        for i in range(len(self.dp_layers) - 1, -1, -1):
            dp_feature = self.dp_layers[i](dp_feature)
            dp_feature = dp_feature + features[i - 1 + len(features) - len(self.dp_layers)]
        dp_uv = self.dp_uv_end(dp_feature)
        dp_mask = self.dp_mask_end(dp_feature)
        dp_out = torch.cat((dp_mask, dp_uv), dim=1)
        return dp_out, dp_feature, codes


def get_LNet(options):
    if options.model == 'DecoMR':
        uv_net = UVNet(uv_channels=options.uv_channels,
                       uv_res=options.uv_res,
                       warp_lv=options.warp_level,
                       uv_type=options.uv_type,
                       norm_type=options.norm_type)
    return uv_net


# UVNet returns location map
class UVNet(nn.Module):
    def __init__(self, uv_channels=64, uv_res=128, warp_lv=2, uv_type='SMPL', norm_type='BN'):
        super(UVNet, self).__init__()

        nl_layer = nn.ReLU(inplace=True)

        self.fc_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nl_layer,
            nn.Linear(512, 256),
        )

        self.camera = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nl_layer,
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 3)
        )

        self.warp_lv = warp_lv
        channel_list = [3, 64, 256, 512, 1024, 2048]
        warp_channel = channel_list[warp_lv]
        self.uv_res = uv_res
        self.warp_res = int(256 // (2 ** self.warp_lv))

        if uv_type == 'SMPL':
            ref_file = 'data/SMPL_ref_map_{0}.npy'.format(self.warp_res)
        elif uv_type == 'BF':
            ref_file = 'data/BF_ref_map_{0}.npy'.format(self.warp_res)

        if not os.path.exists(ref_file):
            sampler = Index_UV_Generator(UV_height=self.warp_res, uv_type=uv_type)
            ref_vert, _ = read_obj('data/reference_mesh.obj')
            ref_map = sampler.get_UV_map(torch.FloatTensor(ref_vert))
            np.save(ref_file, ref_map.cpu().numpy())
        self.ref_map = torch.FloatTensor(np.load(ref_file)).permute(0, 3, 1, 2)

        self.uv_conv1 = nn.Sequential(
            nn.Conv2d(256 + warp_channel + 3 + 3, 2 * warp_channel, kernel_size=1),
            nl_layer,
            nn.Conv2d(2 * warp_channel, 2 * warp_channel, kernel_size=1),
            nl_layer,
            nn.Conv2d(2 * warp_channel, warp_channel, kernel_size=1))

        uv_lv = 0 if uv_res == 256 else 1
        self.hg = HgNet(in_channels=warp_channel, level=5 - warp_lv, nl_layer=nl_layer, norm_type=norm_type)

        cur = min(8, 2 ** (warp_lv - uv_lv))
        prev = cur
        self.uv_conv2 = ConvBottleNeck(warp_channel, uv_channels * cur, nl_layer, norm_type=norm_type)

        layers = []
        for lv in range(warp_lv, uv_lv, -1):
            cur = min(prev, 2 ** (lv - uv_lv - 1))
            layers.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                              ConvBottleNeck(uv_channels * prev, uv_channels * cur, nl_layer, norm_type=norm_type))
            )
            prev = cur
        self.decoder = nn.Sequential(*layers)
        self.uv_end = nn.Sequential(ConvBottleNeck(uv_channels, 32, nl_layer, norm_type=norm_type),
                                    nn.Conv2d(32, 3, kernel_size=1))

    def forward(self, dp_out, dp_feature, codes):
        n_batch = dp_out.shape[0]
        local_feature = warp_feature(dp_out, dp_feature, self.warp_res)
        global_feature = self.fc_head(codes)
        global_feature = global_feature[:, :, None, None].expand(-1, -1, self.warp_res, self.warp_res)
        self.ref_map = self.ref_map.to(local_feature.device).type(local_feature.dtype)
        uv_map = torch.cat([local_feature, global_feature, self.ref_map.expand(n_batch, -1, -1, -1)], dim=1)
        uv_map = self.uv_conv1(uv_map)
        uv_map = self.hg(uv_map)
        uv_map = self.uv_conv2(uv_map)
        uv_map = self.decoder(uv_map)
        uv_map = self.uv_end(uv_map).permute(0, 2, 3, 1)

        cam = self.camera(codes)
        return uv_map, cam



