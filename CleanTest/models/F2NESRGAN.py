# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

##########################################################################################
###############F2N-ESRGAN by xiaoming li
##########################################################################################

class F2NESRGAN(object):
    def __init__(self, CheckPointPath='../checkpoints/F2NESRGAN.pth', device='cuda'):
        self.device = device
        self.modelBG = RRDBNet()
        Num_Parameter = print_networks(self.modelBG)
        print('[F2NESRGAN] Total Number of Parameters : {:.2f} M'.format(Num_Parameter))
        self.modelBG.load_state_dict(torch.load(CheckPointPath)['params'], strict=True) #
        self.modelBG.eval()
        for k, v in self.modelBG.named_parameters():
            v.requires_grad = False
        self.modelBG = self.modelBG.to(self.device)
        torch.cuda.empty_cache()


    def handle_restoration(self, bg, tile_size=512):
        sf=4
        tile_overlap = 32
        window_size = 8
        height, width = bg.shape[:2]
        with torch.no_grad():
            LQ = transforms.ToTensor()(bg)
            LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)
            LQ = LQ.unsqueeze(0)
            LQ = LQ.to(self.device)

            _, _, h_old, w_old = LQ.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            LQ = torch.cat([LQ, torch.flip(LQ, [2])], 2)[:, :, :h_old + h_pad, :]
            LQ = torch.cat([LQ, torch.flip(LQ, [3])], 3)[:, :, :, :w_old + w_pad]
            if tile_size is None:
                SQ = self.modelBG(LQ)
            else:
                b, c, h, w = LQ.size()
                tile = min(tile_size, h, w)
                assert tile % window_size == 0, "tile size should be a multiple of window_size"

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E = torch.zeros(b, c, h*sf, w*sf).type_as(LQ)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = LQ[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = self.modelBG(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)
                        E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                        W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
                SQ = E.div_(W)

            SQ = SQ[..., :h_old * sf, :w_old * sf]
            SQ = SQ * 0.5 + 0.5
            SQ = SQ.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
            SQ = np.clip(SQ.float().cpu().numpy(), 0, 1) * 255.0
        return SQ[:,:,::-1]

def print_networks(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params / 1e6


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


if __name__ == '__main__':
    print('Test')
