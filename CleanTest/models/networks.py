# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import math
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

##########################################################################################
###############ReDegNet by xiaoming li
##########################################################################################
class SynNetEncoder(nn.Module):
    def __init__(self, size=256, channel_multiplier=1):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.log_size = int(math.log(size, 2))
        conv = [SpectralNorm(nn.Conv2d(3, channels[size], 3, 1, 1)), nn.LeakyReLU(0.2)]

        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 2, 1)), nn.LeakyReLU(0.2)]
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        convf = [SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2), SynResBlock(out_channel, out_channel)] 
        self.ecdf = nn.Sequential(*convf)

    def forward(self, inputs):
        noise = []
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
        noise.append(self.ecdf(inputs))
        return noise[::-1]
    
class SynResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        self.act = nn.LeakyReLU(0.2)
        self.skip = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))

    def forward(self, input):
        out = self.conv1(input)
        out = self.act(out)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) 
        return out

class SynNetGenerator(nn.Module):
    def __init__(self, size=256, style_dim=512, n_mlp=8, channel_multiplier=1, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super().__init__()
        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.inputself = ConstantInputSelf(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)
        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.n_latent = self.log_size * 2 - 2
    def forward(
        self,
        styles,
        noise=None,
    ):
        styles = [self.style(s) for s in styles]#
        #
        latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
        out = self.inputself(noise[0])
        out = self.conv1(out, latent[:, 0], noise=noise[1])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        noise_i = 3
        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise[(noise_i + 1)//2]) #
            out = conv2(out, latent[:, i + 1], noise=None) #
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
            noise_i += 2
        image = skip

        return image

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class ConstantInputSelf(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        conv = [SpectralNorm(nn.Conv2d(channel, channel, 3, 1, 1)), nn.LeakyReLU(0.2)]
        self.conv = nn.Sequential(*conv)
    def forward(self, input):
        out = self.conv(input)
        return out


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True,):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.conv2 = nn.Sequential(*[SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2)])

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.conv2(out)
        return out


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1], ):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.modulation = nn.Linear(style_dim, in_channel)
        self.demodulate = demodulate


    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            out = self.up(input)
            out = F.conv2d(out, weight, padding=1, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is not None:
            return image + self.weight * noise
        if noise is None:
            return image

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.upsample = upsample

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(
                    skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return torch.tanh(out)

class DegNet(nn.Module):
    def __init__( self, size=256, style_dim = 512, channel_multiplier=1, ):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.log_size = int(math.log(size, 2))
        conv = [SpectralNorm(nn.Conv2d(6, channels[size], 3, 1, 1)), nn.LeakyReLU(0.2)]
        self.deg_ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]
        self.names = ['deg_ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 2, 1)), nn.LeakyReLU(0.2)]
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(*[nn.Linear(channels[4] * 4 * 4, style_dim), nn.LeakyReLU(0.2), nn.Linear(style_dim, style_dim)])

    def forward(self, inputs):
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
        inputs = inputs.view(inputs.shape[0], -1)
        outs = self.final_linear(inputs)
        return outs 


if __name__ == '__main__':
    print('Test')
