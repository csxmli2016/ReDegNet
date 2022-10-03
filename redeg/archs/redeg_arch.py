from email.policy import strict
import math
import random
import torch

from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from functools import reduce
import torch.nn.utils.spectral_norm as SpectralNorm
#
import numpy as np
#------------------------------------------------------------------------------
#   BaseModel
#------------------------------------------------------------------------------
class BaseModel(nn.Module):
	def __init__(self):
		super(BaseModel, self).__init__()

	def init_weights(self):
		print("[%s] Initialize weights..." % (self.__class__.__name__))
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def load_pretrained_model(self, pretrained):
		if isinstance(pretrained, str):
			print("[%s] Load pretrained model from %s" % (self.__class__.__name__, pretrained))
			pretrain_dict = torch.load(pretrained, map_location='cpu')
			if 'state_dict' in pretrain_dict:
				pretrain_dict = pretrain_dict['state_dict']
		elif isinstance(pretrained, dict):
			print("[%s] Load pretrained model" % (self.__class__.__name__))
			pretrain_dict = pretrained

		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				if state_dict[k].shape==v.shape:
					model_dict[k] = v
				else:
					print("[%s]"%(self.__class__.__name__), k, "is ignored due to not matching shape")
			else:
				print("[%s]"%(self.__class__.__name__), k, "is ignored due to not matching key")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)


#------------------------------------------------------------------------------
#   Class of ReDegNet
#------------------------------------------------------------------------------
@ARCH_REGISTRY.register()
class SynNetEncoder(nn.Module):
    def __init__(
        self,
        size,
        channel_multiplier=1,
    ):
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

@ARCH_REGISTRY.register()
class SynNetGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=1,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
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
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
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
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
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



def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


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



@ARCH_REGISTRY.register()
class DegNet(nn.Module):
    def __init__(
        self,
        size,
        style_dim = 512,
        channel_multiplier=1,
    ):
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
        #
        self.final_linear = nn.Sequential(*[nn.Linear(channels[4] * 4 * 4, style_dim), nn.LeakyReLU(0.2), nn.Linear(style_dim, style_dim)])
    def forward(self, inputs):
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
        inputs = inputs.view(inputs.shape[0], -1)
        outs = self.final_linear(inputs)
        return outs 

####################################################################
###############Discriminator
####################################################################
@ARCH_REGISTRY.register()
class ReDegSNDiscriminator(nn.Module):
    def __init__(self, scale = 256):
        super(ReDegSNDiscriminator, self).__init__()
        self.netD  = NScaleSNDiscriminator(Scale = scale)

    def forward(self,input, gt, z, feature_match=False):
        output, _ = self.netD(input, gt, z, feature_match) 
        return output

class ReDegMultiScaleSNDiscriminator(nn.Module):
    def __init__(self, Scales = [512,256,128]):
        super(ReDegMultiScaleSNDiscriminator, self).__init__()
        self.D_pools = nn.ModuleList()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        for scale in Scales:
            netD  = NScaleSNDiscriminator(Scale = scale)
            self.D_pools.append(netD)
    def forward(self,input, gt, z, feature_match=False):
        results = []
        for netD in self.D_pools:
            output = netD(input, gt, z, feature_match) 
            results.append(output)
            # Downsample input
            input = self.downsample(input)
            gt = self.downsample(gt)
        return results

class NScaleSNDiscriminator(nn.Module):
    def __init__(self, dim = 64, Scale = 512):
        super(NScaleSNDiscriminator, self).__init__()
        self.model = []
        self.model.append(SNFirstResBlockDiscriminator(6, dim * 1))
        cur_dim = dim
        BlockNum = 5 - int(math.log(512//Scale,2))
        for i in range(BlockNum):
            self.model.append(SNResBlockDiscriminator(cur_dim, min(cur_dim*2,512), stride=2))
            cur_dim = cur_dim * 2
            cur_dim = min(cur_dim,512)

        self.model.append(nn.AvgPool2d(8))
        self.models = nn.Sequential(*self.model)
        self.fc = nn.Linear(cur_dim*2, cur_dim)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)
        self.fc2 = SpectralNorm(nn.Linear(cur_dim, 1))
    def forward(self, x, y, z, feature_match=False):
        x = torch.cat([x,y], dim=1)
        return_features = []
        for idx, m in enumerate(self.models):
            x = m(x) 
        x = torch.cat([x.view(-1,512),z], dim=1)
        x = self.fc(x)
        x = self.fc2(x)
        return x, return_features

class SNFirstResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SNFirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
        #
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.LeakyReLU(0.2),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )
    def forward(self, x):
        return self.model(x) + self.bypass(x)

class SNResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SNResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
    def forward(self, x):
        return self.model(x) + self.bypass(x)