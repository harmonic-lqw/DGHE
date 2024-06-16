import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from .psp import pSp
import math

class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )
    
class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

# class Adapter(nn.Module):
#     def _init__(self):
#         super(Adapter, self).__init__()

#         self.adapter_scale = nn.Sequential(
#             EqualConv2d(512, 256, 3, stride=1, padding=1, bias=True),
#             ScaledLeakyReLU(0.2),
#             EqualConv2d(256, 256, 3, stride=1, padding=1, bias=True))

#         self.adapter_shift = nn.Sequential(
#             EqualConv2d(512, 256, 3, stride=1, padding=1, bias=True),
#             ScaledLeakyReLU(0.2),
#             EqualConv2d(256, 256, 3, stride=1, padding=1, bias=True))
    
#     def forward(self, conditions):
#         adapter_conditions = [self.adapter_scale(conditions[0]), self.adapter_shift(conditions[1])]
#         return adapter_conditions
    
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_scale = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))

        self.adapter_shift = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, conditions):
        out = []
        out.append(self.adapter_scale(conditions[0]))
        out.append(self.adapter_shift(conditions[1]))
        # for i in range(len(conditions)):
        #     conditions[i] = (conditions[i][:, ::2] + conditions[i][:, 1::2]) / 2.0
        # out[0] += conditions[0]
        # out[1] += conditions[1]
        return out

def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    # is_cars = 'car' in opts['dataset_type']
    # is_faces = 'ffhq' in opts['dataset_type']
    # if is_faces:
    opts['stylegan_size'] = 1024
    # elif is_cars:
    #     opts['stylegan_size'] = 512
    # else:
    #     opts['stylegan_size'] = 256

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts['is_train'] = False
    opts = argparse.Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net = net.to(device)
    return net, opts

class HyperHFGI(nn.Module):
    def __init__(self):
        super().__init__()
        hfgi, self.opts = setup_model("/HDDdata/LQW/Diffusion/utils_model/HFGI-main/checkpoint/ckpt.pt", "cuda")

        self.ada = hfgi.grid_align
        self.ada.eval()
        self.ec = hfgi.residue
        self.ec.eval()
    
    def forward(self, edit_image, rec_image, src_image):
        res = src_image - rec_image
        res_aligned = self.ada(torch.cat((res, edit_image), 1))
        conditions = self.ec(res_aligned)
        return conditions

