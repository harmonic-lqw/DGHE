import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import argparse
from models.gan_direction.e4e_d.stylegan2.op import fused_leaky_relu
from models.gan_direction.e4e_d.psp import pSp
from models.gan_direction.e4e_d.encoders.psp_encoders import GradualStyleBlock
from models.gan_direction.e4e_d.stylegan2.model import EqualLinear, PixelNorm
from models.gan_direction.e4e_d.encoders.helpers import _upsample_add


def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    net = pSp(opts)
    # net.eval()
    net = net.to(device)
    return net, opts

class E4EDirection(nn.Module):
    def __init__(
            self,
            args,
            clip_dim=512,
            lr_mlp=0.01,
            image_size=256,
        ):
        super(E4EDirection, self).__init__()

        self.args = args

        log_size = int(math.log(image_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        # MOD for coarse
        ModLayers = [PixelNorm()]
        ModLayers.append(
            nn.Conv2d(512, 512, (3, 3), 1, 1)
        )
        self.mod_coarse = nn.Sequential(*ModLayers)

        coarse = []
        coarse.append(
            EqualLinear(clip_dim, 512, lr_mul=lr_mlp, activation='fused_lrelu'),
        )
        for _ in range(1, 4):
            coarse.append(
                EqualLinear(
                    512, 512, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.fc_coarse = nn.Sequential(*coarse)
        self.scale_coarse = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512, bias=True)
        )
        self.shift_coarse = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512, bias=True)
        )

        self.relu_coarse = nn.LeakyReLU(negative_slope=0.2)

        # MOD for medium
        ModLayers = [PixelNorm()]
        ModLayers.append(
            nn.Conv2d(512, 512, (3, 3), 1, 1)
        )
        self.mod_medium = nn.Sequential(*ModLayers)

        medium = []
        medium.append(
            EqualLinear(clip_dim, 512, lr_mul=lr_mlp, activation='fused_lrelu'),
        )
        for _ in range(1, 4):
            medium.append(
                EqualLinear(
                    512, 512, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.fc_medium = nn.Sequential(*medium)

        self.scale_medium = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512, bias=True)
        )
        self.shift_medium = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512, bias=True)
        )

        self.relu_medium = nn.LeakyReLU(negative_slope=0.2)


        # MOD for fine
        ModLayers = [PixelNorm()]
        ModLayers.append(
            nn.Conv2d(512, 512, (3, 3), 1, 1)
        )
        self.mod_fine = nn.Sequential(*ModLayers)

        fine = []
        fine.append(
            EqualLinear(
                clip_dim, 512, lr_mul=lr_mlp, activation='fused_lrelu'
            )
        )
        for _ in range(1, 4):
            fine.append(
                EqualLinear(
                    512, 512, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.fc_fine = nn.Sequential(*fine)

        self.scale_fine = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512, bias=True)
        )
        self.shift_fine = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512, bias=True)
        )
        self.relu_fine = nn.LeakyReLU(negative_slope=0.2)

        self.wplus2w = nn.Sequential(
            EqualLinear(
                512*self.style_count, 512*self.style_count, lr_mul=lr_mlp, activation='fused_lrelu'
            ),
            EqualLinear(
                512*self.style_count, 512, lr_mul=lr_mlp, activation='fused_lrelu'
            )
        )
        
        
        
        # fix memory
        # e4e_checkpoint_path = "/HDDdata/LQW/Auxiliary_models/pSp/pretrained_checkpoint/e4e_ffhq_encode.pt"
        e4e_checkpoint_path = self.args.gan_model_path
        self.net, self.opts = setup_model(e4e_checkpoint_path, 'cuda')
        # self.body = self.net.encoder.body
        # # if image_size = 256 -> len(m2s) = 14
        # self.m2s = self.net.encoder.styles
        # self.input_layer = self.net.encoder.input_layer
        # self.latlayer1 = self.net.encoder.latlayer1
        # self.latlayer2 = self.net.encoder.latlayer2


    def forward(self, x, text_feature):
        x = self.net.encoder.input_layer(x)
        modulelist = list(self.net.encoder.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
            # (N, 128, 64, 64)
                c1 = x
            elif i == 20:
            # (N, 256, 32, 32)
                c2 = x
            elif i == 23:
            # (N, 512, 16, 16)
                c3 = x

        # p3 = c3
        p3 = c3
        p2 = _upsample_add(p3, self.net.encoder.latlayer1(c2))
        p1 = _upsample_add(p2, self.net.encoder.latlayer2(c1))


        # get scale and shift from text_feature
        feat3 = self.fc_coarse(text_feature)
        scale3 = self.scale_coarse(feat3).view(1, 512, 1, 1)
        shift3 = self.shift_coarse(feat3).view(1, 512, 1, 1)

        feat2 = self.fc_medium(text_feature)
        scale2 = self.scale_medium(feat2).view(1, 512, 1, 1)
        shift2 = self.scale_medium(feat2).view(1, 512, 1, 1)
        
        feat1 = self.fc_fine(text_feature)
        scale1 = self.scale_fine(feat1).view(1, 512, 1, 1)
        shift1 = self.shift_fine(feat1).view(1, 512, 1, 1)

        # get moded p3/p2/p1
        p3 = self.mod_coarse(p3)
        p3 = p3 * (1 + scale3) + shift3
        p3 = self.relu_coarse(p3)

        p2 = self.mod_medium(p2)
        p2 = p2 * (1 + scale2) + shift2
        p2 = self.relu_medium(p2)

        p1 = self.mod_fine(p1)
        p1 = p1 * (1 + scale1) + shift1
        p1 = self.relu_fine(p1)

        # get w+
        w0 = self.net.encoder.styles[0](p3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        features = p3
        for i in range(1, self.style_count):
            if i == self.coarse_ind:
                features = p2
            if i == self.middle_ind:
                features = p1
            delta_i = self.net.encoder.styles[i](features) # (N, 512)
            # w[:, i].size() (N, 512)
            w[:, i] += delta_i

        # w.size() (N, style_count, 512)
        if self.args.bs_train == 1:
            w = w.view(1, -1)
        else:
            w = w.reshape(self.args.bs_train, -1)
        
        w = self.wplus2w(w)

        # return w[:, 0] # (N, 512)

        return w # (N, 512)









