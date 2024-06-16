import torch
from torch import nn
from torch import fft
from torch.nn import functional as F

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    # 幅值和相位
    def forward(self, x, y):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss

    # 幅值
    # def forward(self, x, y):
    #     fft_x = fft.fft2(x)
    #     fft_y = fft.fft2(y)
    #     magnitude_x = torch.abs(fft_x)
    #     magnitude_y = torch.abs(fft_y)
    #     loss = F.mse_loss(magnitude_x, magnitude_y)
    #     return loss

    # 相位
    # def forward(self, x, y):
    #     fft_x = fft.fft2(x)
    #     fft_y = fft.fft2(y)
    #     phase_x = torch.angle(fft_x)
    #     phase_y = torch.angle(fft_y)
    #     loss = F.mse_loss(phase_x, phase_y)
    #     return loss

    

# fft_image1 = fft.fft2(image1_tensor)
# fft_image2 = fft.fft2(image2_tensor)

# # 计算实部（幅值）和虚部（相位）
# magnitude_image1 = torch.abs(fft_image1)
# magnitude_image2 = torch.abs(fft_image2)

# phase_image1 = torch.angle(fft_image1)
# phase_image2 = torch.angle(fft_image2)

# # 计算幅值差距的均方误差
# mse_magnitude = F.mse_loss(magnitude_image1, magnitude_image2)  # 风格

# # 计算相位差距的均方误差
# mse_phase = F.mse_loss(phase_image1, phase_image2)  # 结构

