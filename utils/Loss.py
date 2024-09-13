import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
from torch.nn import L1Loss
from torch.nn import MSELoss
from skimage.metrics import structural_similarity as ssim
from utils.utils import ifft, fft, compute_gaussian
# from pytorch_ssim import ssim as ssim_pytorch
# from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import math

def compute_metrics2c(X:torch.Tensor, Y:torch.Tensor, device_id):
    B, T, C, h, w, _ = X.size()
    X = torch.view_as_complex(X).abs()
    Y = torch.view_as_complex(Y).abs()
    # X = torch.cat([X[...,0], X[...,1]], dim=2)
    # Y = torch.cat([Y[...,0], Y[...,1]], dim=2)
    X = X.reshape(B, T*C, h, w)
    Y = Y.reshape(B, T*C, h, w)
    return psnr_torch(X.detach(), Y.detach()), \
            ssim_torch(X.detach(), Y.detach(), device_id)        

def compute_metrics(X:torch.Tensor, Y:torch.Tensor, device_id):
    B, T, C, h, w = X.size()
    X = X.reshape(B, T*C, h, w)
    Y = Y.reshape(B, T*C, h, w)
    return psnr_torch(X.detach(), Y.detach()), \
            ssim_torch(X.detach(), Y.detach(), device_id)
            # ssim(X_abs.cpu().detach().numpy(), Y_abs.cpu().detach().numpy(), data_range=1.0, win_size=7, channel_axis=0)
            # skimage's ssim is computed along H,W while average along channel dim
            
def compute_metrics_full(X:list, Y:list):
    PSNR_list = []
    SSIM_list = []
    for i in range(len(X)):
        S, T, C, h, w = X[i].shape
        assert X[i].shape == Y[i].shape, 'tensor size inconsistency'
        X[i] = X[i].reshape(S, T*C, h, w)
        Y[i] = Y[i].reshape(S, T*C, h, w)
        PSNR_list.append(psnr_torch(X[i], Y[i]))
        SSIM_list.append(ssim_torch(X[i], Y[i], 'cpu'))
    return PSNR_list, SSIM_list

# def psnr_np(img1, img2):
#     img1 = (img1 - img1.min())/(img1.max()-img1.min())
#     img2 = (img2 - img2.min())/(img2.max()-img2.min())
#     mse = np.mean((img1-img2)**2)
#     if mse == 0:
#         return 100
#     psnr = 20 * math.log10(1.0/math.sqrt(mse))
#     return psnr

def psnr_torch(img1:torch.Tensor, img2:torch.Tensor):
    img1 = (img1 - img1.min())/(img1.max()-img1.min())
    img2 = (img2 - img2.min())/(img2.max()-img2.min())
    mse = ((img1-img2)**2).mean()
    if mse == 0:
        return torch.inf
    else:
        return 10*torch.log10(1.0/mse)

def ssim_torch(img1:torch.Tensor, img2:torch.Tensor, device_id='cuda:0', win_size:int = 7):
    num_channel = 1#img1.shape[1]
    img1 = (img1 - img1.min())/(img1.max()-img1.min())
    img2 = (img2 - img2.min())/(img2.max()-img2.min())
    w = (torch.ones(1, num_channel, win_size, win_size) / win_size**2).to(device_id)
    # w1 = (torch.ones(1, img1.shape[1], win_size, win_size) / win_size**2).to(device_id)
    data_range = torch.tensor([1]).to(device_id)[:, None, None, None]
    NP = win_size**2
    cov_norm = NP / (NP - 1)
    k1 = 0.01
    k2 = 0.03
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    S = 0
    for i in range(img1.shape[1]):
        i1 = img1[:,i,:,:].unsqueeze(1)
        i2 = img2[:,i,:,:].unsqueeze(1)
        ux = F.conv2d(i1, w)
        uy = F.conv2d(i2, w)
        uxx = F.conv2d(i1 * i1, w)
        uyy = F.conv2d(i2 * i2, w)
        uxy = F.conv2d(i1 * i2, w)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S += (A1 * A2) / D
    S = S / img1.shape[1]
    return S.mean()
    

class Loss(nn.Module):
    """
    SSIM loss part is from fastmri module
    """
    def __init__(self, num_channel=1, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, DC = False, device='cuda'):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.DC = DC
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, num_channel, win_size, win_size) / win_size**2)# (channel_out, channel_in, win_size, win_size)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1) 
        self.L1 = L1Loss(size_average=True)
        self.MSE = MSELoss(size_average=True)
        self.device = device
    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        kmask: torch.Tensor,
        data_range: torch.Tensor
    ):
        assert isinstance(self.w, torch.Tensor)
        B, T, C, h, w = X.size()
        # L1
        l1 = self.L1(X, Y)
        # MSE(L2)
        l2 = self.MSE(X, Y)
        # SSIM
        X = X.reshape(B, T*C, h, w)
        Y = Y.reshape(B, T*C, h, w)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        S = 0
        for i in range(X.shape[1]):
            i1 = X[:,i,:,:].unsqueeze(1)
            i2 = Y[:,i,:,:].unsqueeze(1)
            ux = F.conv2d(i1, self.w)
            uy = F.conv2d(i2, self.w)
            uxx = F.conv2d(i1 * i1, self.w)
            uyy = F.conv2d(i2 * i2, self.w)
            uxy = F.conv2d(i1 * i2, self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux**2 + uy**2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            S += (A1 * A2) / D
        S = S / X.shape[1]
        # kspace Loss
        if self.DC:
            gaussian_mask = (1 - compute_gaussian(tile_size=(h,w), device=self.device).unsqueeze(0).unsqueeze(-1))
            ldc = 0
            for i in range(X.shape[1]):
                xi = X[:,i,:,:]
                yi = Y[:,i,:,:]
                xk = torch.view_as_real(fft(xi, shift=True, dim=(1,2)))
                yk = torch.view_as_real(fft(yi, shift=True, dim=(1,2)))
                ldc += self.MSE(xk*gaussian_mask, yk*gaussian_mask)
            ldc = ldc / X.shape[1]
            return 1 - S.mean() + l1 + l2 + ldc
        else:
            return 1 - S.mean() + l1 + l2
