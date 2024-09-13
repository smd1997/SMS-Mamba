import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
import matplotlib.pyplot as plt
import numpy as np
from models.modules.blocks import fft2c, ifft2c
from torch.nn import L1Loss
from torch.nn import MSELoss
from skimage.metrics import structural_similarity as ssim

def compute_metrics2c(out, label):
    psnr_ = 0
    ssim_ = 0
    out = out.contiguous().detach().cpu().numpy()
    label = label.contiguous().detach().cpu().numpy()
    assert label.shape == label.shape, 'tensor size inconsistency'
    B = out.shape[0]
    for i in range(B):
        x = out[i,...]
        y = label[i,...]
        x = np.squeeze(x)
        y = np.squeeze(y)
        # x = (x - x.min())/(x.max()-x.min())
        psnr_ += psnr(x, y)
        ssim_ += ssim(x, y, data_range=1.0, win_size=11)
    # ssim = ssim_torch(out.unsqueeze(1), label.unsqueeze(1), out.device)
    # plt.imshow(out[0,:,:].cpu().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/models/saved/fastMRI/X.jpg')
    # plt.imshow(label[0,:,:].cpu().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/models/saved/fastMRI/Y.jpg')
    return psnr_ / B, ssim_ / B
            
def compute_metrics2c_full(X:list, Y:list):
    PSNR_list = []
    SSIM_list = []
    for i in range(len(X)):
        x = X[i].detach().cpu().numpy()
        y = Y[i].detach().cpu().numpy()
        assert X[i].shape == Y[i].shape, 'tensor size inconsistency'
        PSNR_list.append(psnr(x, y))
        SSIM_list.append(ssim(x, y, data_range=1.0, win_size=11))
    return PSNR_list, SSIM_list

def psnr(img1:torch.Tensor, img2:torch.Tensor):
    mse = ((img1-img2)**2).mean()
    if mse == 0:
        return np.inf
    else:
        return 10*np.log10(1.0/mse)

def ssim_torch(img1:torch.Tensor, img2:torch.Tensor, device_id='cuda:0', win_size:int = 7):
    num_channel = 1
    w = (torch.ones(1, num_channel, win_size, win_size) / win_size**2).to(device_id)
    data_range = torch.tensor([1]).to(device_id)[:, None, None, None]
    NP = win_size**2
    cov_norm = NP / (NP - 1)
    k1 = 0.01
    k2 = 0.03
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(img1, w)
    uy = F.conv2d(img2, w)
    uxx = F.conv2d(img1 * img1, w)
    uyy = F.conv2d(img2 * img2, w)
    uxy = F.conv2d(img1 * img2, w)
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
    S = (A1 * A2) / D
    return S.mean()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = fastmri.SSIMLoss(win_size=11)
    def forward(
        self, out, label
    ):
        # l = 0
        # weights = np.array([1 / (2**i) for i in range(len(out))])[::-1]
        # weights = weights / weights.sum()
        # for i in range(len(out)):
        #     l = l + weights[i] * torch.norm((out[i] - label),'fro') / torch.norm(label,'fro')
            # l = l + weights[i] * self.ssim(out[i].unsqueeze(1), label.unsqueeze(1), torch.tensor([label[:].max()]).to(out[i].device))
        # plt.imshow(out[0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray);plt.savefig(r'/root/AR-Recon/i.jpg')
        # l = self.ssim(out[-1].unsqueeze(1), label.unsqueeze(1), torch.tensor([label[:].max()]).to(out[-1].device))
        l = torch.norm((out - label),'fro') / torch.norm(label,'fro')
        # l += 0.5 * torch.norm((out[1] - label),'fro') / torch.norm(label,'fro')
        return l
