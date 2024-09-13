import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.nn import MSELoss
import fastmri

def fft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data

def ifft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data

def compute_metrics2c(outlist, labellist, device_id):
    psnr = 0
    ssim = 0
    # for i in range(len(outlist)):
    x = outlist[0].contiguous().detach()
    y = labellist[0].contiguous().detach()
    C, T, h, w = x.shape
    assert x.shape == y.shape, 'tensor size inconsistency'
    # x = torch.view_as_complex(x).abs()
    # y = torch.view_as_complex(y).abs()
    x = x.reshape(C*T, 1, h, w)
    y = y.reshape(C*T, 1, h, w)
    psnr = psnr_torch(x, y)
    ssim = ssim_torch(x, y, device_id)
    return psnr, ssim
            
def compute_metrics2c_full(X:list, Y:list):
    PSNR_list = []
    SSIM_list = []
    for i in range(len(X)):
        S, T, h, w, _ = X[i].shape
        assert X[i].shape == Y[i].shape, 'tensor size inconsistency'
        x = torch.view_as_complex(X[i]).abs().reshape(S*T, 1, h, w)
        y = torch.view_as_complex(Y[i]).abs().reshape(S*T, 1, h, w)
        PSNR_list.append(psnr_torch(x, y))
        SSIM_list.append(ssim_torch(x, y, x.device))
    return PSNR_list, SSIM_list

def psnr_torch(img1:torch.Tensor, img2:torch.Tensor):
    img1 = (img1 - img1.min())/(img1.max()-img1.min())
    img2 = (img2 - img2.min())/(img2.max()-img2.min())
    mse = ((img1-img2)**2).mean()
    if mse == 0:
        return torch.inf
    else:
        return 10*torch.log10(1.0/mse)

def ssim_torch(img1:torch.Tensor, img2:torch.Tensor, device_id='cuda:0', win_size:int = 7):
    num_channel = 1
    img1 = (img1 - img1.min())/(img1.max()-img1.min())
    img2 = (img2 - img2.min())/(img2.max()-img2.min())
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
    """
    SSIM loss part is from fastmri module
    """
    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.L1 = L1Loss(size_average=True)
        self.MSE = MSELoss(size_average=True)
        self.ssim = fastmri.SSIMLoss()
    def forward(
        self,
        out: list,
        label: list
    ):
        SSIM_loss = 0
        Xi = out[0]
        Yi = label[0]
        # l1 = 0
        # l2 = 0
        # for Xi, Yi in zip(out, label):
        C, T, h, w= Xi.size()
        # L1
        # l1 += self.L1(Xi, Yi)
        # MSE(L2)
        l2 = self.MSE(Xi, Yi)
        # SSIM
        SSIM_loss = self.ssim(Xi.reshape(C*T, 1, h, w), Yi.reshape(C*T, 1, h, w), Yi.max().unsqueeze(0))
        
        return SSIM_loss + l2
    