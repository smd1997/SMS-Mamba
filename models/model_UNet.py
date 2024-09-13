import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple
from models.modules.blocks import *
import fastmri
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt

class UNet(nn.Module):
    def __init__(self,
                input_channels: int,
                n_stages: int,
                conv_op: Type[_ConvNd],
                kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                strides: Union[int, List[int], Tuple[int, ...]],
                n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                num_output_channels: int,
                UNet_base_num_features: int,
                UNet_max_num_features: int,
                n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                conv_bias: bool = False,
                norm_op: Union[None, Type[nn.Module]] = None,
                norm_op_kwargs: dict = None,
                dropout_op: Union[None, Type[_DropoutNd]] = None,
                dropout_op_kwargs: dict = None,
                nonlin: Union[None, Type[torch.nn.Module]] = None,
                nonlin_kwargs: dict = None,
                deep_supervision: bool = False,
                nonlin_first: bool = False,
                output: bool = False,
                DC_type: str = 'VN'
                ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        assert DC_type in ('VN', 'AM')
        self.DC_type = DC_type
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        features_per_stage = [min(UNet_base_num_features * 2 ** i,
                                UNet_max_num_features) for i in range(n_stages)]
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)

        self.contour_encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                                n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                                nonlin_first=nonlin_first)
        de_feat_per_stage = [fea*2 for fea in features_per_stage]
        self.encoder1 = PlainConvEncoder(input_channels, n_stages, de_feat_per_stage, conv_op, kernel_sizes, strides,
                                                n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                                nonlin_first=nonlin_first)
        # import pdb
        # pdb.set_trace()
        num_output_channels = input_channels
        self.decoder = UNetDecoder(self.encoder1, num_output_channels, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        if self.DC_type == 'VN':
            self.DC = DC_layer2c(soft=True)
        else:
            self.DC = DC_layer2c(soft=False)
        self.out_put = output
        self.apply(self._init_weights)

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std
    
    def unnorm(self,
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    
    def forward(self, x, contour, mask, uk0):
        # x = ifft2c(uk, dim=(-2, -1))   
        B, C, h, w, _ = x.shape
        x = x.permute(0,4,1,2,3).reshape(B, 1*C, h, w)
        skips = self.encoder(x)

        B1, C1, h1, w1, _ = contour.shape
        contour = contour.permute(0,4,1,2,3).reshape(B1, 1*C1, h1, w1)
        contour_map = self.contour_encoder(contour)

        # import pdb
        # pdb.set_trace()
        mid = []
        for i in range(5):
            concatenated_matrix = torch.cat((skips[i], contour_map[i]), dim=1)  # 在第2维拼接
            mid.append(concatenated_matrix)

        out_put = self.decoder(mid)[0]
        # import pdb
        # pdb.set_trace()
        i_rec = out_put.reshape(B, 1, C, h, w).permute(0,2,3,4,1)
        # Gk = fft2c(i_rec, dim=(-2, -1))
        # k_out = self.DC(Gk, mask, uk0)
        # plt.imshow(torch.log(torch.view_as_complex(k_out)[0,0,:,:].abs()).cpu().detach().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/krefine.jpg');plt.close()
        # if self.out_put:
        #     out = fastmri.complex_abs(ifft2c(k_out, dim=(-2, -1)).squeeze(1))
        #     plt.imshow(out[0,:,:].abs().cpu().detach().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/iout_r.jpg');plt.close()
        # else:
        #     out = k_out
        # import pdb
        # pdb.set_trace()
        return i_rec
    
    # def forward(self, uk, mask, uk0):
    #     x = ifft2c(uk, dim=(-2, -1))   
    #     B, C, h, w, _ = x.shape
    #     x = x.permute(0,4,1,2,3).reshape(B, 2*C, h, w)
    #     x, mean, std = self.norm(x)
    #     skips = self.encoder(x)
    #     out_put = self.decoder(skips)[0]
    #     out_put = self.unnorm(out_put, mean, std)
    #     i_rec = out_put.reshape(B, 2, C, h, w).permute(0,2,3,4,1)
    #     Gk = fft2c(i_rec, dim=(-2, -1))
    #     if self.DC_type == 'VN':
    #         k_out = self.DC(uk, mask, uk0) - Gk
    #     elif self.DC_type == 'AM':
    #         k_out = self.DC(uk + Gk, mask, uk0)
    #     if self.out_put:
    #         out = fastmri.complex_abs(ifft2c(k_out, dim=(-2, -1)).squeeze(1))
    #     else:
    #         out = k_out
    #     return out

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            trunc_normal_(m.weight, std=0.01)

class UNet_LS(UNet):
    def __init__(self,
                DC_type = 'VN',
                output = False,
                **kwargs):
        super().__init__(DC_type=DC_type,
                         output=output,
                         **kwargs)
        self.output = output
        self.padsize = 128
        
    @staticmethod
    def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, t, h, w = x.shape
        x = x.view(b, c, t * h * w)
        mean = x.mean(dim=2).view(b, c, 1, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1, 1)
        x = x.view(b, c, t, h, w)
        return (x - mean) / std, mean, std

    @staticmethod
    def unnorm(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    
    @staticmethod
    def sens_expand(x: torch.Tensor, sens_map: torch.Tensor) -> torch.Tensor:
        x = fastmri.complex_mul(x.unsqueeze(0), sens_map)
        x = fft2c(x, dim=(-2, -1))
        return x
    
    @staticmethod
    def sens_reduce(x: torch.Tensor, sens_map: torch.Tensor, dim=0) -> torch.Tensor:
        x = ifft2c(x, dim=(-2, -1))
        x = fastmri.complex_mul(x, fastmri.complex_conj(sens_map)).sum(dim)#(C,T,kx,ky,2)
        return x
        
    def pad(self, x: torch.Tensor):
        # x:(C,T,h,w)
        _, _, h, w = x.shape
        ph = self.padsize - h
        pw = self.padsize - w
        if ph % 2 != 0:
            ph1 = ph // 2
            ph2 = ph1 + 1
        else:
            ph1 = ph // 2
            ph2 = ph // 2
        if pw % 2 != 0:
            pw1 = pw // 2
            pw2 = pw1 + 1
        else:
            pw1 = pw // 2
            pw2 = pw // 2
        return F.pad(x, (pw2,pw1,ph2,ph1), 'constant', 0), (ph1, ph2, pw1, pw2)
    
    def unpad(self, x: torch.Tensor, p: tuple):
        # x:(T,h,w,2)
        _, h, w, _ = x.shape
        (ph1, ph2, pw1, pw2) = p
        x = x[:, ph1:h-ph2, pw1:w-pw2, :]
        return x
    
    def forward(self, uk, uk0, smap, mask):
        batch = []
        padsize = []
        # cat batch
        for i, uki in enumerate(uk):
            img = self.sens_reduce(uki, smap[i]).permute(3,0,1,2)
            img, p = self.pad(img)
            padsize.append(p)
            batch.append(img)#(2,T,h,w)
        i = torch.stack(batch, dim=0)
        i, mean, std = self.norm(i)
        skips = self.encoder(i)
        out_put = self.decoder(skips)[0]
        out_put = self.unnorm(out_put, mean, std)
        out_put = out_put.permute(0,2,3,4,1)#(B,T,h,w,2)
        out_list = []
        for i, uki in enumerate(uk):
            img = self.unpad(out_put[i], padsize[i])
            Gk = self.sens_expand(img, smap[i])
            if self.DC_type == 'VN':
                k_out = self.DC(uk, mask, uk0) - Gk
            elif self.DC_type == 'AM':
                k_out = self.DC(uk + Gk, mask, uk0)
            if self.output:
                i_out = self.sens_reduce(k_out, smap[i])
                out_list.append(i_out)
            else:
                out_list.append(k_out)
        return out_list

class UNet_unrolled_LS(nn.Module):
    def __init__(self, iter=6, DC_type='VN', kwargs=None):
        super().__init__()
        self.layers = []
        for _ in range(iter-1):
            self.layers.append(UNet_LS(**kwargs, DC_type=DC_type, output=False))
        self.layers.append(UNet_LS(**kwargs, DC_type=DC_type, output=True))
        self.layers = nn.ModuleList(self.layers)
        
    def batch_to_list(self, batch):
        l = []
        for i in range(batch.shape[0]):
            l.append(batch[i])
        return l
    
    def forward(self, list):
        uk0 = [l[0] for l in list]
        smap = [l[1] for l in list]
        kmask = [l[2] for l in list]
        x = uk0
        for i, layer in enumerate(self.layers):
            x = layer(x, uk0, smap, kmask)
        # x = self.batch_to_list(self.tail(torch.stack(x).permute(0,4,1,2,3)).permute(0,2,3,4,1))
        return x

class UNet_unrolled(nn.Module):
    def __init__(self, iter=6, DC_type='VN', kwargs=None):
        super().__init__()
        self.layers = []
        for _ in range(iter-1):
            self.layers.append(UNet(**kwargs, DC_type=DC_type, output=False))
        self.layers.append(UNet(**kwargs, DC_type=DC_type, output=True))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x, contour, mask):
        uk0 = x.clone() #uk
        x = x.unsqueeze(-1)
        contour = contour.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, contour, mask, uk0)
        return x

class DC_layer2c(nn.Module):
    def __init__(self, soft=False):
        super(DC_layer2c, self).__init__()
        self.soft = soft
        if self.soft:
            self.dc_weight = nn.Parameter(torch.ones(1))
    def forward(self, k_rec, mask, uk):
        if len(mask.shape) < len(k_rec.shape):
            mask = mask.unsqueeze(-1)
        masknot = 1 - mask
        if self.soft:
            k_out = masknot * k_rec * (1 - self.dc_weight) + uk * self.dc_weight
        else:
            k_out = masknot * k_rec + uk
        return k_out
