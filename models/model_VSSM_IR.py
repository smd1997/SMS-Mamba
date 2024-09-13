import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from typing import Union, Type, List, Tuple
import sys
sys.path.append(__file__.rsplit('/', 2)[0])
from models.modules.vmamba import *
from models.modules.blocks import *
from timm.models.layers import trunc_normal_
import fastmri
import matplotlib.pyplot as plt

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x.permute(0,3,1,2)).permute(0,2,3,1)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x.permute(0,3,1,2)).permute(0,2,3,1)

class VSSM_Decoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 ouput_channels: int,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision: bool = False
                 ):

        super().__init__()
        stages = []
        transpconvs = []
        conv_op = encoder.conv_op
        n_stages_encoder = len(encoder.output_channels)
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            transpconvs.append(
                Upsample(n_feat=input_features_below)
            )
            stage_modules = []
            if s != n_stages_encoder - 1:
                stage_modules.append(
                    conv_op(
                    in_channels=input_features_skip*2,
                    out_channels=input_features_skip,
                    kernel_size=1,
                    bias=False)
                )
                for _ in range(n_conv_per_stage[s-1]):
                    stage_modules.append(
                    VSSBlock(
                        hidden_dim=input_features_skip,
                        forward_type="v2_noz"
                    )
                    )
            else:
                for _ in range(4):
                    stage_modules.append(
                    VSSBlock(
                        hidden_dim=input_features_skip*2,
                        forward_type="v2_noz"
                    )
                    )
                stage_modules.append(
                    nn.Sequential(
                    conv_op(
                    in_channels=input_features_skip*2,
                    out_channels=ouput_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                    Permute(0,3,1,2))
                )
            stages.append(nn.Sequential(*stage_modules))
            
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.deep_supervision = deep_supervision
    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), -1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(x)
            elif s == (len(self.stages) - 1):
                seg_outputs.append(x)
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

class VSSM_Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 patch_size:int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 padding: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 return_skips: bool = False
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(padding, int):
            padding = [padding] * n_stages
            
        stages = []
        self.return_skips = return_skips
        self.output_channels = features_per_stage
        self.strides = strides
        self.padding = padding
        if conv_op == nn.Conv2d:
            conv_op = Conv2d_channel_last
        elif conv_op == nn.Conv3d:
            conv_op = Conv3d_channel_last
        
        self.conv_op = conv_op
        if conv_op == nn.Conv2d or conv_op == Conv2d_channel_last:
            stages.append(nn.Sequential(
                nn.Conv2d(input_channels, features_per_stage[0], kernel_size=3, stride=1, padding=1, bias=False),
                Permute(0,2,3,1)
                ))
        else:
            stages.append(nn.Sequential(
                nn.Conv3d(input_channels, features_per_stage[0], kernel_size=3, stride=1, padding=1, bias=False),
                Permute(0,2,3,1)
                ))
        for s in range(n_stages):
            stage_modules = []
            if s > 0:
                stage_modules.append(Downsample(n_feat=input_channels))
            for _ in range(n_conv_per_stage[s]):
                stage_modules.append(
                VSSBlock(
                    hidden_dim=features_per_stage[s],
                    forward_type="v2_noz"
                )
                )
            input_channels = features_per_stage[s]
            stages.append(nn.Sequential(*stage_modules))
        self.stages = nn.ModuleList(stages)
    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

class VSSMUNet_IR(nn.Module):
    def __init__(self,
                input_channels: int,
                patch_size: int,
                d_model: int,
                n_stages: int,
                conv_op: Type[_ConvNd],
                kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                strides: Union[int, List[int], Tuple[int, ...]],
                padding: Union[int, List[int], Tuple[int, ...]],
                n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                num_output_channels: int,
                UNet_base_num_features: int,
                UNet_max_num_features: int,
                n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                deep_supervision: bool = True,
                out_put: bool = False,
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
        features_per_stage = [min(UNet_base_num_features * 2 ** i,
                                UNet_max_num_features) for i in range(n_stages)]
        self.input_channels = input_channels
        num_output_channels = input_channels
        self.d_model = d_model
        if self.DC_type == 'VN':
            self.DC = DC_layer2c(soft=True)
        else:
            self.DC = DC_layer2c(soft=False)
        self.out_put = out_put
        self.encoder = VSSM_Encoder(input_channels=input_channels,
                                    patch_size=patch_size,
                                    n_stages=n_stages, 
                                    n_conv_per_stage=n_conv_per_stage,
                                    features_per_stage=features_per_stage, 
                                    conv_op=conv_op,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    padding=padding,
                                    return_skips=True)
        self.decoder = VSSM_Decoder(self.encoder, 
                                    num_output_channels, 
                                    n_conv_per_stage_decoder, 
                                    deep_supervision=False)
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
    
    def forward(self, uk, mask, uk0):
        x = ifft2c(uk, dim=(-2, -1))   
        B, C, h, w, _ = x.shape
        x = x.permute(0,4,1,2,3).reshape(B, 2*C, h, w)
        x, mean, std = self.norm(x)
        res = x
        skips = self.encoder(x)
        out_put = self.decoder(skips)
        out_put = out_put + res
        out_put = self.unnorm(out_put, mean, std)
        i_rec = out_put.reshape(B, 2, C, h, w).permute(0,2,3,4,1)
        Gk = fft2c(i_rec, dim=(-2, -1))
        # if self.DC_type == 'VN':
        #     k_out = self.DC(uk, mask, uk0) - Gk
        # elif self.DC_type == 'AM':
        #     k_out = self.DC(uk + Gk, mask, uk0)
        k_out = self.DC(Gk, mask, uk0)
        if self.out_put:
            out = fastmri.complex_abs(ifft2c(k_out, dim=(-2, -1)).squeeze(1))
            # plt.imshow(out[0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
            # plt.savefig(r'/mnt/data/zlt/AR-Recon/models/saved/fastMRI/X.jpg')
        else:
            out = k_out
        return out
        
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

class VSSMUNetIR_unrolled(nn.Module):
    def __init__(self, iter=8, DC_type='VN', kwargs=None):
        super().__init__()
        self.layers = []
        for _ in range(iter-1):
            self.layers.append(VSSMUNet_IR(**kwargs, DC_type=DC_type))
        self.layers.append(VSSMUNet_IR(**kwargs, DC_type=DC_type, out_put=True))
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x, mask):
        uk0 = x.clone() #uk
        for layer in self.layers:
            x = layer(x, mask, uk0)
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

