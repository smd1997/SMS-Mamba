import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from typing import Union, Type, List, Tuple
import sys
sys.path.append(__file__.rsplit('/', 2)[0])
from models.modules.vmamba import *
from models.modules.blocks import *
from models.model_UNet import UNet
from timm.models.layers import trunc_normal_
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from utils.utils import kwargs_unet as unet_kwargs
from utils.utils import kwargs_plainunet as plainunet_kwargs
import torch.distributed as dist
import matplotlib.pyplot as plt
import fastmri

def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
    # if channel first, then Norm and Output are both channel_first
    stride = patch_size // 2
    kernel_size = stride + 1
    padding = 1
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
        (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
        (norm_layer(embed_dim) if patch_norm else nn.Identity()),
    )

def _make_patch_embed_v23D(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
    # if channel first, then Norm and Output are both channel_first
    stride = patch_size // 2
    kernel_size = stride + 1
    padding = 1
    return nn.Sequential(
        nn.Conv3d(in_chans, embed_dim // 2, kernel_size=(1, kernel_size, kernel_size), stride=(1,stride,stride), padding=(0,padding,padding)),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 4, 1)),
        (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
        (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 4, 1, 2, 3)),
        nn.GELU(),
        nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=(1, kernel_size, kernel_size), stride=(1,stride,stride), padding=(0,padding,padding)),
        (nn.Identity() if channel_first else Permute(0, 2, 3, 4, 1)),
        (norm_layer(embed_dim) if patch_norm else nn.Identity()),
    )
   
def _make_patch_expand(in_chans, out_chans, patch_size):
    return nn.Sequential(
        # nn.ConvTranspose2d(in_chans, in_chans, kernel_size=patch_size//2, stride=patch_size//2),
        # LayerNorm(out_chans, data_format="channels_first"),
        # nn.LeakyReLU(),
        nn.ConvTranspose2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size),
    )

# def _make_patch_expand(in_chans, out_chans, patch_size):
#     return nn.Sequential(
#         nn.ConvTranspose2d(in_chans, in_chans, kernel_size=patch_size//2, stride=patch_size//2),
#         StackedConvBlocks(
#             num_convs=2, 
#             conv_op=nn.Conv2d,
#             input_channels=in_chans,
#             output_channels=in_chans, 
#             kernel_size=3,
#             initial_stride=1,
#             conv_bias=True,
#             norm_op=nn.InstanceNorm2d,
#             norm_op_kwargs={'eps': 1e-5, 'affine': True},
#             dropout_op=None,
#             dropout_op_kwargs=None,
#             nonlin=nn.LeakyReLU,
#             nonlin_kwargs={'inplace': True},
#             nonlin_first=False),
#         nn.ConvTranspose2d(in_chans, in_chans, kernel_size=patch_size//2, stride=patch_size//2),
#         StackedConvBlocks(
#             num_convs=2, 
#             conv_op=nn.Conv2d,
#             input_channels=in_chans,
#             output_channels=out_chans, 
#             kernel_size=3,
#             initial_stride=1,
#             conv_bias=True,
#             norm_op=nn.InstanceNorm2d,
#             norm_op_kwargs={'eps': 1e-5, 'affine': True},
#             dropout_op=None,
#             dropout_op_kwargs=None,
#             nonlin=nn.LeakyReLU,
#             nonlin_kwargs={'inplace': True},
#             nonlin_first=False),
#     )
    
def _make_patch_expand3D(in_chans, out_chans, patch_size):
    return nn.Sequential(
        nn.ConvTranspose3d(in_chans, out_chans, kernel_size=(1,patch_size//2,patch_size//2), stride=(1,patch_size//2,patch_size//2)),
        LayerNorm(out_chans, data_format="channels_first"),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(out_chans, out_chans, kernel_size=(1,patch_size//2,patch_size//2), stride=(1,patch_size//2,patch_size//2)),
    )

class ReconVSSBlock(VSSBlock):
    def __init__(
        self,
        hidden_dim: int = 0,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, **kwargs)
        self.conv = Conv2d_channel_last(hidden_dim, hidden_dim, 3, 1, 1)
        self.CA = nn.Sequential(
            Permute(0,3,1,2),
            CAB(num_feat=hidden_dim),
            Permute(0,2,3,1)
        )
        self.conv1x1_1 = Conv2d_channel_last(hidden_dim, hidden_dim, 1, 1, 0)
        self.conv1x1_2 = Conv2d_channel_last(hidden_dim, hidden_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv1x1_1(x) + self.drop_path(self.op(self.norm(x)))
        x = self.conv1x1_2(x) + self.drop_path(self.CA(self.conv(self.norm2(x))))
        return x

class VSSM_Decoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 ouput_channels: int,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision: bool = True
                 ):

        super().__init__()
        stages = []
        transpconvs = []
        seg_layers = []
        if encoder.conv_op == Conv2d_channel_last:
            transpconv_op = Conv2dTran_channel_last
        elif encoder.conv_op == nn.Conv2d:
            transpconv_op = nn.ConvTranspose2d
        elif encoder.conv_op == nn.Conv3d:
            transpconv_op = nn.ConvTranspose3d
        else:
            transpconv_op = Conv3dTran_channel_last
        conv_op = encoder.conv_op
        n_stages_encoder = len(encoder.output_channels)
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]*2  # concatenate的时候这里不用*2
            input_features_skip = encoder.output_channels[-(s + 1)]*2  # concatenate的时候这里不用*2
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                in_channels=input_features_below, 
                out_channels=input_features_skip, 
                kernel_size=stride_for_transpconv, 
                stride=stride_for_transpconv,
            )
            )
            stage_modules = []
            # stage_modules.append(
            #     conv_op(
            #     in_channels=input_features_skip*2,
            #     out_channels=input_features_skip,
            #     kernel_size=1,
            #     bias=True)
            # )
            # stage_modules.append(LayerNorm(input_features_skip))
            # stage_modules.append(nn.LeakyReLU())
            # for _ in range(n_conv_per_stage[s-1]):
            stage_modules.append(
                StackedConvBlocks(
                    num_convs=n_conv_per_stage[s-1], 
                    conv_op=conv_op, 
                    input_channels=input_features_skip*2, # concatenate的时候这里不用*2
                    output_channels=input_features_skip, 
                    kernel_size=3,
                    initial_stride=1,
                    conv_bias=True,
                    norm_op=Instance2d_channel_last,
                    norm_op_kwargs={'eps': 1e-5, 'affine': True},
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nn.LeakyReLU,
                    nonlin_kwargs={'inplace': True},
                    nonlin_first=False))
                # VSSBlock(
                #     hidden_dim=input_features_skip,
                #     ssm_init="v2",
                #     forward_type="v2_noz",
                #     # ssm_d_state=64,
                #     # ssm_ratio=1.0,
                #     # ssm_conv_bias=False,
                #     # ssm_act_layer=nn.SiLU
                # ))
            stages.append(nn.Sequential(*stage_modules))
            # seg_layers.append(encoder.conv_op(input_features_skip, ouput_channels, 1, 1, 0, bias=True))
            
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        # self.seg_layers = nn.ModuleList(seg_layers)
        self.deep_supervision = deep_supervision
    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)  # 输入通道1024
            x = torch.cat((x, skips[-(s+2)]), -1)
            x = self.stages[s](x)
            # if self.deep_supervision:
            #     seg_outputs.append(self.seg_layers[s](x))
            # elif s == (len(self.stages) - 1):
            #     seg_outputs.append(self.seg_layers[-1](x))
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
            stages.append(_make_patch_embed_v2(
                            in_chans=input_channels,
                            embed_dim=features_per_stage[0],
                            patch_size=patch_size))
        else:
            stages.append(_make_patch_embed_v23D(
                            in_chans=input_channels,
                            embed_dim=features_per_stage[0],
                            patch_size=patch_size))
        for s in range(n_stages):
            stage_modules = []
            conv_stride = strides[s]
            conv_padding = padding[s]
            if s > 0:
                stage_modules.append(
                    conv_op(
                    in_channels=input_channels,
                    out_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=conv_stride,
                    padding=conv_padding
                    )
                    )
            # stage_modules.append(LayerNorm(features_per_stage[s]))
            # stage_modules.append(nn.LeakyReLU())
            for _ in range(n_conv_per_stage[s]):
                stage_modules.append(
                VSSBlock(
                    hidden_dim=features_per_stage[s],
                    ssm_init="v2",
                    forward_type="v2_noz"
                    # ssm_d_state=64,
                    # ssm_ratio=1.0,
                    # ssm_conv_bias=False,
                    # ssm_act_layer=nn.SiLU
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

class VSSMUNet(nn.Module):
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
                DC_type: str = 'AM'
                ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        assert DC_type in ('VN', 'AM', '')
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
        if conv_op==nn.Conv2d:
            self.patch_embed = _make_patch_embed_v2(in_chans=input_channels,
                                                        embed_dim=d_model,
                                                        patch_size=patch_size)
            self.patch_expand = _make_patch_expand(in_chans=d_model*2,  # concatenate的时候这里不用*2
                                                        out_chans=input_channels,
                                                        patch_size=patch_size)
        else:
            self.patch_embed = _make_patch_embed_v23D(in_chans=input_channels,
                                                        embed_dim=d_model,
                                                        patch_size=patch_size)
            self.patch_expand = _make_patch_expand3D(in_chans=UNet_base_num_features*2,  # concatenate的时候这里不用*2
                                                        out_chans=input_channels,
                                                        patch_size=patch_size)
        if DC_type != '':
            self.DC = DC_layer2c(DC_type=DC_type, soft=True)
        else:
            print('Hard DC')
            self.DC = DC_layer2c(DC_type=DC_type, soft=False)
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
                                    deep_supervision)

        self.contour_encoder = VSSM_Encoder(input_channels=input_channels,
                                    patch_size=patch_size,
                                    n_stages=n_stages, 
                                    n_conv_per_stage=n_conv_per_stage,
                                    features_per_stage=features_per_stage, 
                                    conv_op=conv_op,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    padding=padding,
                                    return_skips=True)
        # self.apply(self._init_weights)
        
    # def forward(self, x, mask, uk, res=None):
    #     B, T, C, h, w = x.size()
    #     x = x.reshape(B, T*C, h, w)
    #     # x = self.patch_embed(x)# (B, Hp, Wp, D)
    #     skips = self.encoder(x)
    #     out = self.decoder(skips)
    #     out_put = out[0].permute(0,3,1,2)#(B, T*C, h, w)
    #     if res is not None:
    #         return self.DC(self.patch_expand(out_put).reshape(B, T, C, h, w) + res, mask, uk)
    #     else:
    #         return self.DC(self.patch_expand(out_put).reshape(B, T, C, h, w), mask, uk)
    
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
    
    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        if w % 32 !=0:
            w_pad = 32 - w % 32
            w1 = math.floor((w_pad) / 2)
            w2 = math.ceil((w_pad) / 2)
        else:
            w1 = 0
            w2 = 0
        if h % 32 !=0:
            h_pad = 32 - h % 32
            h1 = math.floor((h_pad) / 2)
            h2 = math.ceil((h_pad) / 2)
        else:
            h1 = 0
            h2 = 0
        x = F.pad(x, (w1, w2, h1, h2, 0, 0, 0, 0), 'constant', 0)

        return x, (h1, h2, w1, w2)

    def unpad(
        self,
        x: torch.Tensor,
        pad: tuple
    ) -> torch.Tensor:
        return x[..., pad[0]: x.shape[-2] - pad[1], pad[2]: x.shape[-1] - pad[3]]
    # def forward(self, uk, mask, uk0):
    #     k_out = []
    #     for i, x in enumerate(uk):
    #         x = ifft2c(x, dim=(-2, -1))
    #         B, C, h, w, _ = x.shape
    #         x = x.permute(0,4,1,2,3).reshape(B, 2*C, h, w)
    #         x, mean, std = self.norm(x)
    #         skips = self.encoder(x)
    #         out_put = self.decoder(skips)[0].permute(0,3,1,2)
    #         out_put = self.patch_expand(out_put)
    #         out_put = self.unnorm(out_put, mean, std)
    #         i_rec = out_put.reshape(B, 2, C, h, w).permute(0,2,3,4,1)
    #         Gk = fft2c(i_rec, dim=(-2, -1))
    #         k_dc = self.DC(uk[i], mask[i], uk0[i])
    #         if self.out_put:
    #             k_out.append(ifft2c((k_dc - Gk), dim=(-2, -1)).squeeze())
    #         else:
    #             k_out.append(k_dc - Gk)
    #     return k_out
    
    def forward(self, input, contour, mask, uk0):
        # x = ifft2c(uk, dim=(-2, -1))
        x = input
        # x = torch.cat((input,contour),dim=1)  # 在c上拼接
        B, C, h, w, _ = x.shape
        x = x.permute(0,4,1,2,3).reshape(B, 1*C, h, w)  #原来是2*c
        # x, mean, std = self.norm(x)
        skips = self.encoder(x)

        # 对contour的预处理
        B1, C1, h1, w1, _ = contour.shape
        contour = contour.permute(0,4,1,2,3).reshape(B1, 1*C1, h1, w1)
        #
        contour_map = self.contour_encoder(contour) # 也对轮廓信息编码，得到对应特征 contour_map
        # out_put = self.decoder(skips)[0]  # 原始版本
        # 将两特征图拼接
        # mid = [a + b for a, b in zip(skips, contour_map)]

        mid = []
        for i in range(6):
            concatenated_matrix = torch.cat((skips[i], contour_map[i]), dim=-1)  # 在最后一维拼接
            mid.append(concatenated_matrix)
        
        out_put = self.decoder(mid)[0]
        out_put = self.patch_expand(out_put.permute(0,3,1,2))
        # out_put = self.unnorm(out_put, mean, std)
        i_rec2 = out_put.reshape(B, 1, C, h, w).permute(0,2,3,4,1) # B, 2, C, h, w
        # 我们在这里暂时不考虑DC
        # Gk = fft2c(i_rec2, dim=(-2, -1))
        # if self.DC_type == 'VN':
        #     k_out = self.DC(uk, mask, uk0) - Gk
        # elif self.DC_type == 'AM' or self.DC_type == '':
        #     k_out = self.DC(uk + Gk, mask, uk0)
        # if self.out_put:
        #     i_rec2 = ifft2c(k_out, dim=(-2, -1))
        #     i_rec = fastmri.complex_abs(i_rec2).squeeze(1)
        #     i_rec2 = i_rec2.permute(0,4,1,2,3).reshape(B, 2*C, h, w)
        #     # if dist.get_rank() == 0:
        #     # plt.imshow(torch.log(torch.view_as_complex(fft2c(i_rec2))[0,0,:,:].abs()).cpu().detach().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/kout.jpg');plt.close()
        #     # plt.imshow(torch.log(torch.view_as_complex(k_out)[0,0,:,:].abs()).cpu().detach().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/kout.jpg');plt.close()
        #     # plt.imshow(torch.view_as_complex(ifft2c(k_out, dim=(-2, -1))[0,0,:,:]).abs().cpu().detach().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/iout.jpg');plt.close()
        #     # plt.imshow(i_rec[0,:,:].abs().cpu().detach().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/iout.jpg');plt.close()
        #     return [i_rec, i_rec2]
        # else:
        #     return k_out
        return i_rec2

class VSSMUNet_unrolled(nn.Module):
    def __init__(self, iter=6, DC_type='VN', kwargs=None):
        super().__init__()
        self.layers = []
        for _ in range(iter-1):
            self.layers.append(VSSMUNet(**kwargs, DC_type=DC_type))
        self.layers.append(VSSMUNet(**kwargs, DC_type=DC_type, out_put=True))
        self.layers = nn.ModuleList(self.layers)
        # self.refine = UNet(DC_type='AM', output=True, **unet_kwargs)
        self.refine = PlainConvUNet(**plainunet_kwargs)
        self.apply(self._init_weights)
        
    def forward(self, x, contour, mask, dorefine=False):
        uk0 = x.clone()
        x = x.unsqueeze(-1)
        contour = contour.unsqueeze(-1)
        for layer in self.layers:
            i_rec2 = layer(x, contour, mask, uk0)
        # if not dorefine:
        #     return out[0]#self.refine(x[1], mask, uk0)]
        # else:
        #     i_recr = self.refine(out[1])
        #     i_rec = (i_recr**2).sum(dim=1).sqrt()
        #     # plt.imshow(torch.log(torch.view_as_complex(fft2c(i_recr.permute(0,2,3,1)))[0,:,:].abs()).cpu().detach().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/koutr.jpg');plt.close()
        #     # plt.imshow(i_rec[0,:,:].abs().cpu().detach().numpy(), cmap=plt.cm.gray);plt.savefig(r'/mnt/data/zlt/AR-Recon/ioutr.jpg');plt.close()
        out = torch.squeeze(i_rec2, dim=-1)
        return out
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=1e-2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, Instance2d_channel_last)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d, Conv2d_channel_last, Conv2dTran_channel_last)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            trunc_normal_(m.weight, std=1e-2)
            
class DC_layer2c(nn.Module):
    def __init__(self, DC_type, soft=False):
        super(DC_layer2c, self).__init__()
        self.soft = soft
        if self.soft:
            if DC_type != '':
                self.dc_weight = nn.Parameter(torch.Tensor([1]))
            else:
                self.dc_weight = None
    def forward(self, k_rec, mask, k_ref):
        if len(mask.shape) < len(k_rec.shape):
            mask = mask.unsqueeze(-1)
        masknot = 1 - mask
        if self.soft:
            k_out = masknot * k_rec + mask * k_rec * (1 - self.dc_weight) + mask * k_ref * self.dc_weight
        else:
            k_out = masknot * k_rec + mask * k_ref
        return k_out
    
if __name__ == "__main__":
    from utils.utils import kwargs_vssm_unrolled as kwargs
    model = VSSMUNet_unrolled(iter=6, DC_type='AM', kwargs=kwargs)
    print(model)