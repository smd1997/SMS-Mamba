import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple
import math, copy
import numpy as np
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

def fft(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-4, -3)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, dim=dim, norm=norm)
    if shift:
        data = torch.fft.fftshift(data, dim=dim)
    return data

def ifft(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-4, -3)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.ifftn(data, dim=dim, norm=norm)
    if shift:
        data = torch.fft.fftshift(data, dim=dim)
    return data

class TUNet(nn.Module):
    def __init__(self,
                T: int,
                hw: int,
                N: int,
                d_model: int,
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
                ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
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
        # d_emb = features_per_stage[-1]
        # all_strides = np.prod(strides)
        # d_in = hw*hw//(all_strides*all_strides)*d_emb
        # if T > 1:
        #     self.temporal_trans = make_model(src_padding_len=T, src_len=T, N=N, d_model=d_model, d_in=d_in, d_ff=d_model*4, h=8)
        # else:
        #     self.temporal_trans = nn.Identity()
        num_output_channels = input_channels
        self.decoder = UNetDecoder(self.encoder, num_output_channels, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        self.DC = DC_layer()
        self.initialize()
    def forward(self, src, mask, uk, src_mask=None, tgt_mask=None):
        # mask, uk:(t,c,kx,ky)
        B, T, C, h, w = src.size()
        src = src.reshape(B, T*C, h, w)
        skips = self.encoder(src)
        # if not isinstance(self.temporal_trans, nn.Identity) and src_mask is not None:
        #     skips[-1] = self.temporal_trans(skips[-1], skips[-1], src_mask, tgt_mask).flatten(0,1)
        out = self.decoder(skips)
        return self.DC(out[0].reshape(B, T, C, h, w), mask, uk)#out[0].reshape(B, T, C, h, w)

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
                
    # def _make_refine(self, in_chans, out_chans):
    #     return nn.Sequential(
    #         nn.ConvTranspose2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1),
    #         nn.InstanceNorm2d(),
    #         nn.LeakyReLU(),
    #         nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
    #     )

class DC_layer(nn.Module):
    def __init__(self):
        super(DC_layer, self).__init__()

    def forward(self, i_rec, mask, uk, fov_mask=1):
        uk = torch.view_as_complex(uk.unsqueeze(2))
        k_rec_com = fft(i_rec, shift=True, dim=(3,4))
        masknot = 1 - mask

        k_out = (masknot * k_rec_com + uk) * fov_mask
        x_out_com = ifft(k_out, shift=True, dim=(3,4))
        x_out = torch.abs(x_out_com)
        return x_out

# class DC_layer(nn.Module):
#     def __init__(self):
#         super(DC_layer, self).__init__()

#     def forward(self, mask, x_rec, k_under, fov_mask=1):
#         # 输入是8通道实数图像（4通道复数拼成8通道）
#         # plot_tensor(x_rec, './x_rec.tif', is_img=True)
#         x_rec_com = torch.complex(x_rec[:, :4].double(), x_rec[:, 4:].double())
#         k_rec_com = torch.fft.fftshift(torch.fft.fft2(x_rec_com), dim=(-2, -1))
#         k_rec = torch.cat([k_rec_com.real, k_rec_com.imag], dim=1)
#         # plot_tensor(k_rec, './k_rec.tif', is_img=False)
#         # masks = torch.cat((mask, mask), 1)
#         # matrixones = torch.ones_like(masks.data)
#         masknot = 1 - mask

#         k_out = (masknot * k_rec + k_under) * fov_mask
#         k_out_com = torch.complex(k_out[:, :4], k_out[:, 4:])
#         x_out_com = torch.fft.ifft2(torch.fft.ifftshift(k_out_com, dim=((-2, -1))))
#         x_out = torch.cat([x_out_com.real, x_out_com.imag], dim=1).float()
#         # plot_tensor(x_out, './x_out.tif', is_img=True)
#         return x_out

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    tgt_mask = torch.where(tgt != pad, 1, 0)#.unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

def make_std_mask_eg(tgt, pad):
    tgt_mask = torch.where(tgt != pad, 1, 0).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = (query @ key.transpose(-2, -1) / math.sqrt(d_k))#.to(torch.float32)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -torch.inf)#-65504.0 #-1e9
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    m = p_attn @ value
    return m, p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, padding_len, src_len, encoder, decoder, src_embed=None, tgt_embed=None, out_proj=None):
        super(EncoderDecoder, self).__init__()
        self.padding_len = padding_len
        self.src_len = src_len
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.out_proj = out_proj
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        # src(B,T,C,h,w)
        BT, C, h, w = src.size()
        src = src.reshape(BT//self.src_len, self.src_len, C*h*w)# (B, T, D)
        src = self.src_embed(src)
        if self.padding_len > BT:
            src = F.pad(src, (0,0,0,self.padding_len - T,0,0), 'constant', 0)
        return self.encoder(src, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory: encode output
        BT, C, h, w = tgt.size()
        tgt = tgt.reshape(BT//self.src_len, self.src_len, C*h*w)# (B, T, D)
        tgt = self.tgt_embed(tgt)
        if self.padding_len > BT:
            tgt = F.pad(tgt, (0,0,0,self.padding_len - T,0,0), 'constant', 0)
        out = self.decoder(tgt, memory, src_mask, tgt_mask)
        out = self.out_proj(out).reshape(BT//self.src_len, self.src_len, C, h, w)
        return out

def make_model(src_padding_len, src_len, N=6, 
               d_model=512, d_in=32768, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        src_padding_len,
        src_len,
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(nn.Linear(d_in, d_model)),
        nn.Sequential(nn.Linear(d_in, d_model)),
        nn.Sequential(nn.Linear(d_model, d_in))
    )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
if __name__ == "__main__":
    import sys
    sys.path.append(__file__.rsplit('/', 2)[0])
    from utils.utils import SEED, kwargs
    # (s,kx,ky,kz,kt)
    B, c, h, w, t = 2, 1, 128, 128, 30
    T = 30
    device = torch.device("cpu")
    us = torch.randn(B, c, h, w, t).permute(0,4,1,2,3).to(device)
    gt = torch.randn(B, c, h, w, t).permute(0,4,1,2,3).to(device)
    current_L = t
    pad = 0
    src = F.pad(torch.arange(1,current_L+1), (0,T-current_L), 'constant', pad)
    src = src.unsqueeze(0).repeat(B,1)#(B,T)
    tgt = F.pad(torch.arange(1,current_L+1), (0,T-current_L), 'constant', pad)
    tgt = tgt.unsqueeze(0).repeat(B,1)#(B,T)
    src_mask = torch.where(src != pad, 1, 0).unsqueeze(-2).to(device)
    tgt_mask = make_std_mask_eg(tgt, pad).to(device)
    
    # Modified from dynamic_network_architectures
    model = TUNet(
        **kwargs,
    ).to(device)
    model.apply(InitWeights_He(1e-2))
    out = model(us, src_mask, tgt_mask)
    pass
    # for i in out:
    #     print(i.shape)