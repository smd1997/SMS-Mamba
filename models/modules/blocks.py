import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm import Mamba

class ChannelAttention(nn.Module):
    def __init__(self, channel, squeeze_factor=16, bias=False):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // squeeze_factor, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // squeeze_factor, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, squeeze_factor=16):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)
    
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = Conv2d_channel_last(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = Conv2d_channel_last(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = Conv2d_channel_last(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x.permute(0,3,1,2)).permute(0,2,3,1)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x.permute(0,3,1,2)).permute(0,2,3,1)

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        # print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.channel_token = channel_token ## whether to use channel as tokens
        self.conv1 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(num_features=dim)
        self.norm2 = nn.InstanceNorm3d(num_features=dim)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        B, T, H, W, d_model = x.shape
        x = x.permute(0,4,1,2,3)
        res = x
        x1 = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x1)) + res
        x = x.permute(0,2,3,4,1)
        assert d_model == self.dim
        x_flat = x.reshape(B, d_model, T*H*W).permute(0,2,1)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, T, H, W, d_model)
        # out = x_mamba.permute(0,2,1).reshape(B, d_model, T, H, W)

        return out
    
class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            if len(x.shape) == 4:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x
            elif len(x.shape) == 5:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
                return x

class Conv2d_channel_last(nn.Conv2d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return F.conv2d(x.permute(0,3,1,2), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,1)

class Conv3d_channel_last(nn.Conv3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return F.conv3d(x.permute(0,4,1,2,3), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,4,1)

class Conv2dTran_channel_last(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return F.conv_transpose2d(x.permute(0,3,1,2), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,1)
    
class Conv3dTran_channel_last(nn.ConvTranspose3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return F.conv_transpose3d(x.permute(0,4,1,2,3), weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups).permute(0,2,3,4,1)

class Instance2d_channel_last(nn.InstanceNorm3d):
    def forward(self, x: torch.Tensor):
        # B, H, W, C = x.shape
        return self._apply_instance_norm(x.permute(0,3,1,2)).permute(0,2,3,1)
    
class Instance3d_channel_last(nn.InstanceNorm3d):
    def forward(self, x: torch.Tensor):
        # B, T, H, W, C = x.shape
        return self._apply_instance_norm(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
    
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

def fft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    data = data.to(torch.float64)
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data.to(torch.float32)

def ifft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    data = data.to(torch.float64)
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data.to(torch.float32)