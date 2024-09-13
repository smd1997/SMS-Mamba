from read_ocmr import read_ocmr
from pathlib import Path
from tqdm import tqdm
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms as T
from typing import List, Optional
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
from fastmri.data.subsample import MaskFunc
import torch
import torch.fft
import fastmri
import math
import numpy as np
import os

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

def get_padding_size(padsize:list, coilpd, kdata):
    c = kdata.shape[1]
    px = kdata.shape[2] % padsize[0]
    py = kdata.shape[3] % padsize[1]
    if px != 0:
        px = padsize[0] - px
    if py != 0:
        py = padsize[1] - py
    # bi-directional padding
    if px % 2 != 0:
        px1 = px // 2
        px2 = px1 + 1
    else:
        px1 = px // 2
        px2 = px // 2
    if py % 2 != 0:
        py1 = py // 2
        py2 = py1 + 1
    else:
        py1 = py // 2
        py2 = py // 2
    if coilpd != None:
        return px1, px2, py1, py2, coilpd-c
    else:
        return px1, px2, py1, py2, None

if __name__ == "__main__":
    # v3: combine coils in image domain first then generate kspace
    OCMR_path = '/mnt/data/dataset/OCMR_data'
    # OCMR_path = '/home/bmec/data/OCMR_data'
    padsize = [512, 512]
    dataset = []
    center_fractions=[0.04]
    accelerations=[8]
    preprocess_name = 'preprocessed_{}x{}_{}x_{}c'.format(str(padsize[0]), str(padsize[1]), str(accelerations[0]), str(center_fractions[0]))
    path = Path(OCMR_path)
    files = [file.name for file in path.rglob("f*.h5")]
    files =  tqdm(files)
    print("Convering fullsampled h5 to npy & Preprocess")
    print("Start preprocessing, toltal train number: %s" % str(len(files)))
    if not os.path.exists(os.path.join(OCMR_path, "npydataraw")):
        os.makedirs(os.path.join(OCMR_path, "npydataraw"))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name)):
        os.makedirs(os.path.join(OCMR_path, preprocess_name))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name+'/ui')):
        os.makedirs(os.path.join(OCMR_path, preprocess_name+'/ui'))  
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name+'/fi')):
        os.makedirs(os.path.join(OCMR_path, preprocess_name+'/fi'))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name+'/uk')):
        os.makedirs(os.path.join(OCMR_path, preprocess_name+'/uk'))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name+'/fk')):
        os.makedirs(os.path.join(OCMR_path, preprocess_name+'/fk'))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name+'/mask')):
        os.makedirs(os.path.join(OCMR_path, preprocess_name+'/mask'))
        
    for f in files:
        if not os.path.exists(os.path.join(OCMR_path, "npydataraw")+"/"+f[:-3]+".npy"):
            kdata, param = read_ocmr(os.path.join(OCMR_path, f))#{'kx'  'ky'  'kz'  'coil'  'phase'  'set'  'slice'  'rep'  'avg'}
            if kdata.shape[-1]>1:
                kdata = np.sum(kdata, axis=-1)/kdata.shape[-1]
                kdata = kdata[:,:,:,:,:,0,:,0]#(kx,ky,kz,c,kt,s)
            else:
                kdata = kdata[:,:,:,:,:,0,:,0,0]
            np.save(os.path.join(OCMR_path, "npydataraw")+"/"+f[:-3], kdata)
        else:
            kdata = np.load(os.path.join(OCMR_path, "npydataraw")+"/"+f[:-3]+".npy")
        if not os.path.exists(os.path.join(OCMR_path, preprocess_name)+"/mask/"+f[:-3]+"_mask.npy"):
            kdata = kdata.transpose(5,3,0,1,2,4)#(kx,ky,kz,c,kt,s)->(s,c,kx,ky,kz,kt)
            # 1.padding kx and ky to cropsize*N
            # This step aims to formulate the input size. To keep the zero point, padding is applied 
            # on both end of kx and ky axis.
            px1, px2, py1, py2, _ = get_padding_size(padsize=padsize, coilpd=None, kdata=kdata)
            kdata_pad = np.pad(kdata, ((0,0),(0,0),(px1,px2),(py1,py2),(0,0),(0,0)),'constant', constant_values=(0))
            fs_kspace = torch.view_as_complex(T.to_tensor(kdata_pad))
            # 2.generate urss and mask
            fs_image = ifft(fs_kspace, shift=True, dim=(2,3))
            fs_image_abs = fs_image.abs()
            fs_image_rss = fastmri.rss(fs_image_abs, dim=1)
            # print(fs_image_rss.min(), fs_image_rss.max())
            for i in range(fs_image_rss.shape[-1]):
                fs_image_rss[...,i] = (fs_image_rss[...,i] - fs_image_rss[...,i].min())\
                    /(fs_image_rss[...,i].max() - fs_image_rss[...,i].min())
            new_fs_kspace = fft(fs_image_rss, shift=True, dim=(1,2))#com (s,kx,ky,kz,kt)
            mask_func = RandomMaskFunc(center_fractions=center_fractions, accelerations=accelerations)
            new_fs_kspace = new_fs_kspace.permute(0,3,4,1,2)#com (s,kx,ky,kz,kt)->(s,kz,kt,kx,ky)
            kt, kx, ky = new_fs_kspace.shape[-3], new_fs_kspace.shape[-2], new_fs_kspace.shape[-1]
            mask = torch.zeros(kx, ky, kt)
            for t in range(kt):
                _, maskt, _ = T.apply_mask(new_fs_kspace.unsqueeze(-1), mask_func)
                mask[:,:,t] = maskt.squeeze().unsqueeze(0).repeat(kx, 1)
            masked_kspace = new_fs_kspace*mask.permute(2,0,1).unsqueeze(0).unsqueeze(0)
            
            us_image = ifft(masked_kspace.permute(0,3,4,1,2), shift=True, dim=(1,2))#(s,kz,kt,kx,ky)->(s,kx,ky,kz,kt)
            us_image_rss = us_image.abs()
            # print(us_image_rss.min(), us_image_rss.max())
            # us_image_rss = (us_image_rss - us_image_rss.min())\
            #     /(us_image_rss.max() - us_image_rss.min())

            np.save(os.path.join(OCMR_path, preprocess_name)+"/ui/"+f[:-3]+"_ui", us_image_rss)#(s,kx,ky,kz,kt)
            np.save(os.path.join(OCMR_path, preprocess_name)+"/fi/"+f[:-3]+"_fi", fs_image_rss)#(s,kx,ky,kz,kt)
            np.save(os.path.join(OCMR_path, preprocess_name)+"/uk/"+f[:-3]+"_uk", torch.view_as_real(masked_kspace))#(s,kz,kt,kx,ky,2)
            np.save(os.path.join(OCMR_path, preprocess_name)+"/fk/"+f[:-3]+"_fk", torch.view_as_real(new_fs_kspace))#(s,kz,kt,kx,ky,2)
            np.save(os.path.join(OCMR_path, preprocess_name)+"/mask/"+f[:-3]+"_mask", mask)#(kx,ky,t)
        files.set_description("Preprocessing %s" % f)