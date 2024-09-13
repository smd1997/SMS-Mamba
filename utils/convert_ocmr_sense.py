from read_ocmr import read_ocmr
from pathlib import Path
from tqdm import tqdm
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms as T
from ESPIRIT import espirit
from scipy.io import savemat
import torch
import torch.fft
import fastmri
import math
import numpy as np
import os

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
    if padsize[0] - kdata.shape[2]>=0:
        ifpadx = True
        if px != 0:
            px = padsize[0] - px
        # bi-directional padding
        if px % 2 != 0:
            px1 = px // 2
            px2 = px1 + 1
        else:
            px1 = px // 2
            px2 = px // 2
    else:
        ifpadx = False
        px1=(kdata.shape[2]//2)-padsize[0]//2
        px2=(kdata.shape[2]//2)+padsize[0]//2
    if padsize[1] - kdata.shape[3]>=0:
        ifpady = True
        if py != 0:
            py = padsize[1] - py
        if py % 2 != 0:
            py1 = py // 2
            py2 = py1 + 1
        else:
            py1 = py // 2
            py2 = py // 2
    else:
        ifpady = False
        py1=(kdata.shape[3]//2)-padsize[1]//2
        py2=(kdata.shape[3]//2)+padsize[1]//2
    if coilpd != None:
        return px1, px2, py1, py2, coilpd-c, ifpadx, ifpady
    else:
        return px1, px2, py1, py2, None, ifpadx, ifpady

if __name__ == "__main__":
    OCMR_path = '/mnt/data/dataset/OCMR_data'
    # OCMR_path = '/home/bmec/data/OCMR_data'
    padsize = [128, 128]
    cropsize = padsize
    dataset = []
    center_fractions=[0.04]
    accelerations=[8]
    preprocess_name = 'preprocessed_{}x{}_{}x_{}c_sense'.format(str(padsize[0]), str(padsize[1]), str(accelerations[0]), str(center_fractions[0]))
    path = Path(OCMR_path)
    files = [file.name for file in path.rglob("f*.h5")]
    files.sort()
    files = files[:53]
    files =  tqdm(files)
    print("Convering fullsampled h5 to npy & Preprocess")
    print("Start preprocessing, toltal train number: %s" % str(len(files)))
    if not os.path.exists(os.path.join(OCMR_path, "npydataraw")):
        os.makedirs(os.path.join(OCMR_path, "npydataraw"))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name)):
        os.makedirs(os.path.join(OCMR_path, preprocess_name))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name+'/smap')):
        os.makedirs(os.path.join(OCMR_path, preprocess_name+'/smap'))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name+'/uk')):
        os.makedirs(os.path.join(OCMR_path, preprocess_name+'/uk'))
    if not os.path.exists(os.path.join(OCMR_path, preprocess_name+'/fi')):
        os.makedirs(os.path.join(OCMR_path, preprocess_name+'/fi'))
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
            # print(kdata.shape)
        if not os.path.exists(os.path.join(OCMR_path, preprocess_name)+"/smap/"+f[:-3]+"_smap.npy"):
            kdata = kdata.transpose(5,3,0,1,2,4)#(kx,ky,kz,c,kt,s)->(s,c,kx,ky,kz,kt)
            # 1.padding kx and ky to cropsize*N
            # This step aims to formulate the input size. To keep the zero point, padding is applied 
            # on both end of kx and ky axis.
            # px1, px2, py1, py2, _, ifpadx, ifpady = get_padding_size(padsize=padsize, coilpd=None, kdata=kdata)
            # if ifpadx:
            #     kdata = np.pad(kdata, ((0,0),(0,0),(px1,px2),(0,0),(0,0),(0,0)),'constant', constant_values=(0))
            # else:
            #     kdata = kdata[:, :, px1:px2, :, :, :]
            # if ifpady:
            #     kdata = np.pad(kdata, ((0,0),(0,0),(0,0),(py1,py2),(0,0),(0,0)),'constant', constant_values=(0))
            # else:
            #     kdata = kdata[:, :, :, py1:py2, :, :]
            fs_kspace = torch.view_as_complex(T.to_tensor(kdata))
            s = kdata.shape[0]
            kx = kdata.shape[2]
            ky = kdata.shape[3]
            print(kdata.shape)
            # 2.generate uk
            mask_func = RandomMaskFunc(center_fractions=center_fractions, accelerations=accelerations)
            fs_kspace = fs_kspace.permute(0,1,4,5,2,3)#com (s,c,kx,ky,kz,kt)->(s,c,kz,kt,kx,ky)
            fs_image = ifft(fs_kspace, dim=(-1,-2))# com (s,c,kz,kt,kx,ky)
            # sens_map = np.load(os.path.join(OCMR_path, preprocess_name)+"/smap/"+f[:-3]+"_smap.npy")
            
            fs_image = fs_image.squeeze(2)# (s,c,kt,kx,ky)
            if ky>128:
                fs_image = fs_image[..., kx//2-cropsize[0]//2:kx//2+cropsize[0]//2, ky//2-cropsize[0]//2:ky//2+cropsize[0]//2]
            else:
                fs_image = fs_image[..., kx//2-cropsize[0]//2:kx//2+cropsize[0]//2, :]
            new_fs_kspace = fft(fs_image, dim=(-1,-2))# com (s,c,kz,kt,kx,ky)
            new_fs_kspace = new_fs_kspace.squeeze(2)# (s,c,kt,kx,ky)
            print(new_fs_kspace.shape)
            kt, kx, ky = new_fs_kspace.shape[-3], new_fs_kspace.shape[-2], new_fs_kspace.shape[-1]
            mask = torch.zeros(kx, ky, kt)
            for t in range(kt):
                _, maskt, _ = T.apply_mask(new_fs_kspace.unsqueeze(-1), mask_func)
                mask[:,:,t] = maskt.squeeze().unsqueeze(0).repeat(kx, 1)
            mask = mask.unsqueeze(0).repeat(s, 1, 1, 1)
            masked_kspace = new_fs_kspace*mask.permute(0,3,1,2).unsqueeze(1)
            # uk_image = ifft(masked_kspace, dim=(-1,-2))#com (s,c,t,kx,ky)
            # fs_image_rss = fastmri.rss(fs_image.abs(), dim=1)
            # sens_map = fs_image/fs_image_rss.unsqueeze(1)
            # sens_map = np.zeros_like(new_fs_kspace)
            # with tqdm(range(new_fs_kspace.shape[2])) as pbar:
            #     for ti in range(new_fs_kspace.shape[2]):
            #         for si in range(new_fs_kspace.shape[0]):
            #             es_maps = espirit(new_fs_kspace[si, :, ti, :, :].unsqueeze(0).permute(2,3,0,1).numpy(), 6, 24, 0.01, 0.9925)#(kx,ky,s,c,c)
            #             sens_map[si, :, ti, :, :] = es_maps[:,:,:,:,0].transpose(2,3,0,1)
            #         pbar.update(1)
            
            # np.save(os.path.join(OCMR_path, preprocess_name)+"/smap/"+f[:-3]+"_smap", torch.view_as_real(torch.tensor(sens_map)))#(s,c,t,kx,ky,2)
            np.save(os.path.join(OCMR_path, preprocess_name)+"/uk/"+f[:-3]+"_uk", torch.view_as_real(masked_kspace))#(s,c,kt,kx,ky,2)
            np.save(os.path.join(OCMR_path, preprocess_name)+"/fi/"+f[:-3]+"_fi", torch.view_as_real(fs_image))#(s,c,kt,kx,ky,2)
            np.save(os.path.join(OCMR_path, preprocess_name)+"/fk/"+f[:-3]+"_fk", torch.view_as_real(new_fs_kspace))#(s,c,kt,kx,ky,2)
            np.save(os.path.join(OCMR_path, preprocess_name)+"/mask/"+f[:-3]+"_mask", mask)#(s,kx,ky,t)
        files.set_description("Preprocessed %s" % f)
        # compute smap with JSENSE
        # sens_maps = []
        # for s in range(new_fs_kspace.shape[0]):
        #     sens_maps_tmp = []
        #     for t in range(new_fs_kspace.shape[3]):
        #         sens_maps_tmp.append(torch.from_numpy(JsenseRecon(new_fs_kspace[s,:,:,t,:,:].squeeze().numpy(), max_iter=30, show_pbar=True).run()).unsqueeze(0))
        #     sens_maps.append(torch.cat(sens_maps_tmp, dim=0).unsqueeze(0))#(t,c,kx,ky)
        # sens_map = torch.cat(sens_maps, dim=0).permute(0,2,1,3,4)#(s,c,t,kx,ky)
        # Compute smap with ACS
        # sens_maps = []
        # for s in range(masked_kspace.shape[0]):
        #     sens_maps_tmp = []
        #     for t in range(masked_kspace.shape[3]):
        #         nonzero_line = torch.nonzero(mask.squeeze().sum(0)[:,t])
        #         mid = torch.where(nonzero_line==128)[0]
        #         p1 = nonzero_line[mid]
        #         p1 = nonzero_line[mid]
        #         for i in range(mid+1, nonzero_line.shape[0]):
        #             if nonzero_line[i]==nonzero_line[i-1]+1:
        #                 p2 = nonzero_line[i]
        #             else:
        #                 break
        #         for i in range(mid-1, 0, -1):
        #             if nonzero_line[i]==nonzero_line[i+1]-1:
        #                 p1 = nonzero_line[i]
        #             else:
        #                 break
        #         print(p1, p2)
        #         acs = masked_kspace[s,:,:,t,:,:].squeeze().numpy()
        #         acs[:,:,0:p1-1] = 0
        #         acs[:,:,p2+1:] = 0
        #         smap = torch.from_numpy(JsenseRecon(acs, max_iter=10, show_pbar=False).run()).unsqueeze(0)
        #         sens_maps_tmp.append(smap)
        #         assert torch.any(torch.isnan(smap)) == False, "Nan"
        #     sens_maps.append(torch.cat(sens_maps_tmp, dim=0).unsqueeze(0))#(t,c,kx,ky)
        # sens_map = torch.cat(sens_maps, dim=0).permute(0,2,1,3,4)#(s,c,t,kx,ky)
