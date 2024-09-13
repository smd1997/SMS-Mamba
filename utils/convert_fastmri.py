from pathlib import Path
from tqdm import tqdm
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms as T
from natsort import natsorted
import torch.nn.functional as F
import torch
import fastmri
import h5py
import numpy as np
import os



def crop_or_pad(uniform_train_resolution, kdata):
    kdata = _pad_if_needed(uniform_train_resolution, kdata)
    kdata = _crop_if_needed(uniform_train_resolution, kdata)
    return kdata


def _crop_if_needed(uniform_train_resolution, image):
    w_from = h_from = 0

    if uniform_train_resolution[0] < image.shape[-2]:
        w_from = (image.shape[-2] - uniform_train_resolution[0]) // 2
        w_to = w_from + uniform_train_resolution[0]
    else:
        w_to = image.shape[-2]

    if uniform_train_resolution[1] < image.shape[-1]:
        h_from = (image.shape[-1] - uniform_train_resolution[1]) // 2
        h_to = h_from + uniform_train_resolution[1]
    else:
        h_to = image.shape[-1]

    return image[:, w_from:w_to, h_from:h_to]


def _pad_if_needed(uniform_train_resolution, image):
    pad_w = uniform_train_resolution[0] - image.shape[-2]
    pad_h = uniform_train_resolution[1] - image.shape[-1]

    if pad_w > 0:
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left
    else:
        pad_w_left = pad_w_right = 0

    if pad_h > 0:
        pad_h_left = pad_h // 2
        pad_h_right = pad_h - pad_h_left
    else:
        pad_h_left = pad_h_right = 0

    return F.pad(image, (pad_h_left, pad_h_right, pad_w_left, pad_w_right, 0, 0), 'constant', 0)


if __name__ == "__main__":
    center_fractions = [0.04]
    accelerations = [8]
    uniform_train_resolution = [320, 320]
    task_name = 'knee_singlecoil'
    partlist = ['train', 'val']  # , 'test']
    slicenum = [34742, 7135]
    cutslices = True  #False
    combine = False
    for part, s in zip(partlist, slicenum):
        p = 0
        if combine:#for DCCNN
            Data_all = np.zeros((s, uniform_train_resolution[0], uniform_train_resolution[1], 1), dtype=np.complex64)
            ps = 0
        num_slice_info = []
        fastmri_path = '/mnt/data/dataset/fastMRI/' + task_name + '/' + part
        if not os.path.exists(fastmri_path):
            continue
        preprocess_name = 'preprocessed_{}_{}x_{}c320x320'.format(task_name, str(accelerations[0]),
                                                                  str(center_fractions[0]))
        path = Path(fastmri_path)
        filenames = natsorted([file.name for file in path.rglob("*.h5")])
        files = tqdm(filenames)
        print("Convering fullsampled h5 to npy & Preprocess")
        print("Start preprocessing, toltal train number: %s" % str(len(files)))
        # if not os.path.exists(os.path.join(fastmri_path, "npydataraw")):
        #     os.makedirs(os.path.join(fastmri_path, "npydataraw"))
        if not os.path.exists(os.path.join(fastmri_path, preprocess_name)):
            os.makedirs(os.path.join(fastmri_path, preprocess_name))
        # if not os.path.exists(os.path.join(fastmri_path, preprocess_name+'/ui')):
        #     os.makedirs(os.path.join(fastmri_path, preprocess_name+'/ui'))
        if not os.path.exists(os.path.join(fastmri_path, preprocess_name + '/fi')):
            os.makedirs(os.path.join(fastmri_path, preprocess_name + '/fi'))
        if not os.path.exists(os.path.join(fastmri_path, preprocess_name + '/uk')):
            os.makedirs(os.path.join(fastmri_path, preprocess_name + '/uk'))
        # if not os.path.exists(os.path.join(fastmri_path, preprocess_name + '/fk')):
        #     os.makedirs(os.path.join(fastmri_path, preprocess_name + '/fk'))
        if not os.path.exists(os.path.join(fastmri_path, preprocess_name + '/mask')):
            os.makedirs(os.path.join(fastmri_path, preprocess_name + '/mask'))
        if cutslices:
            if not os.path.exists(os.path.join(fastmri_path, preprocess_name + '/ukslice')):
                os.makedirs(os.path.join(fastmri_path, preprocess_name + '/ukslice'))
            if not os.path.exists(os.path.join(fastmri_path, preprocess_name + '/fislice')):
                os.makedirs(os.path.join(fastmri_path, preprocess_name + '/fislice'))
            # if not os.path.exists(os.path.join(fastmri_path, preprocess_name + '/fislicecom')):
            #     os.makedirs(os.path.join(fastmri_path, preprocess_name + '/fislicecom'))
            if not os.path.exists(os.path.join(fastmri_path, preprocess_name + '/maskslice')):
                os.makedirs(os.path.join(fastmri_path, preprocess_name + '/maskslice'))
        for f in files:
            # if not os.path.exists(os.path.join(fastmri_path, "npydataraw")+"/"+f[:-3]+".npy"):
            data = h5py.File(os.path.join(fastmri_path, f))['kspace'][()]
            num_slice_info.append(data.shape[0])
            # np.save(os.path.join(fastmri_path, "npydataraw")+"/"+f[:-3], data)
            # else:
            # data = np.load(os.path.join(fastmri_path, "npydataraw")+"/"+f[:-3]+".npy")
            # print(data.shape)
            if not os.path.exists(os.path.join(fastmri_path, preprocess_name) + "/mask/" + f[:-3] + "_mask.npy"):
                slice_kspace2 = T.to_tensor(data)  # Convert from numpy array to pytorch tensor (C,kx,ky)
                slice_image = crop_or_pad(uniform_train_resolution, torch.view_as_complex(fastmri.ifft2c(
                    slice_kspace2)))  # Apply Inverse Fourier Transform to get the complex image (C,kx,ky,2)
                slice_image = slice_image / torch.abs(slice_image).max()
                if not combine:
                    slice_image = torch.view_as_real(slice_image)
                    if task_name.find('single') != -1:
                        slice_image_rss = slice_image
                    else:
                        slice_image_rss = fastmri.rss(slice_image, dim=1)
                    slice_kspace2 = fastmri.fft2c(slice_image_rss)
                    mask_func = RandomMaskFunc(center_fractions=center_fractions, accelerations=accelerations)  # Create the mask function object
                    # masked_kspace, mask, _ = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space mask:(1,1,1,ky,1)
                    mask = np.load(os.path.join(fastmri_path, preprocess_name) + "/mask/" + f[:-3] + "_mask.npy")
                    masked_kspace = slice_kspace2 * mask
                    sampled_image = fastmri.ifft2c(masked_kspace)
                
                    sampled_image_abs = fastmri.complex_abs(sampled_image)
                    if task_name.find('single') != -1:
                        sampled_image_rss = sampled_image_abs
                    else:
                        sampled_image_rss = fastmri.rss(sampled_image_abs, dim=1)
                    if cutslices:
                        for i in range(sampled_image_rss.shape[0]):
                            np.save(os.path.join(fastmri_path, preprocess_name) + "/maskslice/" + f[:-3] + "_mask_" + str(p), mask)  # (1,kx,ky,1);
                            np.save(os.path.join(fastmri_path, preprocess_name)+"/ukslice/"+f[:-3]+"_uk_"+str(p), masked_kspace[i].unsqueeze(0))#(1,kx,ky,2);
                            np.save(os.path.join(fastmri_path, preprocess_name)+"/fislice/"+f[:-3]+"_fi_"+str(p), fastmri.complex_abs(slice_image_rss[i]))#(kx,ky);
                            # if os.path.exists(os.path.join(fastmri_path, preprocess_name + '/fislicecom')):
                            #     np.save(os.path.join(fastmri_path, preprocess_name)+"/fislicecom/"+f[:-3]+"_fi_"+str(p), torch.view_as_complex(slice_image_rss[i]))#(kx,ky);
                            # np.save(os.path.join(fastmri_path, preprocess_name)+"/uislice/"+f[:-3]+"_ui_"+str(p), sampled_image_rss[i].unsqueeze(-1))#(kx,ky,1);
                            # np.save(os.path.join(fastmri_path, preprocess_name)+"/fislice/"+f[:-3]+"_fi_"+str(p), fastmri.complex_abs(slice_image_rss)[i].unsqueeze(-1))#(kx,ky,1);
                            p = p + 1
                    else:
                        np.save(os.path.join(fastmri_path, preprocess_name)+"/ui/"+f[:-3]+"_ui", sampled_image_rss)#(kz,kx,ky);
                        np.save(os.path.join(fastmri_path, preprocess_name)+"/fi/"+f[:-3]+"_fi", slice_image_rss)#(kz,kx,ky,2);
                        np.save(os.path.join(fastmri_path, preprocess_name)+"/uk/"+f[:-3]+"_uk", masked_kspace)#(kz,kx,ky,2);
                        # np.save(os.path.join(fastmri_path, preprocess_name)+"/fk/"+f[:-3]+"_fk", slice_kspace2)#(kz,kx,ky,2);
                        np.save(os.path.join(fastmri_path, preprocess_name)+"/mask/"+f[:-3]+"_mask", mask)#(1,kx,ky,1);
                    # print(slice_image_rss.shape, slice_image_rss.shape, mask.shape)
                else:
                    Data_all[ps : ps+slice_image.shape[0], :, :, :] = slice_image.unsqueeze(-1).numpy()
                    ps = ps + slice_image.shape[0]
            files.set_description("Converting %s" % f)
        np.save(fastmri_path+'/datainfo.npy', [filenames, num_slice_info])
        if combine:
            np.save(os.path.join(fastmri_path, preprocess_name) + '/Data_all.npy', Data_all)
