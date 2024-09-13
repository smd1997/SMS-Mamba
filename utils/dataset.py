# from utils.read_ocmr import read_ocmr

import torch
import numpy as np
import fastmri
import pandas as pd
import torch.nn.functional as F
import os
import warnings
from natsort import natsorted
from torch.utils.data.dataset import Dataset
from torch.utils.data import IterableDataset
from pathlib import Path
from fastmri.data import transforms as T
from tqdm import tqdm
from models.modules.blocks import fft2c, ifft2c
warnings.filterwarnings('ignore')


class OCMRDataset(Dataset):
    def __init__(self, rootpath, pre_name, T=25, cutslice=True, index=None, name=''):
        super(OCMRDataset, self).__init__()
        self.dataset = []
        self.fdata_full = []
        self.smap_full = []
        self.num_sample = 0
        self.T = T
        self.T_S_num = []
        path = Path(rootpath+"/npydataraw")
        files = [file.name for file in path.rglob("f*.npy")]#[:2]
        files.sort()
        # index = None
        if index is not None:
            files = [files[i] for i in index]
        files = tqdm(files)
        print("Start reading ocmr %s, total instance number: %s" % (name, str(len(files))))
        print("cutslice:", cutslice)
        for file in files:
            # uk and fk refer to undersampled and fullsampled kspace data
            fi = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/fi/"+file[:-4]+"_fi.npy"))[:,:,:15,:,:,:]#(s,c,kt,kx,ky,2)
            fk = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/fk/"+file[:-4]+"_fk.npy"))#(s,c,kt,kx,ky,2)
            uk = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/uk/"+file[:-4]+"_uk.npy")).squeeze(2)[:,:,:15,:,:,:]#(s,c,kt,kx,ky,2)
            # io.savemat('/mnt/data/zlt/fi.mat', {'fi':torch.view_as_complex(fi[0,:,0,:,:,:]).detach().cpu().numpy()})
            # io.savemat('/mnt/data/zlt/uk.mat', {'uk':torch.view_as_complex(uk[0,:,0,:,:,:]).detach().cpu().numpy()})
            # smap = torch.view_as_real(torch.tensor(hdf5storage.loadmat(os.path.join(rootpath, pre_name)+"/smap/"+file[:-3]+"_smap.mat")['smap'].astype(np.complex64)))
            # np.save(os.path.join(rootpath, pre_name)+"/smap/"+file[:-3]+"_smap.npy", smap)
            # smap = smap[:,:,:15,:,:,:]
            smap = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/smap/"+file[:-3]+"_smap.npy"))[:,:,:15,:,:,:]#(s,c,kt,kx,ky,2)
            kmask = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/mask/"+file[:-3]+"_mask.npy"))[:,:,:,:15]#(s,kx,ky,t)
            # print(kmask.shape[0], kmask.shape[1], kmask.shape[2])
            # Combine real part and image part
            # fi = fastmri.complex_mul(fi, fastmri.complex_conj(smap)).sum(1)
            if name == 'infer':
                self.fdata_full.append(fi)#(s,c,kt,kx,ky,2)
                self.smap_full.append(smap)#(s,c,kt,kx,ky,2)
            num_slice = kmask.shape[0]
            current_L = kmask.shape[3]
            if num_slice>1:
                kmask = kmask[:1, :, :, :]
            # 1.Cutting temporal length
            uklist_L = []
            filist_L = []
            # fklist_L = []
            smaplist_L = []
            k_t = []
            if T > current_L:
                uk = F.pad(uk, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                fi = F.pad(fi, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                fk = F.pad(fk, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                sm = F.pad(smap, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                kmask = F.pad(kmask, (0,T-current_L), 'constant', 0)
                uklist_L.append(uk)
                filist_L.append(fi)
                # fklist_L.append(fk)
                smaplist_L.append(sm)
                k_t.append((0,T))
            elif current_L >= T:
                for i in range(current_L//T):
                    k_t.append((i*T,i*T+T))
                    uklist_L.append(uk[:,:,i*T:i*T+T,...])
                    filist_L.append(fi[:,:,i*T:i*T+T,...])
                    smaplist_L.append(smap[:,:,i*T:i*T+T,...])
                # for i in range(current_L-T+1):
                #     k_t.append((i,T+i))
                #     uklist_L.append(uk[:,:,i:T+i,...])
                #     filist_L.append(fi[:,i:T+i,...])
                #     # fklist_L.append(fk[:,:,i:T+i,...])
                #     smaplist_L.append(smap[:,:,i:T+i,...])
            self.T_S_num.append((len(k_t), current_L, uk.shape[0]))#num_pad, t, S
            #2.Select temporal length
            # cropped = False
            for i in range(len(uklist_L)):
                #3.Cut slice
                for s in range(uklist_L[i].shape[0]):
                    u = uklist_L[i][s]#(C,T,h,w,2)
                    fi = filist_L[i][s]
                    # fk = fklist_L[i][s]
                    sm = smaplist_L[i][s]
                    self.num_sample = self.num_sample + 1
                    self.dataset.append([u, fastmri.complex_abs(fi), sm, kmask.permute(0,3,1,2)[:,k_t[i][0]:k_t[i][1],...]])
            files.set_description("Reading preprocessed data %s" % file)
        print("Total sampled number %d" % self.num_sample)
        
    def __getitem__(self, index):
        
        return self.dataset[index]
    def __len__(self):
            
        return len(self.dataset)

class OCMRDatasetv2(Dataset):
    def __init__(self, rootpath, pre_name, T=5, index=None, name='', showpar=True, overlap_mode=True):
        super(OCMRDatasetv2, self).__init__()
        self.dataset = []
        self.fdata_full = []
        self.datasetpath = []
        self.num_sample = 0
        self.num_instance = 0
        self.instance_tuple = []
        self.name = name
        cached = False
        path = Path(rootpath+"/npydataraw")
        files = [file.name for file in path.rglob("f*.npy")]
        files.sort()
        # index = None
        if index is not None:
            files = [files[i] for i in index]
        if showpar:
            files = tqdm(files)
        cache_path = os.path.join(rootpath, pre_name)+'/'+self.name
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        else:
            cached = True
        # print("cutslice:", cutslice)
        for file in files:
            # uk and fk refer to undersampled and fullsampled kspace data
            self.num_instance = self.num_instance + 1
            if not cached:
                uk = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/uk/"+file[:-4]+"_uk.npy")).squeeze(2)#(s,c,kz,kt,kx,ky,2)->(s,c,kt,kx,ky,2)
                fi = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/fi/"+file[:-4]+"_fi.npy")).squeeze(2)#(s,c,kz,kt,kx,ky)->(s,c,kt,kx,ky)
                smap = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/smap/"+file[:-4]+"_smap.npy"))#(s,c,t,kx,ky,2)
            kmask = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/mask/"+file[:-4]+"_mask.npy"))#(s,kx,ky,t)
            # Combine real part and image part
            if name == 'infer':
                self.fdata_full.append(fi.permute(0,3,1,2,4).unsqueeze(2))#(s,kx,ky,kt,2)->(s,kt,kx,ky,2)
            num_slice = kmask.shape[0]
            current_L = kmask.shape[3]
            if num_slice>1:
                kmask = kmask[:1, :, :, :]
            k_t = []
            if T > current_L:
                if not cached:
                    uk = F.pad(uk, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                    fi = F.pad(fi, (0,0,0,0,0,T-current_L), 'constant', 0)
                    smap = F.pad(smap, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                    kmask = F.pad(kmask, (0,T-current_L), 'constant', 0)
                num_temporal = 1
                k_t.append((0,T))
            elif current_L >= T:
                if overlap_mode:
                    num_temporal = current_L-T+1
                    for i in range(num_temporal):
                        k_t.append((i,T+i))
                else:
                    if current_L % T > 0:
                        pad = T - current_L % T
                        if not cached:
                            uk = F.pad(uk, (0,0,0,0,0,0,0,pad), 'constant', 0)
                            fi = F.pad(fi, (0,0,0,0,0,pad), 'constant', 0)
                            smap = F.pad(smap, (0,0,0,0,0,0,0,pad), 'constant', 0)
                            kmask = F.pad(kmask, (0,pad), 'constant', 0)
                    else:
                        pad = 0
                    num_temporal = (current_L+pad)//T
                    for i in range(num_temporal):
                        k_t.append((T*i,T*i+T))
            for _ in range(num_slice):
                p1 = self.num_sample
                self.num_sample = self.num_sample + num_temporal
                p2 = self.num_sample
                self.instance_tuple.append((p1, p2))
                self.dataset.append([num_temporal, k_t])
            if not cached:
                print(uk.shape)
                # cut slice
                for s in range(num_slice):
                    np.save(cache_path+"/"+file[:-4]+"_{}_{}_uk.npy".format(str(self.num_instance), str(s)), uk[s,...], allow_pickle=True)
                    np.save(cache_path+"/"+file[:-4]+"_{}_{}_fi.npy".format(str(self.num_instance), str(s)), fi[s,...], allow_pickle=True)
                    np.save(cache_path+"/"+file[:-4]+"_{}_{}_smap.npy".format(str(self.num_instance), str(s)), smap[s,...], allow_pickle=True)
                    np.save(cache_path+"/"+file[:-4]+"_{}_{}_mask.npy".format(str(self.num_instance), str(s)), kmask.permute(0,3,1,2), allow_pickle=True)
                if showpar:
                    files.set_description("Saving cached data %s" % file)
            else:
                if showpar:
                    files.set_description("Reading cached datapath %s" % file)
        if showpar:
            print("Total instance number: %s" % (str(len(files))))
            print("Total sampled number %d" % self.num_sample)
        self.csvpath = os.path.join(rootpath, pre_name)+'/'+name+str(T)+'.csv'
        if not os.path.exists(self.csvpath):
            self.datasetpath = [os.path.join(cache_path, file.name) for file in Path(cache_path).rglob("*_uk.npy")]
            self.datasetpath = natsorted(self.datasetpath)
            instance_index_tmp = []
            temporal_index_tmp1 = []
            temporal_index_tmp2 = []
            sampled_datasetpath = []
            for i in range(self.num_sample):
                instance_index, temporal_index = self.get_instance_sample_index(i)
                instance_index_tmp.append(instance_index)
                temporal_index_tmp1.append(temporal_index[0])
                temporal_index_tmp2.append(temporal_index[1])
                sampled_datasetpath.append(self.datasetpath[instance_index])
            self.sampled_datasetpath = sampled_datasetpath
            csv=pd.DataFrame({'path':self.sampled_datasetpath,
                              'instance_index': instance_index_tmp,
                              'temporal_index1':temporal_index_tmp1,
                              'temporal_index2':temporal_index_tmp2})
            csv.to_csv(self.csvpath)
        self.data_iter = pd.read_csv(self.csvpath, iterator=False, header=None, skiprows=1, nrows=self.num_sample)
            
    def get_instance_sample_index(self, index):
        for i in range(len(self.instance_tuple)):
            tuple = self.instance_tuple[i]
            if index >= tuple[0] and index < tuple[1]:
                instance_index = i
                relavate_sample_index = index - tuple[0]
                num_temporal, temporal_index_tuple = self.dataset[instance_index][0], self.dataset[instance_index][1]
                break
        return instance_index, temporal_index_tuple[relavate_sample_index]
    
    def __getitem__(self, index):
        # if index > 0:
        #     pre_instance_index = self.data_iter[2][index-1]
        # print(pre_instance_index)
        path = self.data_iter[1][index]
        # instance_index = self.data_iter[2][index]
        temporal_index1 = self.data_iter[3][index]
        temporal_index2 = self.data_iter[4][index]
        # if (index == 0) or (index > 0 and instance_index != pre_instance_index):
        uk = torch.tensor(np.load(path[:-7]+'_uk'+'.npy'))
        fi = torch.tensor(np.load(path[:-7]+'_fi'+'.npy'))
        smap = torch.tensor(np.load(path[:-7]+'_smap'+'.npy'))
        mask = torch.tensor(np.load(path[:-7]+'_mask'+'.npy'))
        sampled_data = [uk[:,temporal_index1:temporal_index2,:,:,:],#uk(c,kt,kx,ky,2)
                    fi[temporal_index1:temporal_index2,:,:],#fi(kt,kx,ky)
                    smap[:,temporal_index1:temporal_index2,:,:],#smap(c,kt,kx,ky)
                    mask[:,temporal_index1:temporal_index2,:,:]]#kmask(1,kt,kx,ky)
        return sampled_data
    
    def __len__(self):
        
        return self.num_sample

class OCMRDataset_iter(IterableDataset):
    def __init__(self, rootpath, pre_name, T=5, index=None, name='', showpar=True, overlap_mode=True, start=-1, end=-1):
        super(OCMRDataset_iter, self).__init__()
        self.dataset = []
        self.fdata_full = []
        self.datasetpath = []
        self.num_sample = 0
        self.num_instance = 0
        self.instance_tuple = []
        self.name = name
        cached = False
        path = Path(rootpath+"/npydataraw")
        files = [file.name for file in path.rglob("f*.npy")]
        files.sort()
        # index = None
        if index is not None:
            files = [files[i] for i in index]
        if showpar:
            files = tqdm(files)
        cache_path = os.path.join(rootpath, pre_name)+'/'+self.name
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        else:
            cached = True
        # print("cutslice:", cutslice)
        for file in files:
            # uk and fk refer to undersampled and fullsampled kspace data
            self.num_instance = self.num_instance + 1
            if not cached:
                uk = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/uk/"+file[:-4]+"_uk.npy")).squeeze(2)#(s,c,kz,kt,kx,ky,2)->(s,c,kt,kx,ky,2)
                fi = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/fi/"+file[:-4]+"_fi.npy")).squeeze(2)#(s,c,kz,kt,kx,ky)->(s,c,kt,kx,ky)
                smap = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/smap/"+file[:-4]+"_smap.npy"))#(s,c,t,kx,ky,2)
            kmask = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/mask/"+file[:-4]+"_mask.npy"))#(s,kx,ky,t)
            # Combine real part and image part
            if name == 'infer':
                self.fdata_full.append(fi.permute(0,3,1,2,4).unsqueeze(2))#(s,kx,ky,kt,2)->(s,kt,kx,ky,2)
            num_slice = kmask.shape[0]
            current_L = kmask.shape[3]
            if num_slice>1:
                kmask = kmask[:1, :, :, :]
            k_t = []
            if T > current_L:
                if not cached:
                    uk = F.pad(uk, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                    fi = F.pad(fi, (0,0,0,0,0,T-current_L), 'constant', 0)
                    smap = F.pad(smap, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                    kmask = F.pad(kmask, (0,T-current_L), 'constant', 0)
                num_temporal = 1
                k_t.append((0,T))
            elif current_L >= T:
                if overlap_mode:
                    num_temporal = current_L-T+1
                    for i in range(num_temporal):
                        k_t.append((i,T+i))
                else:
                    if current_L % T > 0:
                        pad = T - current_L % T
                        if not cached:
                            uk = F.pad(uk, (0,0,0,0,0,0,0,pad), 'constant', 0)
                            fi = F.pad(fi, (0,0,0,0,0,pad), 'constant', 0)
                            smap = F.pad(smap, (0,0,0,0,0,0,0,pad), 'constant', 0)
                            kmask = F.pad(kmask, (0,pad), 'constant', 0)
                    else:
                        pad = 0
                    num_temporal = (current_L+pad)//T
                    for i in range(num_temporal):
                        k_t.append((T*i,T*i+T))
            for _ in range(num_slice):
                p1 = self.num_sample
                self.num_sample = self.num_sample + num_temporal
                p2 = self.num_sample
                self.instance_tuple.append((p1, p2))
                self.dataset.append([num_temporal, k_t])
            if not cached:
                print(uk.shape)
                # cut slice
                for s in range(num_slice):
                    np.save(cache_path+"/cache_{}_{}_uk.npy".format(str(self.num_instance), str(s)), uk[s,...], allow_pickle=True)
                    np.save(cache_path+"/cache_{}_{}_fi.npy".format(str(self.num_instance), str(s)), fi[s,...], allow_pickle=True)
                    np.save(cache_path+"/cache_{}_{}_smap.npy".format(str(self.num_instance), str(s)), smap[s,...], allow_pickle=True)
                    np.save(cache_path+"/cache_{}_{}_mask.npy".format(str(self.num_instance), str(s)), kmask.permute(0,3,1,2), allow_pickle=True)
                if showpar:
                    files.set_description("Saving cached data %s" % file)
            else:
                if showpar:
                    files.set_description("Reading cached datapath %s" % file)
        if showpar:
            print("Total instance number: %s" % (str(len(files))))
            print("Total sampled number %d" % self.num_sample)
        self.csvpath = os.path.join(rootpath, pre_name)+'/'+name+str(T)+'.csv'
        if not os.path.exists(self.csvpath):
            self.datasetpath = [os.path.join(cache_path, file.name) for file in Path(cache_path).rglob("cache*_uk.npy")]
            self.datasetpath = natsorted(self.datasetpath)
            instance_index_tmp = []
            temporal_index_tmp1 = []
            temporal_index_tmp2 = []
            sampled_datasetpath = []
            for i in range(self.num_sample):
                instance_index, temporal_index = self.get_instance_sample_index(i)
                instance_index_tmp.append(instance_index)
                temporal_index_tmp1.append(temporal_index[0])
                temporal_index_tmp2.append(temporal_index[1])
                sampled_datasetpath.append(self.datasetpath[instance_index])
            self.sampled_datasetpath = sampled_datasetpath
            csv=pd.DataFrame({'path':self.sampled_datasetpath,
                              'instance_index': instance_index_tmp,
                              'temporal_index1':temporal_index_tmp1,
                              'temporal_index2':temporal_index_tmp2})
            csv.to_csv(self.csvpath)
        if start >=0 or end > 0:
            self.start = start
            self.end = end
        else:
            self.start = 0
            self.end = self.num_sample
            
    def get_instance_sample_index(self, index):
        for i in range(len(self.instance_tuple)):
            tuple = self.instance_tuple[i]
            if index >= tuple[0] and index < tuple[1]:
                instance_index = i
                relavate_sample_index = index - tuple[0]
                num_temporal, temporal_index_tuple = self.dataset[instance_index][0], self.dataset[instance_index][1]
                break
        return instance_index, temporal_index_tuple[relavate_sample_index]
    
    def __iter__(self):
        index = 0
        pre_instance_index = -1
        data_iter = pd.read_csv(self.csvpath, iterator=True, header=None, skiprows=1+self.start, nrows=self.end-self.start, chunksize=1)
        # print(pre_instance_index)
        for data in data_iter:
            path = data[1][index]
            instance_index = data[2][index]
            temporal_index1 = data[3][index]
            temporal_index2 = data[4][index]
            if (index == 0) or (index > 0 and instance_index != pre_instance_index):
                uk = torch.tensor(np.load(path[:-7]+'_uk'+'.npy'))
                fi = fastmri.rss(torch.tensor(np.load(path[:-7]+'_fi'+'.npy')), dim=0)
                for i in range(fi.shape[0]):
                    fi[i,:,:] = (fi[i,:,:]-fi[i,:,:].min())/(fi[i,:,:].max()-fi[i,:,:].min())
                smap = torch.tensor(np.load(path[:-7]+'_smap'+'.npy'))
                mask = torch.tensor(np.load(path[:-7]+'_mask'+'.npy'))
                pre_instance_index = instance_index
            index = index + 1
            sampled_dict = [uk[:,temporal_index1:temporal_index2,:,:,:],#uk(c,kt,kx,ky,2)
                        fi[temporal_index1:temporal_index2,:,:],#fi(kt,kx,ky)
                        smap[:,temporal_index1:temporal_index2,:,:,:],#smap(c,kt,kx,ky,2)
                        mask[:,temporal_index1:temporal_index2,:,:]]#kmask(1,kt,kx,ky)
            yield sampled_dict

class OCMRDataset_coilmerged(Dataset):
    def __init__(self, rootpath, pre_name, T=1, cutslice=True, index=None, name=''):
        super(OCMRDataset_coilmerged, self).__init__()
        self.dataset = []
        self.fdata_full = []
        self.num_sample = 0
        self.T = T
        self.T_S_num = []
        path = Path(rootpath)
        files = [file.name for file in path.rglob("f*.h5")]#[:1]
        files.sort()
        # index = None
        if index is not None:
            files = [files[i] for i in index]
        files = tqdm(files)
        print("Start reading ocmr %s, total instance number: %s" % (name, str(len(files))))
        print("cutslice:", cutslice)
        for file in files:
            # udata and fdata refer to undersampled and fullsampled image domain data
            udata = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/ui/"+file[:-3]+"_ui.npy")).squeeze(3)#(s,kx,ky,kz,kt)->(s,kx,ky,kt)
            fdata = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/fi/"+file[:-3]+"_fi.npy")).squeeze(3)#(s,kx,ky,kz,kt)->(s,kx,ky,kt)
            uk = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/uk/"+file[:-3]+"_uk.npy")).squeeze(1)#(s,kz,kt,kx,ky,2)->(s,kt,kx,ky,2)
            kmask = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/mask/"+file[:-3]+"_mask.npy")).unsqueeze(0)#(1,kx,ky,t)
            if name == 'infer':
                self.fdata_full.append(fdata.permute(0,3,1,2).unsqueeze(2))#(s,kx,ky,kt)->(s,kt,kx,ky)
            current_L = udata.shape[-1]
            
            # 1.Cutting temporal length
            udatalist_L = []
            fdatalist_L = []
            uk_L = []
            k_t = []
            if T > current_L:
                udata = F.pad(udata, (0,T-current_L), 'constant', 0)
                fdata = F.pad(fdata, (0,T-current_L), 'constant', 0)
                uk = F.pad(uk, (0,0,0,0,0,0,0,T-current_L), 'constant', 0)
                udatalist_L.append(udata)
                fdatalist_L.append(fdata)
                uk_L.append(uk)
                kmask = F.pad(kmask, (0,T-current_L), 'constant', 0)
                k_t.append((0,T))
            elif current_L >= T:
                for i in range(current_L-T+1):
                    k_t.append((i,T+i))
                    udatalist_L.append(udata[...,i:T+i])
                    fdatalist_L.append(fdata[...,i:T+i])
                    uk_L.append(uk[:,i:T+i,...])
            self.T_S_num.append((len(k_t), current_L, uk.shape[0]))#num_pad, t, S
            #2.Select temporal length
            for i in range(len(udatalist_L)):
                #3.Cut slice
                for s in range(udatalist_L[i].shape[0]):
                    u = udatalist_L[i][s,...].unsqueeze(0)#(h,w,T)->(C=1,h,w,T)
                    f = fdatalist_L[i][s,...].unsqueeze(0)
                    uki = uk_L[i][s,...]#(T,kx,ky)
                    self.num_sample = self.num_sample + 1
                    self.dataset.append([u.permute(3,0,1,2),f.permute(3,0,1,2), kmask.permute(3,0,1,2)[k_t[i][0]:k_t[i][1],...],uki])
                    # print(current_L, len(k_t), k_t[i][0], k_t[i][1])
            files.set_description("Reading preprocessed data %s" % file)
        print("Total sampled number %d" % self.num_sample)
        
    def __getitem__(self, index):  
          
        return self.dataset[index]
    
    def __len__(self):
            
        return len(self.dataset)

class FastmriDataset(Dataset):
    def __init__(self, rootpath, pre_name, name, infer=False, dram=True, useslice=True):
        super(FastmriDataset, self).__init__()
        self.dataset = []
        self.fdata_full = []
        if name == 'val':
            datainfo = np.load(os.path.join(rootpath, (name + "/datainfo.npy")))
            self.filenames = datainfo[0]
            self.slice_num = [int(i) for i in datainfo[1]]
        self.dram = dram
        pre_name = name + '/' + pre_name
        if dram==False and useslice==False:
            useslice = True
            print("Sliced data reading supported only, force useslice=True when not dram")
        # now get filepath
        # for self.dram, get filepath in /fi/
        # for not self.dram, get filepath in /fislice/
        if self.dram:
            if useslice:
                path = Path(os.path.join(rootpath, (pre_name+"/fislice/")))
            else:
                path = Path(os.path.join(rootpath, (pre_name+"/fi/")))
        else:
            path = Path(os.path.join(rootpath, (pre_name+"/fislice/")))
        files = natsorted([file.name for file in path.rglob("*.npy")])#[:128]
        files = tqdm(files)
        print("Start reading fastmri, toltal %s number: %s" % (name, str(len(files))))
        for f in files:
            if useslice:
                ukpath = os.path.join(rootpath, pre_name)+"/ukslice/" + f.replace('fi_', 'uk_')
                fipath = os.path.join(rootpath, pre_name)+"/fislice/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/maskslice/" + f.replace('fi_', 'mask_')
            else:
                ukpath = os.path.join(rootpath, pre_name)+"/uk/" + f.replace('_fi', '_uk')
                fipath = os.path.join(rootpath, pre_name)+"/fi/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/mask/" + f.replace('_fi', '_mask')
            if self.dram:
                fi = torch.tensor(np.load(fipath))#(kz,kx,ky,2);
                mask = torch.tensor(np.load(maskpath))#(1,1,kx,ky);
                uk = torch.tensor(np.load(ukpath))#(1/kz,kx,ky,2);
                # fk = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/fk/"+f[:-3]+"_fk.npy"))#(kz,kx,ky,2);
            else:
                ukpath = os.path.join(rootpath, pre_name)+"/ukslice/" + f.replace('fi_', 'uk_')
                fipath = os.path.join(rootpath, pre_name)+"/fislice/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/maskslice/" + f.replace('fi_', 'mask_')
            if infer:
                fi = torch.tensor(np.load(fipath))
                if useslice:
                    self.fdata_full.append(fi.unsqueeze(0))#(kx,ky)
                else:
                    self.fdata_full.append(fastmri.complex_abs(fi))#(z,kx,ky)
            if self.dram:
                if useslice:
                    # for i in range(uk.shape[1]):
                    #     uki = uk[:,i,...]
                    #     fii = fastmri.complex_abs(fi[i,...])
                    self.dataset.append([uk, fi, mask])
                else:
                    for i in range(uk.shape[0]):
                        uki = uk[i,...].unsqueeze(0)
                        fii = fastmri.complex_abs(fi[i,...])
                        self.dataset.append([uki, fii, mask])
            else:
                self.dataset.append([ukpath, fipath, maskpath])
            files.set_description("Reading processed data/datapath %s" % f)
            
    def __getitem__(self, index):
        if self.dram:
            uki, fii, mask = self.dataset[index]
            # uki = fft2c(torch.view_as_real(fii.unsqueeze(0)))*mask
        else:
            ukpath, fipath, maskpath = self.dataset[index]
            uki = torch.tensor(np.load(ukpath))
            fii = torch.tensor(np.load(fipath))
            mask = torch.tensor(np.load(maskpath))
        return uki, fii, mask
    
    def __len__(self):
        
        return len(self.dataset)
    
class FastmriDatasetforMambaIR(Dataset):
    def __init__(self, rootpath, pre_name, name, infer=False, dram=True, useslice=True):
        super(FastmriDatasetforMambaIR, self).__init__()
        self.dataset = []
        self.fdata_full = []
        if name == 'val':
            datainfo = np.load(os.path.join(rootpath, (name + "/datainfo.npy")))
            self.filenames = datainfo[0]
            self.slice_num = [int(i) for i in datainfo[1]]
        self.dram = dram
        pre_name = name + '/' + pre_name
        if dram==False and useslice==False:
            useslice = True
            print("Sliced data reading supported only, force useslice=True when not dram")
        # now get filepath
        # for self.dram, get filepath in /fi/
        # for not self.dram, get filepath in /fislice/
        if self.dram:
            if useslice:
                path = Path(os.path.join(rootpath, (pre_name+"/fislice/")))
            else:
                path = Path(os.path.join(rootpath, (pre_name+"/fi/")))
        else:
            path = Path(os.path.join(rootpath, (pre_name+"/fislice/")))
        files = natsorted([file.name for file in path.rglob("*.npy")])#[:8]
        files = tqdm(files)
        print("Start reading fastmri, toltal %s number: %s" % (name, str(len(files))))
        for f in files:
            if useslice:
                ukpath = os.path.join(rootpath, pre_name)+"/ukslice/" + f.replace('fi_', 'uk_')
                fipath = os.path.join(rootpath, pre_name)+"/fislice/" + f
                # maskpath = os.path.join(rootpath, pre_name)+"/maskslice/" + f.replace('fi_', 'mask_')
            else:
                ukpath = os.path.join(rootpath, pre_name)+"/uk/" + f.replace('_fi', '_uk')
                fipath = os.path.join(rootpath, pre_name)+"/fi/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/mask/" + f.replace('_fi', '_mask')
            if self.dram:
                fi = torch.tensor(np.load(fipath))#(kz,kx,ky,2);
                # mask = torch.tensor(np.load(maskpath))#(1,1,kx,ky);
                uk = torch.tensor(np.load(ukpath))#(1/kz,kx,ky,2);
                # fk = torch.tensor(np.load(os.path.join(rootpath, pre_name)+"/fk/"+f[:-3]+"_fk.npy"))#(kz,kx,ky,2);
            else:
                ukpath = os.path.join(rootpath, pre_name)+"/ukslice/" + f.replace('fi_', 'uk_')
                fipath = os.path.join(rootpath, pre_name)+"/fislice/" + f
                maskpath = os.path.join(rootpath, pre_name)+"/maskslice/" + f.replace('fi_', 'mask_')
            if infer:
                fi = torch.tensor(np.load(fipath))
                if useslice:
                    self.fdata_full.append(fi.unsqueeze(0))#(kx,ky)
                else:
                    self.fdata_full.append(fastmri.complex_abs(fi))#(z,kx,ky)
            if self.dram:
                if useslice:
                    # for i in range(uk.shape[1]):
                    #     uki = uk[:,i,...]
                    #     fii = fastmri.complex_abs(fi[i,...])
                    self.dataset.append([fastmri.complex_abs(ifft2c(uk)), fi.unsqueeze(0)])
                else:
                    for i in range(uk.shape[0]):
                        uki = uk[i,...].unsqueeze(0)
                        fii = fastmri.complex_abs(fi[i,...])
                        self.dataset.append([uki, fii])
            else:
                self.dataset.append([ukpath, fipath, maskpath])
            files.set_description("Reading processed data/datapath %s" % f)
            
    def __getitem__(self, index):
        if self.dram:
            uii, fii = self.dataset[index]
        else:
            ukpath, fipath, maskpath = self.dataset[index]
            uki = torch.tensor(np.load(ukpath))
            fii = torch.tensor(np.load(fipath))
        return uii, fii
    
    def __len__(self):
        
        return len(self.dataset)
    