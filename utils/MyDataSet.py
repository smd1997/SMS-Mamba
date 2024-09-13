from os import listdir, path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from skimage import data, exposure, img_as_float
import h5py
from scipy.io import loadmat
from scipy import ndimage

class MyDataSet(Dataset):
    def __init__(self, dataDir, labelDir, contourDir, coilDir, maskDir, ind_range=(0, 10000), mode="train"):   
        
        # 3张叠一起的混叠图
        self.dataDir = dataDir
        with h5py.File(str(self.dataDir), 'r') as h5_file:
                self.imagelabel = h5_file['data'][:]  # 混在一起的图像，[1730, 384, 384]
                self.imagelabel = np.expand_dims(self.imagelabel, axis=1).repeat(3, axis=1)  # 插值三倍，[1730, 3, 384, 384]
                self.imagelabel = self.imagelabel[ind_range[0]:ind_range[1]] # 取我们需要的范围，[1211, 3, 384, 384]
                # 将插值的结果合成, [3633, 384, 384]
                self.imagelabel = self.imagelabel.reshape((self.imagelabel.shape[0]*self.imagelabel.shape[1],384,384)) 
                # plt.imshow(self.imagelabel[0,:,:], cmap=plt.cm.gray);plt.savefig(r'/home/bmec-dl/MRI-Reconstruction-main/models/imagelabel.png')
                # import sigpy.plot as pl
                # import matplotlib.pyplot as plt
                

        # 标签
        self.labelDir = labelDir
        with h5py.File(str(self.labelDir), 'r') as label_file:
                self.label_contour = label_file['data'][:]  # 解耦的标签图像, [346, 5, 3, 384, 384], 5代表有5组
                # [1730, 3, 384, 384], 将前两维合并
                self.label_contour = self.label_contour.reshape((self.label_contour.shape[0]*self.label_contour.shape[1], 3, 384, 384)) 
                self.label_contour = self.label_contour[ind_range[0]:ind_range[1]] # [1211, 3, 384, 384]  # 1211是train的个数
                # [3633, 384, 384], 合并维度 1211*3
                self.label_contour = self.label_contour.reshape((self.label_contour.shape[0]*self.label_contour.shape[1],384,384))
        
        # 轮廓
        self.contourDir = contourDir
        with h5py.File(str(self.contourDir), 'r') as file:
                self.SMS_contour = file['contour'][:]  # [1730, 3, 384, 384]
                self.SMS_contour = self.SMS_contour[ind_range[0]:ind_range[1]]  # 用于分离混叠图像的轮廓, [1211, 3, 384, 384]
                # [3633, 384, 384]
                self.SMS_contour = self.SMS_contour.reshape((self.SMS_contour.shape[0]*self.SMS_contour.shape[1],384,384))
        
        # 线圈灵敏度图
        self.coilDir = coilDir
        with h5py.File(str(self.coilDir), 'r') as file:
                self.SMS_coil = file['coil'][:]  # 用于分离混叠图像的线圈灵敏度图， [5190, 384, 384] 
                # self.SMS_coil = self.SMS_coil.reshape(1730, 3, 384, 384)
                self.SMS_coil = self.SMS_coil[ind_range[0]*3:ind_range[1]*3]
                # self.SMS_coil = self.SMS_coil.reshape(self.SMS_coil.shape[0]*self.SMS_coil.shape[1], 384, 384)
                print('0')
        
        self.maskDir = maskDir
        self.mask = loadmat(self.maskDir)['x'].astype(float) #初始mask写死是不是也可以

    def __len__(self):
        return self.imagelabel.shape[0]  ##data is orgnized in CHW

    def __getitem__(self, i):

        image = self.imagelabel[i,:,:]
        # plt.imshow(np.abs(image), cmap='gray')
        # plt.savefig('/home/bmec-dl/MRI-Reconstruction-main/models/image.png')
        # plt.show() 

        label = self.label_contour[i,:,:]
        # plt.imshow(np.abs(label), cmap='gray')
        # plt.savefig('/home/bmec-dl/MRI-Reconstruction-main/models/label.png')
        # plt.show() 

        contour = self.SMS_contour[i,:,:]/255  # (3,384,384)
        # plt.imshow(np.abs(contour), cmap='gray')
        # plt.savefig('/home/bmec-dl/MRI-Reconstruction-main/models/contour.png')
        # plt.show()

        # structure = np.ones((7, 7))
        # # 执行腐蚀操作
        # contour = ndimage.binary_erosion(contour, structure=structure) #看一眼数值范围
        # # 执行膨胀操作
        # contour = ndimage.binary_dilation(contour, structure=structure)


        coil = self.SMS_coil[i,:,:]
        # plt.imshow(np.abs(coil), cmap='gray')
        # plt.savefig('/home/bmec-dl/MRI-Reconstruction-main/models/coil.png')
        # plt.show() 

        mask = self.mask
        # plt.imshow(np.abs(mask), cmap='gray')
        # plt.savefig('/home/bmec-dl/MRI-Reconstruction-main/models/mask.png')
        # plt.show() 

        SMS_kspace = np.fft.fftshift(np.fft.fft2(image))
        un_SMS_kspace = SMS_kspace * mask
        un_SMS = np.fft.ifft2(un_SMS_kspace)

        # 设置全白对照组
        # import pdb
        # pdb.set_trace()
        # white = np.full((384, 384), 1)
        # black = np.full((384, 384), 0)

        image = np.expand_dims(image, 0)
        contour = np.expand_dims(contour, 0)
        label = np.expand_dims(label, 0)
        coil = np.expand_dims(coil, 0)
        mask = np.expand_dims(mask, 0)                                                                                 
        un_SMS = np.expand_dims(un_SMS, 0)
        # black = np.expand_dims(black, 0)

        return {
           "input":torch.from_numpy(un_SMS),   #(1,384,384)  输入就直接放混叠+伪影的了，二合一的input
           "contour": torch.from_numpy(contour),   #(1,384,384)  contourt替换成un_SMS，就是同参数量下的无辅助重建;contourt替换成white
           "label": torch.from_numpy(label),   #(1,384,384)
           "coil": torch.from_numpy(coil),   #(1,384,384)
           "mask": torch.from_numpy(mask)
        }
