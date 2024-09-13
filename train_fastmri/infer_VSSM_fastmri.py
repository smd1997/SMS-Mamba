import os
import torch
import argparse
import numpy as np
import sys
import fastmri
sys.path.append(__file__.rsplit('/', 2)[0])
from tqdm import tqdm
from utils.utils import kwargs_vssm_unrolled as kwargs
from models.model_VSSM import VSSMUNet_unrolled
from utils.dataset import FastmriDataset
from utils.Loss_fastmri import compute_metrics2c_full
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader, Subset
import logging
from utils.MyDataSet import MyDataSet
from utils.Loss_fastmri import Loss, compute_metrics2c
import scipy.io
from scipy import ndimage

# def rebuild(output_list, slice_num):
#     p = 0
#     out = []
#     for i in range(len(slice_num)):
#         tmps = []
#         for _ in range(slice_num[i]):
#             tmps.append(output_list[p])
#             p += 1
#         out.append(torch.cat(tmps))
#     return out

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', type=int, default=5, required=False, help='iteration')
    parser.add_argument('-d_model', type=int, default=64, required=False, help='d_model')
    parser.add_argument('-dctype', type=str, default='AM', required=False, help='VN;AM')
    parser.add_argument('-eval_model', type=str, default='model_final', required=False, help='eval model name')
    parser.add_argument('-model_folder', type=str, default=None, required=False, help='from which filefolder to infer')
    parser.add_argument('-dataset_name', type=str, default='fastMRI', required=False, help='dataset name')
    # parser.add_argument('-data_path', type=str, default='/mnt/data/dataset/fastMRI/knee_singlecoil', required=False, help='dataset path')
    parser.add_argument('-pre_name', type=str, default='SMS_4x_384x384', required=False, help='preprossed folder name')
    # parser.add_argument('-pre_name', type=str, default='preprocessed_knee_singlecoil_8x_0.04c320x320', required=False, help='preprossed folder name')
    parser.add_argument('-save_path', type=str, default='/home/bmec-dl/MRI-Reconstruction-main/models/saved', required=False, help='model save path')
    parser.add_argument('-output_path', type=str, default='/home/bmec-dl/MRI-Reconstruction-main/models/output', required=False, help='output path')
    parser.add_argument('--useslice', action='store_true', default=True, required=False, help='[OPTIONAL] if useslice')
    parser.add_argument('--dorefine', action='store_true', default=False, required=False, help='[OPTIONAL] if use refined output')
    #
    parser.add_argument('-dataDir', type=str, default='/mnt/sda1/dl/Data/fast_SMS/SMS3_matrix_all.h5', required=False, help='dataset path')
    parser.add_argument('-contourDir', type=str, default='/mnt/sda1/dl/Data/fast_SMS/SMS_contour_train_all.h5', required=False, help='dataset path')
    parser.add_argument('-coilDir', type=str, default='/mnt/sda1/dl/Data/fast_SMS/merge_SMS_coil_all.h5', required=False, help='dataset path')
    parser.add_argument('-labelDir', type=str, default='/mnt/sda1/dl/Data/fast_SMS/SMS_contour_label_all.h5', required=False, help='dataset path')
    parser.add_argument('-maskDir', type=str, default='/mnt/sda1/dl/Mask/384mask.mat', required=False, help='dataset path')
    
    args = parser.parse_args()
    # /mnt/sda1/dl
    # In this part we inference the whole raw data and evaluate the average PSNR and SSIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fastmri_path = args.data_path
    Model_path = args.save_path
    # Checking paths
    args.useslice = True
    args.dorefine = True
    # 2024-08-28-11:09:51SMS_4x_384x384_VSSM_unrolled_AM 
    # /home/bmec-dl/MRI-Reconstruction-main/models/saved/fastMRI/2024-09-10-11:10:49SMS_4x_384x384_VSSM_unrolled_AM
    args.eval_model = 'model602024-08-28-15:40:55'    #model602024-08-28-15:40:55
    args.model_folder = '2024-08-28-11:09:51SMS_4x_384x384_VSSM_unrolled_AM'  #2024-08-28-11:09:51SMS_4x_384x384_VSSM_unrolled_AM
    assert args.model_folder is not None, 'A specific model-folder must be given to infer! eg: 2020-01-01-00:00:00***'
    Cur_model_path = os.path.join(os.path.join(Model_path, args.dataset_name), args.model_folder)
    if args.dorefine:
        expname = '_refined'
    else:
        expname = ''
    output_path = os.path.join(os.path.join(os.path.join(args.output_path, args.dataset_name), args.model_folder), args.eval_model) + expname
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)


    # infer_dataset = FastmriDataset(rootpath=Fastmri_path, 
    #             pre_name=args.pre_name,
    #             name='val',
    #             useslice=args.useslice,
    #             infer=True,
    #             )
    # infer_dataloader = DataLoader(dataset=infer_dataset, 
    #                         batch_size=1, 
    #                         shuffle=False, 
    #                         num_workers=0,
    #                         pin_memory=True)
    
    ###
    logging.info('loading train dataset......')
    train_dataset = MyDataSet(args.dataDir, args.labelDir, args.contourDir, args.coilDir, args.maskDir, ind_range=(0, 1211), mode="train")
    logging.info('loading val dataset......')
    test_dataset_all = MyDataSet(args.dataDir, args.labelDir, args.contourDir, args.coilDir, args.maskDir, ind_range=(1211, 1730), mode="val")
    infer_dataset = Subset(test_dataset_all, range(1211-1211, 1557-1211))
    test_dataset = Subset(test_dataset_all, range(1557-1211, 1730-1211))

    # train_dataloader = DataLoader(train_dataset, batch_size = args.bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    infer_dataloader = DataLoader(infer_dataset, batch_size = 1, shuffle = False, num_workers= 4, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers= 4, pin_memory=True, drop_last=True)
    ###
    torch.cuda.empty_cache()
    print("Creating model")
    if kwargs['d_model'] != args.d_model:   
        kwargs['d_model'] = args.d_model
        kwargs['UNet_base_num_features'] = args.d_model
    if kwargs['input_channels'] != 2:
        kwargs['input_channels'] = 1   # 原来是 =2  轮廓和图concat变成2
    for k,v in kwargs.items():
        print(k,v)
    model = VSSMUNet_unrolled(iter=args.iter, DC_type=args.dctype, kwargs=kwargs)
    model = model.to(device)
    print("Start evaluation...")
    checkpoint = torch.load(os.path.join(Cur_model_path, args.eval_model+'.pth'), map_location='cuda:0')['model_state_dict']
    name = next(iter(checkpoint))
    if name[:6] == 'module':
        new_state_dict = {}
        for k,v in checkpoint.items():
            new_state_dict[k[7:]] = v
        checkpoint = new_state_dict
    model.load_state_dict(checkpoint)
    model.eval()

    # 对推理的数据集进行细分
    # label_full = infer_dataset.fdata_full
    # slice_num = infer_dataset.slice_num
    # files = infer_dataset.filenames
    output_list = []
    p = 0
    s = 0
    with torch.no_grad():
        with tqdm(total=len(infer_dataloader)) as pbar:
            for batch_idx, batch in enumerate(infer_dataloader):
                torch.cuda.empty_cache()

                input = batch['input'].to(device = device, dtype = torch.float32)
                contour = batch['contour'].to(device = device, dtype = torch.float32)
                label = batch['label'].to(device = device, dtype = torch.float32)
                coil = batch['coil'].to(device = device, dtype = torch.float32)
                mask = batch['mask'].to(device = device, dtype = torch.float32)


                output = model(input, contour, mask , args.dorefine)
                # PSNR, SSIM = compute_metrics2c(output, label)
                # p = p + PSNR
                # s = s + SSIM
                # f.write("%s test_psnr: %.4f test_ssim: %.4f \n" % (files[i], PSNR_list[i], SSIM_list[i]))
                output_list.append(output.to('cpu'))
                pbar.update(1)
    result = torch.stack(output_list)

    save_path = '/home/bmec-dl/MRI-Reconstruction-main/train_fastmri/result1.mat'
    # 检查路径是否存在，如果不存在则创建
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 使用scipy.io.savemat将矩阵保存为.mat文件
    scipy.io.savemat(save_path, {'x': result})
    print(f'Matrix saved to {save_path}')


    # output_list_full = rebuild(output_list, slice_num)
    # if args.useslice:
    #     label_full = rebuild(label_full, slice_num)
    # # # PSNR_list, SSIM_list = compute_metrics2c_full(output_list_full, label_full)
    # result_file = os.path.join(output_path, "results.txt")
    # with open(result_file, 'w') as f:
    #         f.write('This is only a reference.\n')
    #         for i in range(len(output_list_full)):
    #             f.write("%s test_psnr: %.4f test_ssim: %.4f \n" % (files[i], PSNR_list[i], SSIM_list[i]))
    #             print("%s test_psnr: %.4f test_ssim: %.4f " % (files[i], PSNR_list[i], SSIM_list[i]))
    #         f.write("mean_test_psnr: %.4f mean_test_ssim: %.4f \n" % (sum(PSNR_list)/len(PSNR_list), sum(SSIM_list)/len(SSIM_list)))
    # print("mean_test_psnr: %.4f mean_test_ssim: %.4f " % (sum(PSNR_list)/len(PSNR_list), sum(SSIM_list)/len(SSIM_list)))
    # print("Saving...")
    # for i in range(len(output_list_full)):
    #         np.save(output_path+'/'+files[i][:-3], output_list_full[i])
    #         # np.save(output_path+'/'+files[i]+'_fi', label_full[i])
    # print("Inference done.")