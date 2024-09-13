import os
import torch
import argparse
import numpy as np
import sys
import fastmri
sys.path.append(__file__.rsplit('/', 2)[0])
from utils.utils import SEED, collate
from utils.utils import kwargs_unet_unrolled_LS as kwargs
from pathlib import Path
from models.model_UNet import UNet_unrolled_LS
from tqdm import tqdm
from utils.dataset import OCMRDataset
from utils.Loss_sense import compute_metrics2c_full
from torch.utils.data.dataloader import DataLoader

def rebuild(output_list, T_S_num, T):
    # output:(T,H,W,2)
    p = 0
    tmpt = []
    tmps = []
    out = []
    for i in range(len(T_S_num)):
        for t in range(T_S_num[i][0]):
            for _ in range(T_S_num[i][2]):
                if t > 0:
                    tmps.append(output_list[p][-1:,...])
                else:
                    tmps.append(output_list[p])
                p += 1
            tmpt.append(torch.stack(tmps, dim=0))# cat S (S,T,H,W,2)
            tmps = []
        if T_S_num[i][1] <= T:
            out.append(torch.cat(tmpt, dim=1)[:,:T_S_num[i][1],...])# cat T
        else:
            Tcat = []
            for t in tmpt:
                for ti in range(t.shape[1]):
                    Tcat.append(t[:,ti,...])
            out.append(torch.stack(Tcat, dim=1))
        tmpt = []
    return out

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=15, required=False, help='temporal size')
    parser.add_argument('-iter', type=int, default=6, required=False, help='iteration')
    parser.add_argument('-d_model', type=int, default=32, required=False, help='d_model')
    parser.add_argument('-eval_model', type=str, default='model_final', required=False, help='eval model name')
    parser.add_argument('-model_folder', type=str, default=None, required=False, help='from which filefolder to infer')
    parser.add_argument('-dataset_name', type=str, default='OCMR', required=False, help='dataset name')
    parser.add_argument('-data_path', type=str, default='/home/bmec/data/OCMR_data', required=False, help='dataset path')
    parser.add_argument('-pre_name', type=str, default='preprocessed_8x_0.04c_sense', required=False, help='preprossed folder name')
    parser.add_argument('-save_path', type=str, default='/mnt/data/zlt/AR-Recon/models/saved', required=False, help='model save path')
    parser.add_argument('-output_path', type=str, default='/mnt/data/zlt/AR-Recon/models/output', required=False, help='output path')
    args = parser.parse_args()
    # /mnt/sda1/dl
    # In this part we inference the whole raw data and evaluate the average PSNR and SSIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OCMR_path = args.data_path
    Model_path = args.save_path
    np.random.seed(SEED)
    path = Path(OCMR_path)
    files = [file.name for file in path.rglob("f*.h5")]
    file_index = [i for i in range(len(files))]
    train_index = np.random.choice(file_index, len(file_index)//5*4, replace=False).tolist()
    test_index = [i for i in file_index if i not in train_index]
    files.sort()
    train_index.sort()
    test_index.sort()
    # Checking paths
    args.eval_model = 'model_final'
    args.model_folder = '2024-06-16-20:26:49preprocessed_8x_0.04c_sense_unet_unrolled_LS'
    assert args.model_folder is not None, 'A specific model-folder must be given to infer! eg: 2020-01-01-00:00:00***'
    Cur_model_path = os.path.join(os.path.join(Model_path, args.dataset_name), args.model_folder)
    output_path = os.path.join(os.path.join(os.path.join(args.output_path, args.dataset_name), args.model_folder), args.eval_model)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # test_index = train_index[:5]
    infer_dataset = OCMRDataset(rootpath=OCMR_path,
                    T=args.T,
                    pre_name=args.pre_name,
                    cutslice=True,
                    index=test_index,
                    name='infer'
                    )
    infer_dataloader = DataLoader(dataset=infer_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=collate)
    torch.cuda.empty_cache()
    print("Creating model")
    if kwargs['UNet_base_num_features'] != args.d_model:
        kwargs['UNet_base_num_features'] = args.d_model
    if kwargs['input_channels'] != 2:
        kwargs['input_channels'] = 2
    for k,v in kwargs.items():
        print(k,v)
    model = UNet_unrolled_LS(iter=args.iter, kwargs=kwargs)
    model = model.to(device)
    model = model.to(device)
    print("Start evaluation...")
    checkpoint = torch.load(os.path.join(Cur_model_path, args.eval_model+'.pth'))['model_state_dict']
    name = next(iter(checkpoint))
    if name[:6] == 'module':
        new_state_dict = {}
        for k,v in checkpoint.items():
            new_state_dict[k[7:]] = v
        checkpoint = new_state_dict
    model.load_state_dict(checkpoint)
    model.eval()
    label_full = infer_dataset.fdata_full#(s,t,x,y,2)
    # smap_full = infer_dataset.smap_full#(s,c,t,x,y,2)
    T_S_num = infer_dataset.T_S_num
    output_list = []
    with torch.no_grad():
        label = []
        for batch_idx, datalist in enumerate(infer_dataloader):
            torch.cuda.empty_cache()
            for list in datalist:
                for i in range(len(list)):
                    list[i] = list[i].to(device)
                label.append(list[1].to('cpu'))
                del list[1]
            for out in model(datalist):
                output_list.append(out.to('cpu'))
        output_list_full = rebuild(output_list, T_S_num, infer_dataset.T)
        PSNR_list, SSIM_list = compute_metrics2c_full(output_list_full, label_full)
        result_file = os.path.join(output_path, "results.txt")
        with open(result_file, 'w') as f:
                f.write('This is only a reference.\n')
                for i in range(len(output_list_full)):
                    f.write("%s test_psnr: %.4f test_ssim: %.4f \n" % (files[test_index[i]], PSNR_list[i], SSIM_list[i]))
                    print("%s test_psnr: %.4f test_ssim: %.4f " % (files[test_index[i]], PSNR_list[i], SSIM_list[i]))
                f.write("mean_test_psnr: %.4f mean_test_ssim: %.4f \n" % (sum(PSNR_list)/len(PSNR_list), sum(SSIM_list)/len(SSIM_list)))
        print("mean_test_psnr: %.4f mean_test_ssim: %.4f " % (sum(PSNR_list)/len(PSNR_list), sum(SSIM_list)/len(SSIM_list)))
        print("Saving...")
        for i in range(len(output_list_full)):
                np.save(output_path+'/'+files[test_index[i]][:-3], output_list_full[i])
                np.save(output_path+'/'+files[test_index[i]][:-3]+'_fi', label_full[i])
                # np.save(output_path+'/'+files[test_index[i]][:-3], fastmri.rss_complex(output_list_full[i], dim=1))# RSS
        print("Inference done.")