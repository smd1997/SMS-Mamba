import os
import torch
import argparse
import numpy as np
import sys
sys.path.append(__file__.rsplit('/', 2)[0])
from utils.utils import SEED, get_time, adjust_learning_rate, print_to_log_file
from utils.utils import kwargs_vssm_unrolled as kwargs
from models.model_VSSM import VSSMUNet_unrolled
from tqdm import tqdm
# from utils.dataset import FastmriDataset  # 不适用于SMS
from utils.Loss_fastmri import Loss, compute_metrics2c
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
import logging
from utils.MyDataSet import MyDataSet

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', type=int, default=60, required=False, help='number of epochs')
    parser.add_argument('-lr', type=float, default=4e-4, required=False, help='initial learning rate')
    parser.add_argument('-bs', type=int, default=2, required=False, help='batch size')
    parser.add_argument('-iter', type=int, default=5, required=False, help='iteration')
    parser.add_argument('-d_model', type=int, default=64, required=False, help='d_model')
    parser.add_argument('-amp', type=str, default='fp32', required=False, help='fp16;fp32')
    parser.add_argument('-dctype', type=str, default='AM', required=False, help='VN;AM')
    parser.add_argument('-expname', type=str, default='', required=False)
    parser.add_argument('-eval_continue_model', type=str, default='model_latest', required=False, help='eval model name')
    parser.add_argument('-dataset_name', type=str, default='fastMRI', required=False, help='dataset name')
    parser.add_argument('-read_path', type=str, default='/mnt/data/dataset/fastMRI/knee_singlecoil', required=False, help='dataset path')
    parser.add_argument('-pre_name', type=str, default='SMS_4x_384x384', required=False, help='preprossed folder name')
    parser.add_argument('-save_path', type=str, default='/home/bmec-dl/MRI-Reconstruction-main/models/saved', required=False, help='model save path')
    parser.add_argument('--dcloss', action='store_true', required=False, help='[OPTIONAL] use dc loss')
    parser.add_argument('--c', action='store_true', required=False, help='[OPTIONAL] continue training from latest checkpoint')
    parser.add_argument('-model_folder', type=str, default=None, required=False, help='[INTEGRAL IF --c] from which filefolder to continue')
    parser.add_argument('--ddp', action='store_true', required=False, help='[OPTIONAL] if ddp')
    parser.add_argument('--eval', action='store_true', default=False, required=False, help='[OPTIONAL] if eval')
    ##/mnt/sda1/dl/Data/fast_SMS
    parser.add_argument('-dataDir', type=str, default='/mnt/sda1/dl/Data/fast_SMS/SMS3_matrix_all.h5', required=False, help='dataset path')
    parser.add_argument('-contourDir', type=str, default='/mnt/sda1/dl/Data/fast_SMS/SMS_contour_train_all.h5', required=False, help='dataset path')
    parser.add_argument('-coilDir', type=str, default='/mnt/sda1/dl/Data/fast_SMS/merge_SMS_coil_all.h5', required=False, help='dataset path')
    parser.add_argument('-labelDir', type=str, default='/mnt/sda1/dl/Data/fast_SMS/SMS_contour_label_all.h5', required=False, help='dataset path')
    parser.add_argument('-maskDir', type=str, default='/mnt/sda1/dl/Mask/384mask_0.2.mat', required=False, help='dataset path')
    args = parser.parse_args()
    # /mnt/sda1/dl
    # set default device and random pick
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    Fastmri_path = args.read_path
    Model_path = args.save_path
    load_done = False
    np.random.seed(SEED)
    # Checking paths
    Cur_model_path = os.path.join(Model_path, args.dataset_name)
    if not os.path.exists(Cur_model_path):
        os.makedirs(Cur_model_path)
    # dataloader_params = {'batch_size': args.bs,
    #                 'shuffle':True,
    #                 'num_workers': 4,
    #                 'pin_memory': True, 
    #                 'drop_last' : True}
    dataloader_params_ = {'batch_size': 2,
            'shuffle':True,
            'num_workers': 4,
            'pin_memory': True, 
            'drop_last' : True}
   # 基于dataset的数据预处理
    # if not args.eval:
    #     train_dataset = FastmriDataset(rootpath=Fastmri_path, 
    #                 pre_name=args.pre_name,
    #                 name='train'
    #                 # useslice=False
    #                 )
    #     train_dataloader = DataLoader(dataset=train_dataset, **dataloader_params)
    # test_dataset = FastmriDataset(rootpath=Fastmri_path,
    #                 pre_name=args.pre_name,
    #                 name='val'
    #                 # useslice=False
    #                 )
    # test_dataloader = DataLoader(dataset=test_dataset, **dataloader_params)

    ##
    logging.info('loading train dataset......')
    train_dataset = MyDataSet(args.dataDir, args.labelDir, args.contourDir, args.coilDir, args.maskDir, ind_range=(0, 1211), mode="train")
    logging.info('loading val dataset......')
    test_dataset_all = MyDataSet(args.dataDir, args.labelDir, args.contourDir, args.coilDir, args.maskDir, ind_range=(1211, 1730), mode="val")
    val_dataset = Subset(test_dataset_all, range(1211-1211, 1557-1211))
    test_dataset = Subset(test_dataset_all, range(1557-1211, 1730-1211))

    train_dataloader = DataLoader(train_dataset, batch_size = args.bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers= 4, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers= 4, pin_memory=True, drop_last=True)
    ##
    torch.cuda.empty_cache()
    print("Creating model")
    if kwargs['d_model'] != args.d_model:   
        kwargs['d_model'] = args.d_model
        kwargs['UNet_base_num_features'] = args.d_model
    if kwargs['input_channels'] != 2:   # SMS我预处理保存成单通道图像域了
        kwargs['input_channels'] = 1  # 原来是 =2  轮廓和图concat变成2
    model = VSSMUNet_unrolled(iter=args.iter, DC_type=args.dctype, kwargs=kwargs)
    # to device
    print(f"Total num_gpu = {torch.cuda.device_count()}, DDP = {args.ddp}")
    if args.ddp and torch.cuda.device_count()>=2:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
        device_id = rank % torch.cuda.device_count()
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id],find_unused_parameters=True)
    else:
        device_id = device
        model = model.to(device_id)
        rank = 0
    loss = Loss().to(device_id)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    
    # It is recommended to use amp, 
    # but should be careful about overflow
    if args.amp == 'fp16':
        scaler = GradScaler()
    elif args.amp == 'fp32':
        scaler = None
    # args.model_folder = "2024-07-25-19:38:18preprocessed_knee_singlecoil_8x_0.04c320x320_VSSM_unrolled_AMiterAM6x8"
    # args.eval_continue_model = "model_latest"
    # args.c = True
    # If continute training
    if args.c or args.eval:
        Cur_model_path = os.path.join(Cur_model_path, args.model_folder)
        assert args.model_folder is not None, 'A specific model-folder must be given to continue or evaluate! eg: 2020-01-01-00:00:00'
        checkpoint = torch.load(os.path.join(Cur_model_path, args.eval_continue_model+'.pth'))
        name = next(iter(checkpoint['model_state_dict']))
        if name[:6] == 'module' and not args.ddp:
            new_state_dict = {}
            for k,v in checkpoint['model_state_dict'].items():
                new_state_dict[k[7:]] = v
            checkpoint['model_state_dict'] = new_state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_ep = checkpoint['epoch'] + 1
        load_done = True
    else:
        start_ep = 0
        Cur_model_path = os.path.join(Cur_model_path, get_time()+args.pre_name+'_VSSM_unrolled_'+args.dctype + args.expname)
        os.makedirs(Cur_model_path, exist_ok=True)
        config_file = os.path.join(Cur_model_path,"kwargs.txt")
        with open(config_file, 'w') as f:
            for arg in vars(args):
                f.write(arg+':'+str(getattr(args, arg))+'\n')
            f.write('{\n')
            for k,v in kwargs.items():
                f.write(str(k))
                f.write(':')
                f.write(str(v))
                f.write(',\n')
            f.write('}\n')
    assert start_ep < args.ep, 'countinue training={}, start_ep={}, max_ep={}'.format(args.c, start_ep, args.ep)
    ep = tqdm(range(start_ep, args.ep))
    log_file = os.path.join(Cur_model_path, get_time()+'.txt')
    if rank == 0:
        for k,v in kwargs.items():
            print(k,v)
    if args.eval:
        # notice that validate is on croppped dataset
        # if need to validate on original data, go for the inference
        # ===================offline validate========================
        print("Start evaluation...")
        assert load_done, 'Please follow the steps of continue training first'
        model.eval()
        p = 0
        s = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                
                input = batch['input'].to(device = device_id, dtype = torch.float32)
                contour = batch['contour'].to(device = device_id, dtype = torch.float32)
                label = batch['label'].to(device = device_id, dtype = torch.float32)
                coil = batch['coil'].to(device = device_id, dtype = torch.float32)
                mask = batch['mask'].to(device = device_id, dtype = torch.float32)
                
                # output = model(input.to(device_id), mask.to(device_id))
                # output = model(input, contour, mask)
                output = model(input, contour, mask)
                PSNR, SSIM = compute_metrics2c(output, label)
                p = p + PSNR
                s = s + SSIM
            print("test_psnr: %.4f test_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
        print("Evaluation done.")
    else:
        # ===================train========================
        print_to_log_file(log_file, "Start training...") if not args.c else print_to_log_file(log_file, "Continue training...")
        best_psnr = 0
        dorefine = False
        for epoch in ep:
            torch.cuda.empty_cache()
            model.zero_grad()
            cur_lr = adjust_learning_rate(optimizer, epoch, args.ep, args.lr)
            model.train()
            # 底下这个if没有执行
            if epoch >= args.ep - 10:
                print_to_log_file(log_file, 'do refining')
                cur_lr = adjust_learning_rate(optimizer, epoch - args.ep + 10, 10, args.lr)
                if not dorefine:
                    dorefine = True
                    train_dataloader = DataLoader(dataset=train_dataset, **dataloader_params_)
                    test_dataloader = DataLoader(dataset=test_dataset, **dataloader_params_)
            p = 0
            s = 0
            loss_log = 0 
            for batch_idx, batch in enumerate(train_dataloader):  # [input, label, mask]，mask传进来是为了日后的DC，现在是纯图像域的学习
                # print(batch_idx)
                input = batch['input'].to(device = device_id, dtype = torch.float32)
                contour = batch['contour'].to(device = device_id, dtype = torch.float32)
                label = batch['label'].to(device = device_id, dtype = torch.float32)
                coil = batch['coil'].to(device = device_id, dtype = torch.float32)
                mask = batch['mask'].to(device = device_id, dtype = torch.float32)
                
                if scaler is not None:
                    with autocast():
                        output = model(input.to(device_id), mask.to(device_id), dorefine)
                        optimizer.zero_grad()
                        l = loss(output, label.to(device_id))
                        loss_log = loss_log + l.data.item()
                    scaler.scale(l).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # output = model(input, contour, mask, dorefine)
                    output = model(input, contour, mask, dorefine)
                    optimizer.zero_grad()
                    l = loss(output, label)
                    loss_log = loss_log + l.data.item()
                    l.backward()
                    optimizer.step()
                PSNR, SSIM = compute_metrics2c(output, label)
                if rank == 0:
                    print('['+str(batch_idx)+'/'+str(len(train_dataloader))+']'+' '+str(PSNR)+' '+str(SSIM) +' '+str(l.data.item()))
                p = p + PSNR
                s = s + SSIM
            if rank == 0:
                print_to_log_file(log_file, "%s Epoch %d lr %.6f train_loss: %.12f " % (get_time(), epoch, cur_lr,  float(loss_log/(batch_idx+1))))
                print_to_log_file(log_file, "train_psnr: %.4f train_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
            if (epoch + 1) % 10 == 0:
                torch.save({
                            'kwargs': kwargs,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(Cur_model_path, 'model{}{}.pth'.\
                            format(epoch+1, get_time())))
            elif (epoch + 1) == args.ep:
                torch.save({
                            'kwargs': kwargs,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(Cur_model_path, 'model_final.pth'))
            else:
                torch.save({
                'kwargs': kwargs,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(Cur_model_path, 'model_latest.pth'))                 
            # ===================online validate========================
            model.eval()
            p = 0
            s = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    input = batch['input'].to(device = device_id, dtype = torch.float32)
                    contour = batch['contour'].to(device = device_id, dtype = torch.float32)
                    label = batch['label'].to(device = device_id, dtype = torch.float32)
                    coil = batch['coil'].to(device = device_id, dtype = torch.float32)
                    mask = batch['mask'].to(device = device_id, dtype = torch.float32)
                    
                    # output = model(input, contour, mask, dorefine)
                    output = model(input, contour, mask, dorefine)
                    # output = model(input.to(device_id), mask.to(device_id), dorefine)
                    PSNR, SSIM = compute_metrics2c(output, label)
                    p = p + PSNR
                    s = s + SSIM
                if rank == 0:
                    print_to_log_file(log_file, "test_psnr: %.4f test_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
            if float(p/(batch_idx+1)) > best_psnr:
                best_psnr = float(p/(batch_idx+1))
                torch.save({
                'kwargs': kwargs,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(Cur_model_path, 'model_best_psnr.pth'))     
            ep.set_description("Trained epoch %s" % epoch)
        print_to_log_file(log_file, "Training done.")
        
    
