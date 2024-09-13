import os
import torch
import argparse
import numpy as np
import sys
sys.path.append(__file__.rsplit('/', 2)[0])
from utils.utils import SEED, get_time, adjust_learning_rate, print_to_log_file, collate_batch, worker_init_fn
from utils.utils import kwargs_vssm_unrolled as kwargs
from models.model_VSSM import VSSMUNet_unrolled
from models.MambaIR import buildMambaIR
from tqdm import tqdm
from utils.dataset import FastmriDatasetforMambaIR
from utils.Loss_fastmri import Loss, compute_metrics2c
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', type=int, default=50, required=False, help='number of epochs')
    parser.add_argument('-lr', type=float, default=4e-4, required=False, help='initial learning rate')
    parser.add_argument('-bs', type=int, default=1, required=False, help='batch size')
    parser.add_argument('-amp', type=str, default='fp16', required=False, help='fp16;fp32')
    parser.add_argument('-expname', type=str, default='', required=False)
    parser.add_argument('-eval_continue_model', type=str, default='model_latest', required=False, help='eval model name')
    parser.add_argument('-dataset_name', type=str, default='fastMRI', required=False, help='dataset name')
    parser.add_argument('-read_path', type=str, default='/mnt/data/dataset/fastMRI/knee_singlecoil', required=False, help='dataset path')
    parser.add_argument('-pre_name', type=str, default='preprocessed_knee_singlecoil_8x_0.04c320x320', required=False, help='preprossed folder name')
    parser.add_argument('-save_path', type=str, default='/mnt/data/zlt/AR-Recon/models/saved', required=False, help='model save path')
    parser.add_argument('--c', action='store_true', required=False, help='[OPTIONAL] continue training from latest checkpoint')
    parser.add_argument('-model_folder', type=str, default=None, required=False, help='[INTEGRAL IF --c] from which filefolder to continue')
    parser.add_argument('--ddp', action='store_true', required=False, help='[OPTIONAL] if ddp')
    args = parser.parse_args()
    # /mnt/sda1/dl
    # set default device and random pick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.ddp = True
    Fastmri_path = args.read_path
    Model_path = args.save_path
    load_done = False
    np.random.seed(SEED)
    # Checking paths
    Cur_model_path = os.path.join(Model_path, args.dataset_name)
    if not os.path.exists(Cur_model_path):
        os.makedirs(Cur_model_path)
    dataloader_params = {'batch_size': args.bs,
                    'shuffle':True,
                    'num_workers': 4,
                    'pin_memory': True, 
                    'drop_last' : True}
    dataloader_params_ = {'batch_size': 4,
                'shuffle':True,
                'num_workers': 4,
                'pin_memory': True, 
                'drop_last' : True}
    train_dataset = FastmriDatasetforMambaIR(rootpath=Fastmri_path, 
                pre_name=args.pre_name,
                name='train'
                # useslice=False
                )
    train_dataloader = DataLoader(dataset=train_dataset, **dataloader_params)
    test_dataset = FastmriDatasetforMambaIR(rootpath=Fastmri_path,
                    pre_name=args.pre_name,
                    name='val'
                    # useslice=False
                    )
    test_dataloader = DataLoader(dataset=test_dataset, **dataloader_params)
    torch.cuda.empty_cache()
    print("Creating model")
    model = buildMambaIR(upscale=1)
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
    # If continute training
    if args.c:
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
        print(start_ep)
        load_done = True
    else:
        start_ep = 0
        Cur_model_path = os.path.join(Cur_model_path, get_time()+args.pre_name+'_MambaIR' + args.expname)
        os.makedirs(Cur_model_path, exist_ok=True)
        config_file = os.path.join(Cur_model_path,"kwargs.txt")
        with open(config_file, 'w') as f:
            for arg in vars(args):
                f.write(arg+':'+str(getattr(args, arg))+'\n')
    assert start_ep < args.ep, 'countinue training={}, start_ep={}, max_ep={}'.format(args.c, start_ep, args.ep)
    ep = tqdm(range(start_ep, args.ep))
    log_file = os.path.join(Cur_model_path, get_time()+'.txt')
    # ===================train========================
    print_to_log_file(log_file, "Start training...") if not args.c else print_to_log_file(log_file, "Continue training...")
    best_psnr = 0
    for epoch in ep:
        torch.cuda.empty_cache()
        model.zero_grad()
        cur_lr = adjust_learning_rate(optimizer, epoch, args.ep, args.lr)
        model.train()
        p = 0
        s = 0
        loss_log = 0
        for batch_idx, [input, label] in enumerate(train_dataloader):
            # print(batch_idx)
            if scaler is not None:
                with autocast():
                    output = model(input.to(device_id))
                    optimizer.zero_grad()
                    l = loss(output, label.to(device_id))
                    loss_log = loss_log + l.data.item()
                scaler.scale(l).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(input.to(device_id))
                optimizer.zero_grad()
                l = loss(output, label.to(device_id))
                loss_log = loss_log + l.data.item()
                l.backward()
                optimizer.step()
            PSNR, SSIM = compute_metrics2c(output.squeeze(1), label.to(device_id).squeeze(1))
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
            for batch_idx, [input, label]in enumerate(test_dataloader):
                output = model(input.to(device_id))
                PSNR, SSIM = compute_metrics2c(output.squeeze(1), label.to(device_id).squeeze(1))
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
        
