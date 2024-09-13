import os
import torch
import argparse
import numpy as np
import sys
sys.path.append(__file__.rsplit('/', 2)[0])
from utils.utils import SEED, get_time, adjust_learning_rate, print_to_log_file, collate, worker_init_fn
from utils.utils import kwargs_vssm_unrolled_LS as kwargs
from pathlib import Path
from models.model_VSSM import VSSMUNet_unrolled_LS
from tqdm import tqdm
from utils.dataset import OCMRDataset, OCMRDatasetv2
from utils.Loss_sense import Loss, compute_metrics2c
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', type=int, default=100, required=False, help='number of epochs')
    parser.add_argument('-lr', type=float, default=1e-3, required=False, help='initial learning rate')
    parser.add_argument('-bs', type=int, default=1, required=False, help='batch size')
    parser.add_argument('-iter', type=int, default=8, required=False, help='iteration')
    parser.add_argument('-T', type=int, default=15, required=False, help='temporal size')
    parser.add_argument('-d_model', type=int, default=48, required=False, help='d_model')
    parser.add_argument('-amp', type=str, default='fp32', required=False, help='fp16;fp32')
    parser.add_argument('-dctype', type=str, default='VN', required=False, help='VN;AM')
    parser.add_argument('-eval_continue_model', type=str, default='model_final', required=False, help='eval model name')
    parser.add_argument('-dataset_name', type=str, default='OCMR', required=False, help='dataset name')
    parser.add_argument('-read_path', type=str, default='/home/bmec/data/OCMR_data', required=False, help='dataset path')
    parser.add_argument('-pre_name', type=str, default='preprocessed_128x128_8x_0.04c_sense', required=False, help='preprossed folder name')
    parser.add_argument('-save_path', type=str, default='/mnt/data/zlt/AR-Recon/models/saved', required=False, help='model save path')
    parser.add_argument('--dcloss', action='store_true', required=False, help='[OPTIONAL] use dc loss')
    parser.add_argument('--c', action='store_true', required=False, help='[OPTIONAL] continue training from latest checkpoint')
    parser.add_argument('-model_folder', type=str, default=None, required=False, help='[INTEGRAL IF --c] from which filefolder to continue')
    parser.add_argument('--ddp', action='store_true', required=False, help='[OPTIONAL] if ddp')
    parser.add_argument('--eval', action='store_true', default=False, required=False, help='[OPTIONAL] if eval')
    args = parser.parse_args()
    # /mnt/sda1/dl
    # set default device and random pick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OCMR_path = args.read_path
    Model_path = args.save_path
    load_done = False
    np.random.seed(SEED)
    path = Path(OCMR_path+'/npydataraw')
    files = [file.name for file in path.rglob("f*.npy")]
    file_index = [i for i in range(len(files))]
    train_index = np.random.choice(file_index, len(file_index)//5*4, replace=False).tolist()
    test_index = [i for i in file_index if i not in train_index]
    files.sort()
    train_index.sort()
    test_index.sort()
    # Checking paths
    Cur_model_path = os.path.join(Model_path, args.dataset_name)
    if not os.path.exists(Cur_model_path):
        os.makedirs(Cur_model_path)
    dataloader_params = {'batch_size': args.bs,
                    'shuffle':True,
                    'num_workers': 0,
                    'pin_memory': True, 
                    'drop_last' : True,
                    'collate_fn':collate}
    if not args.eval:
        train_dataset = OCMRDataset(rootpath=OCMR_path, 
                    pre_name=args.pre_name,
                    T=args.T,
                    index=train_index,
                    name='train'
                    )
        train_dataloader = DataLoader(dataset=train_dataset, **dataloader_params)
    test_dataset = OCMRDataset(rootpath=OCMR_path,
                    pre_name=args.pre_name, 
                    T=args.T,
                    index=test_index,
                    name='test'
                    )
    test_dataloader = DataLoader(dataset=test_dataset, **dataloader_params)
    torch.cuda.empty_cache()
    print("Creating model")
    if kwargs['d_model'] != args.d_model:   
        kwargs['d_model'] = args.d_model
        kwargs['UNet_base_num_features'] = args.d_model
    if kwargs['input_channels'] != 2:
        kwargs['input_channels'] = 2
    model = VSSMUNet_unrolled_LS(iter=args.iter, DC_type=args.dctype, kwargs=kwargs)
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
    loss = Loss().to(device_id)
    optimizer = torch.optim.AdamW(model.parameters())
    
    # It is recommended to use amp, 
    # but should be careful about overflow
    if args.amp == 'fp16':
        scaler = GradScaler()
    elif args.amp == 'fp32':
        scaler = None
    # args.model_folder = "2024-06-05-02:25:51preprocessed_192x192_8x_0.04c_sense_VSSM_unrolled_LS"
    # args.eval_continue_model = "model_best_psnr"
    # args.c = True
    # If continute training
    if args.c or args.eval:
        Cur_model_path = os.path.join(Cur_model_path, args.model_folder)
        assert args.model_folder is not None, 'A specific model-folder must be given to continue or evaluate! eg: 2020-01-01-00:00:00'
        checkpoint = torch.load(os.path.join(Cur_model_path, args.eval_continue_model+'.pth'))
        name = next(iter(checkpoint))
        if name[:6] == 'module' and not args.ddp:
            new_state_dict = {}
            for k,v in checkpoint.items():
                new_state_dict[k[7:]] = v
            checkpoint = new_state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_ep = checkpoint['epoch'] + 1
        load_done = True
    else:
        start_ep = 0
        Cur_model_path = os.path.join(Cur_model_path, get_time()+args.pre_name+'_VSSM_unrolled_LS_'+args.dctype)
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
            for batch_idx, datalist in enumerate(test_dataloader):
                label = []
                for list in datalist:
                    for i in range(len(list)):
                        list[i] = list[i].to(device_id)
                    label.append(list[1])
                    del list[1]
                output = model(datalist)
                PSNR, SSIM = compute_metrics2c(output, label, device_id)
                p = p + PSNR
                s = s + SSIM
            print("test_psnr: %.4f test_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
        print("Evaluation done.")
    else:
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
            for batch_idx, datalist in enumerate(train_dataloader):
                # print(batch_idx)
                labeli = []
                for list in datalist:
                    for i in range(len(list)):
                        list[i] = list[i].to(device_id)
                    labeli.append(list[1])
                    del list[1]
                if scaler is not None:
                    with autocast():
                        output = model(datalist)
                        optimizer.zero_grad()
                        l = loss(output, labeli)
                        loss_log = loss_log + l.data.item()
                    scaler.scale(l).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(datalist)
                    optimizer.zero_grad()
                    l = loss(output, labeli)
                    loss_log = loss_log + l.data.item()
                    l.backward()
                    optimizer.step()
                PSNR, SSIM = compute_metrics2c(output, labeli, device_id)
                p = p + PSNR
                s = s + SSIM
            print_to_log_file(log_file, "%s Epoch %d lr %.6f train_loss: %.6f " % (get_time(), epoch, cur_lr,  float(loss_log/(batch_idx+1))))
            print_to_log_file(log_file, "train_psnr: %.4f train_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
            if (epoch + 1) % 10 == 0:
                torch.save({
                            'kwargs': kwargs,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(Cur_model_path, 'model{}{}.pth'.\
                            format(epoch+1, get_time())))
                torch.save({
                            'kwargs': kwargs,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(Cur_model_path, 'model_latest.pth'))
            if (epoch + 1) == args.ep:
                torch.save({
                            'kwargs': kwargs,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(Cur_model_path, 'model_final.pth'))                    
            # ===================online validate========================
            model.eval()
            p = 0
            s = 0
            with torch.no_grad():
                for batch_idx, datalist in enumerate(test_dataloader):
                    labeli = []
                    for list in datalist:
                        for i in range(len(list)):
                            list[i] = list[i].to(device_id)
                        labeli.append(list[1])
                        del list[1]
                    output = model(datalist)
                    PSNR, SSIM = compute_metrics2c(output, labeli, device_id)
                    p = p + PSNR
                    s = s + SSIM
                print_to_log_file(log_file, "test_psnr: %.4f test_ssim: %.4f " % (float(p/(batch_idx+1)), float(s/(batch_idx+1))))
            if float(p/(batch_idx+1)) > best_psnr:
                best_psnr = float(p/(batch_idx+1))
                torch.save({
                'kwargs': kwargs,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(Cur_model_path, 'model_best_psnr.pth'))     
            ep.set_description("Training epoch %s" % epoch)
        print_to_log_file(log_file, "Training done.")
        
    
