# %%
import os
import argparse
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--gpu', default='0', type=str, help='gpu to train')
parser.add_argument('--batch', default=8, type=int, help='batch size')
args = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import model
import utils
import timm.optim
import timm.scheduler
from timm.models import create_model
import torch
from torch import nn
from engine import train_one_epoch, val_one_epoch
import torch.backends.cudnn as cudnn
from colorama import Fore, Back, Style
sr_ = Style.RESET_ALL


def main():
    seed = 42
    utils.setup_seed(seed)
    cudnn.benchmark = True

    is_cuda = True
    if is_cuda: print("cuda: {}\n".format(torch.cuda.get_device_name()))
    pretrain = False
    use_amp = True
    scaler = torch.cuda.amp.GradScaler()

    base_lr = 1e-4
    min_lr = 1e-6
    warmup_epochs = 20
    num_epochs = 200
    restart_epochs = 200
    CS_ratios = [10]
    model_name = 'base'

    batch_size = args.batch
    img_size = 132
    train_folder = "../data/train/"
    val_folder = "../data/val/"
    train_iter, val_iter = utils.get_train_val_iter_folder(train_folder, val_folder, batch_size, img_size, use_augs=True)
   
    device = torch.device('cuda') if is_cuda else torch.device('cpu')
    for ratio in CS_ratios:
        model = create_model(model_name, pretrained=pretrain, ratio=ratio/100).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        sch_lr = timm.scheduler.CosineLRScheduler(
            optimizer=optimizer,
            t_initial=restart_epochs,
            lr_min=min_lr,
            warmup_t=warmup_epochs,
            warmup_lr_init=base_lr/warmup_epochs,
            # cycle_limit=2,
            # decay_rate=0.5
        )
        loss_fun = nn.MSELoss().to(device)

        print(f"{Fore.GREEN}Net: {model_name}, CS ratio: {ratio}, epoches: {num_epochs}, batch_size: {batch_size}, base lr: {base_lr}, image size: {img_size}{sr_}")

        best_epoch, psnr_max = 0, -1
        trainwriter, evalwriter = utils.load_logging("{}_{}".format(model_name, ratio))
        save_path = os.path.join('.', 'model', str(ratio))
        save_name = model_name+'_{}_best.pkl'.format(ratio)
        for epoch in range(num_epochs):
            sch_lr.step(epoch)
            # print('Current Learning Rate:', optimizer.param_groups[0]['lr'])
            # train
            train_one_epoch(ratio, epoch, model, train_iter, optimizer, loss_fun, device, trainwriter, use_amp, scaler)
            # val
            avg_psnr = val_one_epoch(ratio, epoch, model, val_iter, loss_fun, device, evalwriter, use_amp)
            # save
            if avg_psnr > psnr_max:
                print(f"{Fore.GREEN}Valid Score Improved ({psnr_max:0.6f} ---> {avg_psnr:0.6f})")
                psnr_max = avg_psnr
                best_epoch = epoch
                utils.save_model(model, save_name, save_path)
                print(f"Saved: {save_name}, from epoch: {best_epoch}{sr_}")
            else:
                print(f"{Fore.RED}Not Saved, Best epoch: {best_epoch}{sr_}")


if __name__ == '__main__':
    main()
# %%
