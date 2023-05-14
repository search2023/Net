# %%
import os
import argparse
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--gpu', default='0', type=str, help='gpu to train')
parser.add_argument('--ratio', default=25, type=int, help='ratio')
args = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import model
from torch.autograd import Variable
import numpy as np
from skimage import metrics
import cv2
import time
from scipy import io
import torch
from torch import nn
import utils
from timm.models import create_model
import gc


def reconstruction_padding(model:nn.Module,
                           image_name:str,
                           imsize:int,
                           save_path:str,
                           device,
                           is_save:bool=True,):
    model.eval()
    with torch.no_grad():
        test_img = np.array(io.loadmat(image_name)['data'], dtype=np.float32)
        org_img = test_img
        [row, col] = org_img.shape

        Ipad = utils.padding_img(org_img, imsize)
        inputs = Variable(torch.from_numpy(Ipad.astype('float32')).to(device))

        inputs = torch.unsqueeze(inputs, dim=0)
        inputs = torch.unsqueeze(inputs, dim=0)

        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model(inputs)
        torch.cuda.synchronize()
        time_recon = time.time() - start_time

        outputs = torch.squeeze(outputs)
        outputs = outputs.cpu().data.numpy()

        recon_img = outputs[0:row, 0:col]
        recon_img[recon_img > 1.] = 1.
        recon_img[recon_img < 0.] = 0.

        org_img = np.array(org_img, dtype=np.float32)
        recon_img = np.array(recon_img, dtype=np.float32)
        ssim = metrics.structural_similarity(org_img, recon_img, data_range=1.)
        psnr = metrics.peak_signal_noise_ratio(org_img, recon_img, data_range=1.)

        res_info = "IMG: {}, PSNR/SSIM: {:.6f}/{:.6f}, time: {:.5f}\n".format(
            image_name, psnr, ssim, time_recon)
        print(res_info)

        save_name = image_name.split('/')[-1].split('\\')[-1]
        with open(os.path.join(save_path, 'results.csv'), 'a+') as f:
            store_info = "{},{},{},{}\n".format(
                image_name, psnr, ssim, time_recon)
            f.write(store_info)

        if is_save:
            recon_img_name = "{}_{:.4f}_{:.6f}.png".format(
                os.path.join(save_path, save_name[:-4]), psnr, ssim)
            cv2.imwrite(recon_img_name, recon_img*255)

        return psnr, ssim, time_recon


def main():
    is_cuda = True
    device = torch.device('cuda') if is_cuda else torch.device('cpu')
    if is_cuda: print("cuda: {}\n".format(torch.cuda.get_device_name()))
    ratio = args.ratio
    imsize = 33

    net_name = 'base'
    net = create_model(net_name, pretrained=True, ratio=ratio/100).to(device)

    datasets = ["Set5"]

    is_save = False
    # is_save = not is_save
    save_path = "./results/"+str(ratio)
    for ds in datasets:
        image_path = "../data/test/" + ds
        SSIM, PSNR, time_recon, count = 0, 0, 0, 0
        for _, _, files in os.walk(image_path):
            for file in files:
                print(file)
                image_name = os.path.join(image_path, file)
                psnr, ssim, time_t = reconstruction_padding(
                    net, image_name, imsize, save_path, device, is_save)
                SSIM += ssim
                PSNR += psnr
                time_recon += time_t
                count += 1
                gc.collect()
                torch.cuda.empty_cache()

        avg = "AVERAGE,{},{},{}\n".format(
            PSNR/count, SSIM/count, time_recon/count)
        with open(os.path.join(save_path, 'results.csv'), 'a+') as f:
            f.write(avg)
        print("Average, PSNR, SSIM, time recon")
        print(avg)


if __name__ == '__main__':
    main()