import os
import time
import pdb
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets import Edge2Shoe
from models import BicycleGAN
from utils import *

import argparse
def get_args(args=None):
    parser = argparse.ArgumentParser(description='argparse')
    model_group = parser.add_argument_group(description='BicycleGAN')
    model_group.add_argument("--use_condGAN", type=int, default=0,
                             help="whether use conditional GAN")
    model_group.add_argument("--add_all", type=int, default=1,
                             help="whether add latent code to all intermediate layers")
    model_group.add_argument("--bz", type=int, default=32,
                             help="mini-batch size")
    model_group.add_argument("--max_epochs", type=int, default=50,
                             help="number of epochs")
    model_group.add_argument("--l_G", type=float, default=1.0,
                             help="lambda_GAN_G")
    model_group.add_argument("--l_pixel", type=float, default=10.,
                             help="lambda_pixel")
    model_group.add_argument("--l_pixel_rev", type=float, default=0.05,
                             help="lambda_pixel_rev")
    model_group.add_argument("--l_latent", type=float, default=0.5,
                             help="lambda_latent")
    model_group.add_argument("--l_kl", type=float, default=0.01,
                             help="lambda_kl")
    model_group.add_argument("--nz", type=int, default=8,
                             help="latent code dim")
    model_group.add_argument("--base_lr", type=float, default=1e-5,
                             help="initial lr")
    model_group.add_argument("--weight_decay", type=float, default=0.,
                             help="l2 weight decay")
    model_group.add_argument("--n_gen", type=int, default=8,
                             help="number of plots to generate")
    model_group.add_argument("--model_pth_dir", type=str, default="",
                             help="directory of model path to reload")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_gpu = torch.cuda.is_available()

    torch.manual_seed(1)
    np.random.seed(1)

    image_shape = (3,128,128)

    args = get_args()
    args.device = device
    args.image_shape = image_shape
    args.lr = args.base_lr * args.bz / 32

    img_dir_tr = './data/edges2shoes/train/'
    train_ds = Edge2Shoe(img_dir_tr)
    train_loader = DataLoader(train_ds,
                              batch_size=args.bz,
                              drop_last=True,
                              shuffle=True,
                              num_workers=4)
    img_dir_val = './data/edges2shoes/val/'
    val_ds = Edge2Shoe(img_dir_val)
    val_loader = DataLoader(val_ds, batch_size=1)
    
    model = BicycleGAN(args, val_loader).to(device)
    
    metrics = {
        "loss_GAN_encode": [],
        "loss_GAN_random": [],
        "loss_image_l1": [],
        "loss_image_l1_rev": [],
        "loss_latent_l1": [],
        "loss_KL": [],
        "total_loss": [],
        "fid": [],
        "lpips": []
    }

    total_steps = len(train_loader)*args.max_epochs

    step = 0
    report_freq = 100 * max(1, 128 // args.bz)
    save_freq = 10
    if args.model_pth_dir != "" and len(os.listdir(args.model_pth_dir)):
        # reload results
        for path in os.listdir(args.model_pth_dir):
            if ".pth" in path:
                name = path.split(".")
                start_epoch = int(name[0].split("epoch=")[-1])
                break
        model.generator.load_state_dict(torch.load(os.path.join(args.model_pth_dir, f"generator-epoch={start_epoch}.pth"), map_location=device))
        model.encoder.load_state_dict(torch.load(os.path.join(args.model_pth_dir, f"encoder-epoch={start_epoch}.pth"), map_location=device))
        model.D_VAE.load_state_dict(torch.load(os.path.join(args.model_pth_dir, f"D_VAE-epoch={start_epoch}.pth"), map_location=device))
        model.D_LR.load_state_dict(torch.load(os.path.join(args.model_pth_dir, f"D_LR-epoch={start_epoch}.pth"), map_location=device))

        with open(f"./logs/metrics-epoch={start_epoch}", "r") as f:
            metrics = json.load(f)
    else:
        start_epoch = 0

    print("training starts ...")
    for e in range(start_epoch, args.max_epochs):
        start = time.time()
        
        model.train()
        for idx, data in enumerate(train_loader):
            
            ########## Process Inputs ##########
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
            real_A = edge_tensor; real_B = rgb_tensor;

            # divide mini-batch images
            bz1 = len(real_A) // 2
            bz2 = len(real_A) - bz1

            real_A_encoded = real_A[0:bz1]
            real_B_encoded = real_B[0:bz1]
            real_A_random = real_A[bz1:]
            real_B_random = real_B[bz1:]

            #-------------------------------
            # Optimize EG and D
            #-------------------------------
            model.optimize(real_A_encoded, real_B_encoded, real_A_random, real_B_random)

            #-------------------------------
            #             Logging
            #-------------------------------
            log_and_report(model, metrics, step, report_freq, e, total_steps)
            step += 1
                
        
        end = time.time()
        print("Train Epoch: {}, Time: {:.3f}s".format(e+1, end-start))

        model.eval()
        # visualize
        infer_viz(model, val_loader, epoch=e, n_plot=10, n_gen=args.n_gen)
        
        # save results
        if (e+1) % save_freq == 0:
            save_results(model, metrics, epoch=e)
        
        # report performance
        # FID
        FID_score = eval_FID_score(model, val_loader, evaluate_num=200)
        print('FID_score between real_dataset and generated image set: {:.3f}'.format(FID_score))
        metrics["fid"].append(FID_score)

        # LPIPS score
        lpips_score = eval_LPIPS_score(model, val_loader)
        print('LPIPS score between real_dataset and generated image set: {:.3f}'.format(lpips_score))
        metrics["lpips"].append(lpips_score)