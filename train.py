import os
import warnings
warnings.filterwarnings("ignore")
import time
import pdb
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets import Edge2Shoe
from models import BicycleGAN
from utils import norm, log_and_report, save_results, eval_FID_score, eval_LPIPS_score
from vis_tools import infer_viz
# import vis_tools
# import importlib
# importlib.reload(vis_tools)

import argparse
def get_args(args=None):
    parser = argparse.ArgumentParser(description='argparse')
    model_group = parser.add_argument_group(description='BicycleGAN')
    model_group.add_argument("--run_mode", type=int, default=2,
                             help="which mode we are running:\
                                    image_load_viz:1,\
                                    bbox_stat_hist_viz:2,\
                                    pos_anchor_viz:3,\
                                    model_train:4,\
                                    top_proposal_viz:5,\
                                    nms_infer_viz:6")
    model_group.add_argument("--bz", type=int, default=4,
                             help="mini-batch size")
    model_group.add_argument("--max_epochs", type=int, default=25,
                             help="number of epochs")
    model_group.add_argument("--l_pixel", type=float, default=10.,
                             help="lambda_pixel")
    model_group.add_argument("--l_latent", type=float, default=0.5,
                             help="lambda_latent")
    model_group.add_argument("--l_kl", type=float, default=0.01,
                             help="lambda_kl")
    model_group.add_argument("--nz", type=int, default=8,
                             help="latent code dim")
    model_group.add_argument("--lr", type=float, default=0.0002,
                             help="initial lr")
    model_group.add_argument("--weight_decay", type=float, default=0.,
                             help="l2 weight decay")
    model_group.add_argument("--lr_scheduler", type=list, default=[10,20,25],
                             help="lr scheduler")
    model_group.add_argument("--model_pth", type=str, default="",
                             help="reload model path")
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

    img_dir_tr = './data/edges2shoes/train/'
    train_ds = Edge2Shoe(img_dir_tr)
    train_loader = DataLoader(train_ds, batch_size=args.bz, drop_last=True)
    img_dir_val = './data/edges2shoes/val/'
    val_ds = Edge2Shoe(img_dir_val)
    val_loader = DataLoader(val_ds, batch_size=1, drop_last=True)
    
    model = BicycleGAN(args, val_loader).to(device)

    losses = {
        "loss_GAN_encode": [],
        "loss_GAN_random": [],
        "loss_image_l1": [],
        "loss_latent_l1": [],
        "loss_KL": [],
        "total_loss": []

    }

    total_steps = len(train_loader)*args.max_epochs

    step = 0
    report_feq = 1000
    save_freq = 1

    for e in range(args.max_epochs):
        start = time.time()
        
        for idx, data in enumerate(train_loader):
            
            ########## Process Inputs ##########
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
            real_A = edge_tensor; real_B = rgb_tensor;

            # # option 4: divide mini-batch images
            # half_size = bz // 2
            # # A1, B1 for encoded; A2, B2 for random
            # real_A_encoded = real_A[0:half_size]
            # real_B_encoded = real_B[0:half_size]
            # real_A_random = real_A[half_size:]
            # real_B_random = real_B[half_size:]

            #-------------------------------
            # Optimize EG and D
            #-------------------------------
            
            model.optimize(real_A, real_B)

            #-------------------------------
            #             Logging
            #-------------------------------
            log_and_report(model, losses, step, report_feq, e, total_steps)
            step += 1
                
        
        end = time.time()
        print("Train Epoch: {}, Time: {:.3f}s".format(e+1, end-start))

        if e+1 == save_freq:
            # visualize
            n_plot = 10
            infer_viz(model, val_loader, epoch=e, n_plot=n_plot)
            
            # save results
            save_results(model, losses, epoch=e)
            
            # report performance
            # FID
            gen_dataset, FID_score = eval_FID_score(model, val_loader, evaluate_num=200)
            print('FID_score between real_dataset and generated image set: {:.3f}'.format(FID_score))

            # LPIPS score
            lpips_score = eval_LPIPS_score(model, gen_dataset)
            print('LPIPS score between real_dataset and generated image set: {:.3f}'.format(lpips_score))