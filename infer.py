import os
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
    model_group.add_argument("--max_epochs", type=int, default=25,
                             help="number of epochs")
    model_group.add_argument("--l_G", type=float, default=1.0,
                             help="lambda_GAN_G")
    model_group.add_argument("--l_pixel", type=float, default=10.,
                             help="lambda_pixel")
    model_group.add_argument("--l_pixel_rev", type=float, default=1.,
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
                             help="number of images to generate per input")
    model_group.add_argument("--n_eval", type=int, default=20,
                             help="number of image to evaluate")
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

    img_dir_val = './data/edges2shoes/val/'
    val_ds = Edge2Shoe(img_dir_val)
    val_loader = DataLoader(val_ds, batch_size=1)
    
    model = BicycleGAN(args, val_loader).to(device)
    if args.model_pth_dir != "":
        # reload results
        model_pth_dir = os.path.join(args.model_pth_dir, "checkpoints")
        for path in os.listdir(model_pth_dir):
            if ".pth" in path:
                name = path.split(".")
                start_epoch = int(name[0].split("epoch=")[-1])
                break
        
        _get_reload_pth = lambda name: os.path.join(model_pth_dir, f"{name}-epoch={start_epoch}.pth")
        model.generator.load_state_dict(torch.load(_get_reload_pth("generator"), map_location=device))
        model.encoder.load_state_dict(torch.load(_get_reload_pth("encoder"), map_location=device))
        model.D_VAE.load_state_dict(torch.load(_get_reload_pth("D_VAE"), map_location=device))
        model.D_LR.load_state_dict(torch.load(_get_reload_pth("D_LR"), map_location=device))

    model.eval()
    infer_viz(model, val_loader, epoch=-1, n_eval=args.n_eval, n_gen=args.n_gen)