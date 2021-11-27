import warnings

warnings.filterwarnings("ignore")
import argparse, os
import time
import pdb

import numpy as np
import torch
from torch.utils import data
from torch import nn, optim

from datasets import Edge2Shoe
from models import Encoder, Generator, Discriminator
from utils import norm, denorm

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
                             help="anchor box scale")
    model_group.add_argument("--model_pth", type=str, default="",
                             help="reload model path")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args

if __name__ == "__main__":
    img_dir = './data/edges2shoes/train/'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_gpu = torch.cuda.is_available()

    torch.manual_seed(1)
    np.random.seed(1)

    image_shape = (3,128,128)

    args = get_args()
    args.device = device
    args.image_shape = image_shape

    train_ds = Edge2Shoe(img_dir)
    train_loader = data.DataLoader(train_ds, batch_size=args.bz, drop_last=True)

    mae_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()

    generator = Generator(args).to(device)
    encoder = Encoder(args).to(device)
    D_VAE = Discriminator(args).to(device)
    D_LR = Discriminator(args).to(device)


    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=args.lr)
    optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=args.lr)

    valid = torch.ones(1)
    fake = torch.zeros(1)
    valid_target, fake_target = None, None

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
    report_feq = 20
    for e in range(args.max_epochs):
        start = time.time()
        for idx, data in enumerate(train_loader):
            
            ########## Process Inputs ##########
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
            real_A = edge_tensor; real_B = rgb_tensor;

            bz = real_A.size(0)

            #-------------------------------
            # Train Generator and Encoder
            #-------------------------------
            # cVAE-GAN
            _ = D_VAE.requires_grad_(False)
            _ = D_LR.requires_grad_(False)
            _ = generator.requires_grad_(True)
            _ = encoder.requires_grad_(True)
            optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            
            mu_B, logvar_B = encoder(real_B)
            z_encoded = torch.randn(mu_B.size(0), mu_B.size(1), device=device) * torch.exp(logvar_B) + mu_B
            fakeB_encoded = generator(real_A, z_encoded)
            pred_fakeB_encoded = D_VAE([real_A, fakeB_encoded])
            if valid_target is None:
                valid_target = valid.expand_as(pred_fakeB_encoded).to(device)
                fake_target = fake.expand_as(pred_fakeB_encoded).to(device)
                
            loss_G_GAN_encoded = bce_loss(pred_fakeB_encoded, fake_target) * bz
            loss_image_l1 = mae_loss(fakeB_encoded, real_B)
            loss_KL = 0.5 * (torch.exp(logvar_B) + mu_B**2 - 1 - logvar_B).sum()
            
            loss_image_l1 *= args.l_pixel
            loss_KL *= args.l_kl
            loss = loss_G_GAN_encoded + loss_image_l1 + loss_KL
            loss.backward()
            optimizer_E.step()
            optimizer_G.step()
            
            # cLR-GAN
            z_random = torch.randn(mu_B.size(0), mu_B.size(1), device=device)
            fake_B_random = generator(real_A, z_random)
            mu_z_recons, logvar_z_recons = encoder(fake_B_random)
            pred_fakeB_random = D_LR([real_A, fake_B_random])
            loss_G_GAN_random = bce_loss(pred_fakeB_random, fake_target) * bz
            loss_latent_l1 = mae_loss(z_random, mu_z_recons)

            loss_latent_l1 *= args.l_latent
            loss = loss_G_GAN_random + loss_latent_l1
            loss.backward()

            #-------------------------------
            # Train Discriminator (cVAE-GAN)
            #-------------------------------
            _ = D_VAE.requires_grad_(True)
            _ = D_LR.requires_grad_(True)
            _ = generator.requires_grad_(False)
            _ = encoder.requires_grad_(False)
            optimizer_D_VAE.zero_grad()
            pred_realB_encoded = D_VAE([real_A, real_B])
            loss_D_GAN_encoded_real = bce_loss(pred_realB_encoded, valid_target) * bz
            pred_fakeB_encoded = D_VAE([real_A, fakeB_encoded.detach()])
            loss_D_GAN_encoded_fake = bce_loss(pred_fakeB_encoded, fake_target) * bz

            loss_D_GAN_encoded = loss_D_GAN_encoded_real + loss_D_GAN_encoded_fake
            loss_D_GAN_encoded.backward()
            optimizer_D_VAE.step()


            #-------------------------------
            # Train Discriminator (cLR-GAN)
            #-------------------------------
            optimizer_D_LR.zero_grad()
            pred_realB_random = D_LR([real_A, real_B])
            loss_D_GAN_random_real = bce_loss(pred_realB_random, valid_target) * bz
            pred_fakeB_random = D_LR([real_A, fake_B_random.detach()])
            loss_D_GAN_random_fake = bce_loss(pred_fakeB_random, fake_target) * bz

            loss_D_GAN_random = loss_D_GAN_random_real + loss_D_GAN_random_fake
            loss_D_GAN_random.backward()
            optimizer_D_LR.step()

            #-------------------------------
            #             Logging
            #-------------------------------
            loss_GAN_encode = loss_G_GAN_encoded.item() + loss_D_GAN_encoded.item()
            loss_GAN_random = loss_G_GAN_random.item() + loss_D_GAN_random.item()
            loss_image_l1 = loss_image_l1.item()
            loss_latent_l1 = loss_latent_l1.item()
            loss_KL = loss_KL.item()
            total_loss = loss_GAN_encode + loss_GAN_random + loss_image_l1 + loss_latent_l1 + loss_KL

            losses["loss_GAN_encode"].append(loss_GAN_encode)
            losses["loss_GAN_random"].append(loss_GAN_random)
            losses["loss_image_l1"].append(loss_image_l1)
            losses["loss_latent_l1"].append(loss_latent_l1)
            losses["loss_KL"].append(loss_KL)
            losses["total_loss"].append(total_loss)

            if idx % report_feq == report_feq-1:
                print('Epoch {}, Step {}/{}: Total Loss: {:.4f} loss_GAN_encode: {:.4f} loss_GAN_random: {:.4f} loss_image_l1: {:.4f} loss_latent_l1: {:.4f} loss_KL: {:.4f}'.format
                    (e+1, idx+1, total_steps, total_loss, loss_GAN_encode, loss_GAN_random, loss_image_l1, loss_latent_l1, loss_KL))

            pass
            # Optional TODO: 
            # 1. You may want to visualize results during training for debugging purpose
            # 2. Save your model every few iteration
            
        end = time.time()
        print("Train Epoch: {}, Time: {:.4f}s".format(e+1, end-start))
        start = end
