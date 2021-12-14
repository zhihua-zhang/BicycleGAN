import os
import subprocess
import json
import numpy as np
import tqdm
import scipy
from scipy import linalg
from matplotlib import pyplot as plt

import torch
from torch.utils.data import TensorDataset
from pytorch_fid.inception import InceptionV3
import torchvision.utils as vutils


# Normalize image tensor
def norm(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
    return (tensor+1.0)/2.0

def build_feature_table(dataset, model, batch_size, dim, device):
    '''
    Argms: 
    Input:
        dataset: pytorch dataset, you want to evaluate IS score on
        model: Inception network v3
        batch_size: int number
        dim: for IS computation, dim should be 1000 as the final softmax out put dimension
        device: device type torch.device("cuda:0") or torch.device("cpu")
    Output:
        feature_table: (n,dim) numpy matrix
    '''
    # model enter eval mode
    model.eval()

    # initalize the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    n = len(dataset)
    idx_counter = 0

    # feature table
    feature_table = np.zeros((n, dim))

    for i, data in tqdm.tqdm(enumerate(dataloader, 0)):
        image = data[0].to(device)
        
        with torch.no_grad():
            pred = model(image)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            feature_table[idx_counter:idx_counter+pred.shape[0]] = pred
            idx_counter += len(pred)
    
    return feature_table


def compute_stat(feature_table):
    '''
    Argms: 
    Input:
        feature_table: (n,dim) numpy matrix
    Output:
        mu: mean along row dimension
        sigma: covarance matrix of dataset
    '''
    # compute mean and sigma based on activation table
    mu = np.mean(feature_table, axis=0)
    sigma = np.cov(feature_table, rowvar=False)

    return mu, sigma


def compute_FID(mu_1, sigma_1, mu_2, sigma_2, eps=1e-6):
    '''
    Argms: 
    Input:
        mu_1: mean vector we get for dataset1 
        sigma_1: covariance matrix for dataset1
        mu_2: mean vector we get for dataset2 
        sigma_2: covariance matrix for dataset1
    Output:
        FID score: float
    '''
    # compute mu difference
    mu_diff = mu_1 - mu_2

    # compute square root of Sigma1*Sigma2 using "linalg.sqrtm" from scipy 
    # please name the resulting matrix as covmean
    covmean = scipy.linalg.sqrtm(sigma_1.dot(sigma_2))

    # The following block take care of imagionary part of covmean 
    #################################################################
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma_1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_1 + offset).dot(sigma_2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    #################################################################

    # compute FID score, based on eqution.(10) in pdf FID part.
    FID_score = np.sum(mu_diff**2) + np.trace(sigma_1) + np.trace(sigma_2) - 2*np.trace(covmean)
    return FID_score


def FID(dataset_1, dataset_2, device, batch_size=64, dim=2048, block_idx = 3):
    '''
    Argms: 
    Input:
        dataset_1: pytorch dataset
        dataset_2: pytorch dataset
        device: device type torch.device("cuda:0") or torch.device("cpu")
        batch_size: int number
        dim: for IS computation, dim should be 1000 as the final softmax out put dimension
        block_idx: the block stage index we want to use in inception module
    Output:
        FID_score: float
    '''
    # load InveptionV3 model
    model = InceptionV3([block_idx]).to(device)

    # build up the feature table 
    feature_table_1 = build_feature_table(dataset_1, model, batch_size, dim, device)
    feature_table_2 = build_feature_table(dataset_2, model, batch_size, dim, device)

    # compute mu, sigma for dataset 1&2
    mu_1, sigma_1 = compute_stat(feature_table_1)
    mu_2, sigma_2 = compute_stat(feature_table_2)

    # FID score computation
    FID_score = compute_FID(mu_1, sigma_1, mu_2, sigma_2, eps=1e-6)
    
    return FID_score

def eval_FID_score(model, val_loader, evaluate_num = 200):
    device = model.device
    os.makedirs("./data/eval/gen_B", exist_ok=True)
    
    gen_set = []
    with torch.no_grad():
        for idx, data in enumerate(val_loader, 0):
            if idx == evaluate_num:
                break

            edge_tensor, rgb_tensor = data
            edge_tensor = norm(edge_tensor).to(device)
            real_A = edge_tensor
            bz = real_A.size(0)
            
            z_random = torch.randn(bz, model.args.nz, device=device)
            gen_B_random = model.generator(real_A, z_random)
            fake_denorm = denorm(gen_B_random)
            gen_set.append(fake_denorm)
            
            # plt.imsave('./data/eval/gen_B/gen_B' + str(idx) + '.png', fake_denorm.type(torch.uint8).cpu().squeeze().permute(1,2,0).numpy())
            
    gen_dataset = TensorDataset(torch.cat(gen_set))
    FID_score = FID(model.real_dataset, gen_dataset, device)
    # output = subprocess.run([f"python -m pytorch_fid './data/eval/real_B' './data/eval/gen_B' --device {device}"], shell=True, capture_output=True, text=True)
    # FID_score = float(output.stdout.split()[-1])
    return FID_score

def eval_LPIPS_score(model, val_loader):
    device = model.device
    lpips_score = []

    with torch.no_grad():
        for idx, data in enumerate(val_loader, 0):
            edge_tensor, rgb_tensor = data
            edge_tensor = norm(edge_tensor).to(device)
            real_A = edge_tensor
            bz = real_A.size(0)
            
            gen_set = []
            for _ in range(10):
                z_random = torch.randn(bz, model.args.nz, device=device)
                gen_B_random = model.generator(real_A, z_random)
                fake_denorm = denorm(gen_B_random)
                gen_set.append(fake_denorm)
            
            d_all = []
            for i1 in range(10):
                for i2 in range(i1+1, 10):
                    d = model.loss_fn_alex(gen_set[i1], gen_set[i2])
                    d_all.append(d)
            
            lpips_score.append(sum(d_all) / len(d_all))
            
    lpips_score = torch.mean(torch.cat(lpips_score)).item()
    return lpips_score



def log_and_report(model, metrics, step, report_feq, epoch, total_steps):
    loss_GAN_encode = model.loss_G_GAN_encoded.item() + model.loss_D_GAN_encoded.item()
    loss_GAN_random = model.loss_G_GAN_random.item() + model.loss_D_GAN_random.item()
    loss_image_l1 = model.loss_image_l1.item()
    loss_image_l1_rev = model.loss_image_l1_rev.item()
    loss_latent_l1 = model.loss_latent_l1.item()
    loss_KL = model.loss_KL.item()
    total_loss = loss_GAN_encode + loss_GAN_random + loss_image_l1 + loss_latent_l1 + loss_KL

    metrics["loss_GAN_encode"].append(loss_GAN_encode)
    metrics["loss_GAN_random"].append(loss_GAN_random)
    metrics["loss_image_l1"].append(loss_image_l1)
    metrics["loss_image_l1_rev"].append(loss_image_l1_rev)
    metrics["loss_latent_l1"].append(loss_latent_l1)
    metrics["loss_KL"].append(loss_KL)
    metrics["total_loss"].append(total_loss)

    if (step+1) % report_feq == 0:
        print('Epoch {}, Step {}/{}: Total Loss: {:.3f} loss_GAN_encode: {:.3f} loss_GAN_random: {:.3f}\n\t loss_image_l1: {:.3f} loss_image_l1_rev: {:.3f} loss_latent_l1: {:.3f} loss_KL: {:.3f}'.format
                    (epoch+1, step+1, total_steps, total_loss, loss_GAN_encode, loss_GAN_random, loss_image_l1, loss_image_l1_rev, loss_latent_l1, loss_KL))

def save_results(model, metrics, epoch):
    if os.path.exists("./checkpoints"):
        os.system("rm -rf ./checkpoints")
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.generator.state_dict(), "./checkpoints/generator-epoch={}.pth".format(epoch+1))
    torch.save(model.encoder.state_dict(), "./checkpoints/encoder-epoch={}.pth".format(epoch+1))
    torch.save(model.D_VAE.state_dict(), "./checkpoints/D_VAE-epoch={}.pth".format(epoch+1))
    torch.save(model.D_LR.state_dict(), "./checkpoints/D_LR-epoch={}.pth".format(epoch+1))

    if os.path.exists("./logs"):
        os.system("rm -rf ./logs")
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/metrics-epoch={}".format(epoch+1), "w") as f:
        json.dump(metrics, f)

def infer_viz(model, val_loader, epoch, n_eval, n_gen=8):
    device = model.device

    real_edges = []
    real_shoes = []
    gen_shoes_encode = []
    gen_shoes_random = [[] for _ in range(n_gen)]
    with torch.no_grad():
        for idx, data in enumerate(val_loader, 0):
            if idx == n_eval:
                break
            
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
            real_A = edge_tensor
            real_B = rgb_tensor
            bz = real_A.size(0)

            # cVAE-GAN G
            mu_B, logvar_B = model.encoder(real_B)
            z_encoded = torch.randn(bz, model.args.nz, device=device) * torch.exp(0.5 * logvar_B) + mu_B
            gen_B_encoded = model.generator(real_A, z_encoded).detach()

            # cLR-GAN G
            for j in range(n_gen):
                z_random = torch.randn(bz, model.args.nz, device=device)
                gen_B_random = model.generator(real_A, z_random).detach().cpu()
                gen_shoes_random[j].append(gen_B_random)
            
            real_edges.append(real_A.cpu())
            real_shoes.append(real_B.cpu())
            gen_shoes_encode.append(gen_B_encoded.cpu())

    real_edges, real_shoes, gen_shoes_encode = map(lambda x: torch.cat(x), [real_edges, real_shoes, gen_shoes_encode])
    for j in range(n_gen):
        gen_shoes_random[j] = torch.cat(gen_shoes_random[j])

    viz_real_edge = vutils.make_grid(real_edges, nrow=1, padding=2, normalize=True).permute(1,2,0)
    viz_real_shoe = vutils.make_grid(real_shoes, nrow=1, padding=2, normalize=True).permute(1,2,0)
    viz_gen_shoes_encode = vutils.make_grid(gen_shoes_encode, nrow=1, padding=2, normalize=True).permute(1,2,0)
    viz_gen_shoes_random = [vutils.make_grid(gen_shoes_random[j], nrow=1, padding=2, normalize=True).permute(1,2,0) for j in range(n_gen)]

    fig, ax = plt.subplots(1, 3+n_gen, figsize=(3*(3+n_gen), 3*n_eval))
    
    ax[0].imshow(viz_real_edge); ax[0].set_title("Real edge")
    ax[1].imshow(viz_real_shoe); ax[1].set_title("Real shoe")
    ax[2].imshow(viz_gen_shoes_encode); ax[2].set_title("Gen shoe (encode)")
    for j in range(n_gen):
        ax[j+3].imshow(viz_gen_shoes_random[j])
        if j == 0:
            ax[j+3].set_title("Gen shoe (random)")
    
    
    os.makedirs("./results", exist_ok=True)
    fig.savefig(f"./results/biGAN_viz_epoch={epoch+1}.png")