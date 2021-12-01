
import numpy as np
import pdb

import torch
from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset

import lpips


##############################
#        Bicycle GAN 
##############################
class BicycleGAN(nn.Module):
    def __init__(self, args, val_loader):
        super().__init__()
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.args = args
        
        self.mae_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)

        self.generator = Generator(args).to(device)
        self.encoder = Encoder(args).to(device)
        self.D_VAE = Discriminator(args).to(device)
        self.D_LR = Discriminator(args).to(device)


        self.optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.lr)
        self.optimizer_D_VAE = torch.optim.Adam(self.D_VAE.parameters(), lr=args.lr)
        self.optimizer_D_LR = torch.optim.Adam(self.D_LR.parameters(), lr=args.lr)

        self.valid = torch.ones(1)
        self.fake = torch.zeros(1)
        self.valid_target, fake_target = None, None

        # FID real_set
        real_set = []
        for idx, data in enumerate(val_loader, 0):
            edge_tensor, rgb_tensor = data
            real_B = rgb_tensor = rgb_tensor.to(device)
            
            real_set.append(real_B)
        self.real_dataset = TensorDataset(torch.cat(real_set))
    
    def forward_EG(self):
        _ = self.D_VAE.requires_grad_(False)
        _ = self.D_LR.requires_grad_(False)
        _ = self.generator.requires_grad_(True)
        _ = self.encoder.requires_grad_(True)
        
        device = self.device
        bz = len(self.real_A)
        nz = self.args.nz
        
        # cVAE-GAN EG
        self.mu_B, self.logvar_B = self.encoder(self.real_B)
        self.z_encoded = torch.randn(bz, nz, device=device) * torch.exp(self.logvar_B) + self.mu_B

        self.fakeB_encoded = self.generator(self.real_A, self.z_encoded)
        self.pred_fakeB_encoded = self.D_VAE([self.real_A, self.fakeB_encoded])
        
        # cLR-GAN G
        self.z_random = torch.randn(bz, nz, device=device)
        self.fake_B_random = self.generator(self.real_A, self.z_random)
        self.pred_fakeB_random = self.D_LR([self.real_A, self.fake_B_random])
        
        self.mu_z_recons, self.logvar_z_recons = self.encoder(self.fake_B_random)

    def backward_EG(self):
        self.opt_set_zero_grad([self.optimizer_E, self.optimizer_G])
        device = self.device
        
        if self.valid_target is None:
            self.valid_target = self.valid.expand_as(self.pred_fakeB_encoded).to(device)
            self.fake_target = self.fake.expand_as(self.pred_fakeB_encoded).to(device)
        
        self.loss_G_GAN_encoded = self.bce_loss(self.pred_fakeB_encoded, self.fake_target)
        self.loss_image_l1 = self.mae_loss(self.fakeB_encoded, self.real_B) * self.args.l_pixel
        self.loss_KL = (0.5 * (torch.exp(self.logvar_B) + self.mu_B**2 - 1 - self.logvar_B).sum()) * self.args.l_kl

        self.loss_G_GAN_random = self.bce_loss(self.pred_fakeB_random, self.fake_target)
        
        _ = self.encoder.requires_grad_(False)
        self.loss_latent_l1 = self.mae_loss(self.z_random, self.mu_z_recons) * self.args.l_latent
        _ = self.encoder.requires_grad_(True)

        loss = self.loss_G_GAN_encoded + self.loss_image_l1 + self.loss_KL + self.loss_G_GAN_random + self.loss_latent_l1

        loss.backward()
    
    def forward_D(self):
        _ = self.D_VAE.requires_grad_(True)
        _ = self.D_LR.requires_grad_(True)
        _ = self.generator.requires_grad_(False)
        _ = self.encoder.requires_grad_(False)
        
        self.pred_realB_encoded = self.D_VAE([self.real_A, self.real_B])
        self.pred_fakeB_encoded = self.D_VAE([self.real_A, self.fakeB_encoded.detach()])
        self.pred_realB_random = self.D_LR([self.real_A, self.real_B])
        self.pred_fakeB_random = self.D_LR([self.real_A, self.fake_B_random.detach()])
    
    def backward_D(self):
        self.opt_set_zero_grad([self.optimizer_D_VAE, self.optimizer_D_LR])
        
        #-------------------------------
        # Train Discriminator (cVAE-GAN)
        #-------------------------------
        loss_D_GAN_encoded_real = self.bce_loss(self.pred_realB_encoded, self.valid_target)
        loss_D_GAN_encoded_fake = self.bce_loss(self.pred_fakeB_encoded, self.fake_target)
        self.loss_D_GAN_encoded = loss_D_GAN_encoded_real + loss_D_GAN_encoded_fake
        self.loss_D_GAN_encoded.backward()

        #-------------------------------
        # Train Discriminator (cLR-GAN)
        #-------------------------------
        loss_D_GAN_random_real = self.bce_loss(self.pred_realB_random, self.valid_target)
        loss_D_GAN_random_fake = self.bce_loss(self.pred_fakeB_random, self.fake_target)
        self.loss_D_GAN_random = loss_D_GAN_random_real + loss_D_GAN_random_fake
        self.loss_D_GAN_random.backward()
    
    def opt_set_zero_grad(self, optimizers):
        for opt in optimizers:
            opt.zero_grad()
    
    def opt_take_step(self, optimizers):
        for opt in optimizers:
            opt.step()
    
    def update_EG(self):
        self.forward_EG()
        self.backward_EG()
        self.opt_take_step([self.optimizer_E, self.optimizer_G])

    def update_D(self):
        self.forward_D()
        self.backward_D()
        self.opt_take_step([self.optimizer_D_VAE, self.optimizer_D_LR])
    
    def optimize(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B
        
        self.update_EG()
        self.update_D()



##############################
#        Encoder 
##############################
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        """ The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
            This encoder uses resnet-18 to extract features, and further encode them into a distribution
            similar to VAE encoder. 

            Note: You may either add "reparametrization trick" and "KL divergence" or in the train.py file
            
            Args in constructor: 
                nz: latent dimension for z 
  
            Args in forward function: 
                img: image input (from domain B)

            Returns: 
                mu: mean of the latent code 
                logvar: sigma of the latent code 
        """
        self.args = args
        nz = args.nz
        self.device = args.device

        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)
        # feature_extractor: (1,3,128,128) -> (1,256,8,8)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        # pooling: (1,256,8,8) -> (1,256,1,1)
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, nz)
        self.fc_logvar = nn.Linear(256, nz)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar
    
##############################
#        Generator 
##############################
class Unet_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False):
        super().__init__()

        self.nz = nz
        self.input_nc = input_nc + nz
        self.outermost = outermost
        
        # downsample
        down = []
        down += [nn.ReflectionPad2d(1)]
        down += [nn.Conv2d(self.input_nc, inner_nc, kernel_size=4, stride=2, padding=0)]
        if not innermost:
            # (w,h)==(1,1), unable to ins norm
            down += [nn.InstanceNorm2d(inner_nc)]
        if not outermost:
            down = [nn.ReLU(inplace=True)] + down
            # option 1: use Leaky ReLU
            # down = [nn.LeakyReLU(0.2, inplace=True)] + down
        
        # upsample
        up = []
        up += [nn.ReLU()]
        # option 3: how to upsample
        if innermost:
            up += [nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)]
        else:
            up += [nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)]
        if not outermost:
            up += [nn.InstanceNorm2d(outer_nc)]
        else:
            up += [nn.Tanh()]
            
        # combine together
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)
    
    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.reshape(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            input = torch.cat([x, z_img], dim=1)
        else:
            input = x
            
        out = self.down(input)
        
        # no submodule at innermost
        if self.submodule:
            out = self.submodule(out, z)

        out = self.up(out)
        if not self.outermost:
            out = torch.cat([out, x], dim=1)
        return out
        
class Generator(nn.Module):
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B
        
        Args in constructor: 
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
            nz: latent dimension for z 
        
        Args in forward function: 
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns: 
            fake_B: generated image in domain B
    """
    def __init__(self, args, n_unet=7, base_nc=64, dropout=0.0):
        super(Generator, self).__init__()
        # image_shape = (3,128,128)
        image_nc, self.h, self.w = args.image_shape
        nz = args.nz
        
        # option 2: pass z to submodule
        unet_block = Unet_z(base_nc*8, base_nc*8, base_nc*8, 0, submodule=None, innermost=True)
        
        for _ in range(n_unet - 5):
            unet_block = Unet_z(base_nc*8, base_nc*8, base_nc*8, 0, submodule=unet_block)

        unet_block = Unet_z(base_nc*4, base_nc*4, base_nc*8, 0, submodule=unet_block)
        unet_block = Unet_z(base_nc*2, base_nc*2, base_nc*4, 0, submodule=unet_block)
        unet_block = Unet_z(base_nc, base_nc, base_nc*2, 0, submodule=unet_block)
        
        unet_block = Unet_z(image_nc, image_nc, base_nc, nz, submodule=unet_block, outermost=True)
        self.model = unet_block

    def forward(self, x, z):
        out = self.model(x, z)
        return out

##############################
#   DiscriminatorL PatchGAN
##############################
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        """ The discriminator used in both cVAE-GAN and cLR-GAN
            
            Args in constructor: 
                image_shape: shape for real and fake

            Args in forward function: 
                x: image input (real_A, real_B or fake_B)
 
            Returns: 
                discriminator output: could be a single value or a matrix depending on the type of GAN
        """
        
        image_nc, height, width = args.image_shape
        input_nc = 2*image_nc

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        real_A, input_B = x
        input = torch.cat([real_A, input_B], dim=1)
        return self.model(input)