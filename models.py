
import numpy as np
import pdb

from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import seed_everything

from utils import norm, denorm

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

        # print("out before up size {}".format(out.size()))
        
        out = self.up(out)
        if not self.outermost:
            out = torch.cat([out, x], dim=1)

        # print("out after up size {}".format(out.size()))
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