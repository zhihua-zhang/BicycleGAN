# Normalize image tensor
def norm(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
	return ((tensor+1.0)/2.0)*255.0


def loss_discriminator(fakefG, netD, real2D, labels_real, labels_fake, criterion_GAN):
    loss_real = criterion_GAN(netD(real2D), labels_real)
    loss_fake = criterion_GAN(netD(fakefG.detach()), labels_fake)
    loss_D = loss_real + loss_fake
    return loss_D


def loss_generator(netG, real2G, netD, labels_real, criterion_GAN):
    fake_o_G = netG(real2G)
    loss_G = criterion_GAN(netD(fake_o_G), labels_real)
    return loss_G, fake_o_G


def loss_cycle_consis(netG, fakefG, labels_real, criterion_cycle):
    loss_cycle = criterion_cycle(netG(fakefG), labels_real)
    return loss_cycle