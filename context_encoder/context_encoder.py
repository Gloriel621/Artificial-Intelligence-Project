"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to './data/'
4. Run the sript using command 'python3 context_encoder.py'
"""
if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    import math

    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from PIL import Image

    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torch.autograd import Variable

    from datasets import *
    from models import *

    import torch.nn as nn
    import torch.nn.functional as F
    import torch

    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="size of the batches")
    parser.add_argument("--dataset_name", type=str,
                        default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128,
                        help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=64,
                        help="size of random mask")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=2000,
                        help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False
    checkpoint = os.path.isfile("./checkpoint.pt")
    saved_parameters = torch.load('./checkpoint.pt') if checkpoint else False

    # Calculate output of image discriminator (PatchGAN)
    patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
    patch = (1, patch_h, patch_w)


    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    generator = Generator(channels=opt.channels)
    discriminator = Discriminator(channels=opt.channels)
    

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Dataset loader
    transforms_ = [
        transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        ImageDataset("./data/{}".format(opt.dataset_name),
                    transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    test_dataloader = DataLoader(
        ImageDataset("./data/{}".format(opt.dataset_name),
                    transforms_=transforms_, mode="val"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Checkpoint
    saved_epoch = 0
    saved_batch = 0
    if checkpoint :
        generator.load_state_dict(saved_parameters["modelg_state_dict"])
        discriminator.load_state_dict(saved_parameters["modeld_state_dict"])
        adversarial_loss.load_state_dict(saved_parameters["costadv"])
        pixelwise_loss.load_state_dict(saved_parameters["costpixel"])
        saved_epoch = saved_parameters["epoch"]
        saved_batch = saved_parameters["batch"]
        optimizer_G.load_state_dict(saved_parameters["optimizerg_state_dict"])
        optimizer_D.load_state_dict(saved_parameters["optimizerd_state_dict"])
        
    # Cuda
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    def save_sample(batches_done):
        samples, masked_samples, i = next(iter(test_dataloader))
        samples = Variable(samples.type(Tensor))
        masked_samples = Variable(masked_samples.type(Tensor))
        i = i[0].item()  # Upper-left coordinate of mask
        # Generate inpainted image
        gen_mask = generator(masked_samples)
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i: i + opt.mask_size, i: i + opt.mask_size] = gen_mask
        # Save sample
        sample = torch.cat(
            (masked_samples.data, filled_samples.data, samples.data), -2)
        save_image(sample, "images/{}.png".format(batches_done),
                nrow=6, normalize=True)


    # ----------
    #  Training
    # ----------
    for epoch in range(saved_epoch, opt.n_epochs):
        for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):
            if epoch == saved_epoch and i <= saved_batch:
                continue
            # Adversarial ground truths
            valid = Variable(
                Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(
                Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            masked_parts = Variable(masked_parts.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = generator(masked_imgs)

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(discriminator(gen_parts), valid)
            g_pixel = pixelwise_loss(gen_parts, masked_parts)
            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_pixel

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
            )

            # Generate sample and save model at sample interval 
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == opt.sample_interval-1:
                torch.save(
                {
                "epoch": epoch,
                "modelg_state_dict": generator.state_dict(),
                "optimizerg_state_dict": optimizer_G.state_dict(),
                "modeld_state_dict": discriminator.state_dict(),
                "optimizerd_state_dict": optimizer_D.state_dict(),
                "costadv": adversarial_loss.state_dict(),
                "costpixel": pixelwise_loss.state_dict(),
                "description": f"checkpoint(epoch:{epoch}, batch:{i})",
                "batch": i
                },
                f"./checkpoint.pt",)
                save_sample(batches_done)
