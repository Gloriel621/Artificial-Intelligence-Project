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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
    
    Tensor = torch.FloatTensor

    transforms_ = [
        transforms.Resize((128, 128), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    dataloader = DataLoader(
        ImageDataset("./data/{}".format("img_align_celeba"),
                    transforms_=transforms_, mode="val"),
        batch_size=1,
        shuffle=False,
    )
    
    modelg = Generator()
    modeld = Discriminator()
    checkpoint = torch.load('./checkpoint.pt')

    modelg.load_state_dict(checkpoint["modelg_state_dict"])
    modeld.load_state_dict(checkpoint["modeld_state_dict"])

    conv_blocks_idx = []
    conv_blocks = []
    
    for idx, module in enumerate(modelg.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            conv_blocks_idx.append(idx)
            conv_blocks.append(module)
            
    print(conv_blocks_idx)
    print(conv_blocks)
    
    feature_maps = []

    def hook(module, input, output):
        feature_maps.append(output)
        
    for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        gen_mask = modelg(masked_imgs)
        xx = masked_parts[0].item()
        
        filled_samples = masked_imgs.clone()
        filled_samples[:, :, xx : xx + 64, xx : xx + 64] = gen_mask
        
        fig = plt.figure(figsize=(10, 10))   
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(imgs[0].detach().numpy().swapaxes(0,1).swapaxes(1,2))
        ax.axis('off')
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(masked_imgs[0].detach().numpy().swapaxes(0,1).swapaxes(1,2))
        ax.axis('off')
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(filled_samples[0].detach().numpy().swapaxes(0,1).swapaxes(1,2))
        ax.axis('off')
        plt.savefig('original_img.png', dpi=300, bbox_inches='tight')
        
        for j, (conv) in enumerate(conv_blocks):
            conv.register_forward_hook(hook)
            output = modelg(masked_imgs)
            feature_map = feature_maps[j]
            feature_map = feature_map.detach().cpu().numpy()
            fig = plt.figure(figsize=(10, 10))   
            for idx in range(64):
                ax = fig.add_subplot(8, 8, idx+1)
                img = feature_map[0, -idx, :, :]
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                plt.savefig('feature_map_block{}.png'.format(j), dpi=300, bbox_inches='tight')
        break
        
