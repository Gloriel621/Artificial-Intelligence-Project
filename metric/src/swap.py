import os 

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import *
from model import *


def swap(config):
    generator = Generator(channels=3)
    generator.load_state_dict(config["state_dict"])

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = SwapDataset(
        "metric/data/raw", 
        transform, 
        image_size=config["image_size"], 
        mask_size=config["mask_size"]
    )

    generator.eval()
    with torch.no_grad():
        for i, (image, gap) in enumerate(tqdm(dataset)):
            output = generator(image)
            image[:, :, gap: gap + config["mask_size"], gap: gap + config["mask_size"]] = output

            filename = dataset.path[i].split("/")[-1]
            path = os.path.join(config["output_path"], filename)

            save_image(image, path, normalize=True)


if __name__ == "__main__":
    config = {
        "state_dict" : torch.load("metric/save/swap_partial.pth", map_location="cpu"), 
        "image_size" : 128,
        "mask_size" : 64,
        "output_path" : "metric/data/swap_partial",
    }
    swap(config)