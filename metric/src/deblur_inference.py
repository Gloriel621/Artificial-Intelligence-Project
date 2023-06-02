import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import *
from model import *


def inference(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    deblurrer = NAFNet(3, 8, 1, [1, 1, 28], [1, 1, 1])
    deblurrer.load_state_dict(config["state_dict"])

    dataset = DeblurDataset(config["input_path"], "test")

    deblurrer.to(device)
    deblurrer.eval()
    with torch.no_grad():
        for i, image in enumerate(tqdm(dataset)):
            image = image.unsqueeze(0).to(device)

            output = deblurrer(image)
            filename = dataset.path[i].split("/")[-1]
            path = os.path.join(config["output_path"], filename)

            save_image(output, path, normalize=True, range=(-1, 1))

if __name__ == "__main__":
    config = {
        "state_dict" : torch.load("metric/save/deblur.pth"), 
        "input_path" : "metric/data/swap_partial",
        "output_path" : "metric/data/deblur_partial",
    }
    inference(config)