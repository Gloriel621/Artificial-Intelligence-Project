import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.utils import save_image

from dataset import *
from model import *


class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        mse = torch.mean((x - y) ** 2)
        return 10 * torch.log10(mse)


def train(config):
    deblurrer = NAFNet(3, 8, 1, [1, 1, 28], [1, 1, 1])

    train_dataset = DeblurDataset(config["data_path"], "train")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    val_dataset = DeblurDataset(config["data_path"], "val") 
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], drop_last=True)

    criterion = PSNRLoss()
    optimizer = optim.Adam(deblurrer.parameters(), lr=config["lr"])
    scaler = torch.cuda.amp.GradScaler()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion.to(device)
    deblurrer.to(device)

    deblurrer.train()
    train_loss = 0
    val_loss = 1E9
    for it in range(config["iteration"]):
        image, label = next(iter(train_loader))
        image = image.to(device)
        label = label.to(device)

        with torch.cuda.amp.autocast():
            output = deblurrer(image)
            loss = criterion(output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss
        if it != 0 and (it % config["eval_step"] == 0 or it == config["iteration"] - 1):
            new_val_loss = eval(deblurrer, val_loader, criterion, device)
            train_loss = (train_loss / config["eval_step"]).item()

            print(f"Iteration {it} | train_loss: {train_loss:.3f}  val_loss: {new_val_loss:.3f}")

            if new_val_loss < val_loss:
                val_loss = new_val_loss
                torch.save(deblurrer.state_dict(), "metric/save/deblur.pth")
                print("Model with best score saved")

            train_loss = 0

        del image, label, output, loss


def eval(deblurrer, val_loader, criterion, device):
    deblurrer.eval()
    val_loss = 0

    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device)
            label = label.to(device)

            with torch.cuda.amp.autocast():
                output = deblurrer(image)
                loss = criterion(output, label)

            if val_loss == 0:
                save_image(torch.cat([image, output, label], dim=2), "metric/save/deblur.png", nrow=8)

            val_loss += loss
            del image, label, output, loss

    deblurrer.train()
    return val_loss / len(val_loader)


if __name__ == "__main__":
    config = {
        "lr": 3e-5,
        "batch_size": 32,
        "iteration": 20000,
        "eval_step": 256,
        "data_path" : "metric/data/celeba",
    }
    print(config)
    train(config)