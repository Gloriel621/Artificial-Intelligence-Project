import glob
from PIL import Image

from torch.utils.data import Dataset


class SwapDataset(Dataset):
    def __init__(self, data_path, transforms=None, image_size=128, mask_size=64):
        self.path = glob.glob("{}/*.jpg".format(data_path))
        self.image_size = image_size
        self.mask_size = mask_size
        self.transform = transforms
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        image = Image.open(self.path[idx])
        image = self.transform(image)

        gap = (self.image_size - self.mask_size) // 2
        image[:, gap : gap + self.mask_size, gap : gap + self.mask_size] = 1
        image = image.unsqueeze(0)
        
        return image, gap