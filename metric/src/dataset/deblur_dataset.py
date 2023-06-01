import glob

import cv2
from torch.utils.data import Dataset

train_transform = A.Compose([
    


class DeblurDataset(Dataset):
    def __init__(self, data_path, transforms=None, data_type="train"):
        self.path = sorted(glob.glob("{}/*.jpg".format(data_path)))
        self.path = self.path[:-2000] if data_type == "train" else self.path[-2000:]

        self.transform = transforms
        self.data_type = data_type
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        image = cv2.imread(self.path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(idx)

        if self.data_type == "train":
            label = image.copy()
            image = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=3)
            image, label = self.transform(images=[image, label])["images"]
            return image, label
        else:
            image = self.transform(image=image)["image"]
            return image

        