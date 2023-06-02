import glob

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

train_transform = A.Compose([
    A.Resize(128, 128),
    A.HorizontalFlip(),
    A.ShiftScaleRotate(rotate_limit=15),
])
val_transform = A.Compose([
    A.Resize(128, 128),
])
final_transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2(),
])

class DeblurDataset(Dataset):
    def __init__(self, data_path, data_type="train"):
        self.path = sorted(glob.glob("{}/*.jpg".format(data_path)))
        if data_type != "test":
            self.path = self.path[:-2000] if data_type == "train" else self.path[-2000:]

        self.transform = train_transform if data_type == "train" else val_transform
        self.data_type = data_type
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        image = cv2.imread(self.path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.data_type != "test":
            image = self.transform(image=image)["image"]

            label = image.copy()
            image = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=4)

            image = final_transform(image=image)["image"]
            label = final_transform(image=label)["image"]

            return image, label
        else:
            image = cv2.resize(image, (128, 128))
            image = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=4)
            image = final_transform(image=image)["image"]
            return image

        