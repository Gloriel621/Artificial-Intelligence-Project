import os
import glob

from tqdm import tqdm
import cv2
import albumentations as A

def blur(config):
    path = glob.glob(os.path.join(config["input_path"], "*.jpg"))
    for p in tqdm(path):
        image = cv2.imread(p)
        image = A.resize(image, 128, 128)
        image = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=4)

        filename = p.split("/")[-1]
        cv2.imwrite(os.path.join(config["output_path"], filename), image)

if __name__ == "__main__":
    config = {
        "input_path" : "metric/data/raw",
        "output_path" : "metric/data/blur_raw",
    }
    blur(config)