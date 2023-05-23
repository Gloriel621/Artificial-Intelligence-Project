import os
import random
import shutil
from PIL import Image

# Set paths
img_folder = 'img_celeba'
label_folder = 'yolov7/celeba_demo/labels'
image_folder = 'yolov7/celeba_demo/images'

# Create train, val and test directories
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(label_folder, folder), exist_ok=True)
    os.makedirs(os.path.join(image_folder, folder), exist_ok=True)

# Read bounding box file
with open('list_bbox_celeba.txt', 'r') as f:
    data = f.readlines()[2:]
    
# # Shuffle data
# random.shuffle(data)

# Split data into train, val, and test
train_data = data[:180000]
val_data = data[180000:190000]
test_data = data[190000:200000]

# Write train data to files and copy images to train folder
for line in train_data:
    line = line.strip().split()
    filename = line[0][:-3] + 'txt'
    
    # Open image and get dimensions
    img_path = os.path.join(img_folder, line[0])
    with Image.open(img_path) as img:
        img_width, img_height = img.size
        
    # Write label file in YOLO format
    with open(os.path.join(label_folder, 'train', filename), 'w') as f:
        
        x1 = int(line[1])
        y1 = int(line[2])
        w = int(line[3])
        h = int(line[3])

        x = x1 + (w / 2)
        y = y1 + (h / 2)
        xn = x / img_width
        yn = y / img_height
        wn = w / img_width
        hn = h / img_height
        f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}'.format(xn, yn, wn, hn))
    # Copy image to train folder
    shutil.copy(img_path, os.path.join(image_folder, 'train', line[0]))

# Write val data to files and copy images to val folder
for line in val_data:
    line = line.strip().split()
    filename = line[0][:-3] + 'txt'
    
    # Open image and get dimensions
    img_path = os.path.join(img_folder, line[0])
    with Image.open(img_path) as img:
        img_width, img_height = img.size
        
    # Write label file in YOLO format
    with open(os.path.join(label_folder, 'val', filename), 'w') as f:
        
        x1 = int(line[1])
        y1 = int(line[2])
        w = int(line[3])
        h = int(line[3])

        x = x1 + (w / 2)
        y = y1 + (h / 2)
        xn = x / img_width
        yn = y / img_height
        wn = w / img_width
        hn = h / img_height
        f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}'.format(xn, yn, wn, hn))
    
    # Copy image to val folder
    shutil.copy(img_path, os.path.join(image_folder, 'val', line[0]))

# Write test data to files and copy images to test folder
for line in test_data:
    line = line.strip().split()
    filename = line[0][:-3] + 'txt'
    
    # Open image and get dimensions
    img_path = os.path.join(img_folder, line[0])
    with Image.open(img_path) as img:
        img_width, img_height = img.size
        
    # Write label file in YOLO format
    with open(os.path.join(label_folder, 'test', filename), 'w') as f:
        
        x1 = int(line[1])
        y1 = int(line[2])
        w = int(line[3])
        h = int(line[3])

        x = x1 + (w / 2)
        y = y1 + (h / 2)
        xn = x / img_width
        yn = y / img_height
        wn = w / img_width
        hn = h / img_height
        f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}'.format(xn, yn, wn, hn))
    
    # Copy image to test folder
    shutil.copy(img_path, os.path.join(image_folder, 'test', line[0]))