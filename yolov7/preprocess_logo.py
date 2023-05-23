import os
import shutil
import random

# Set the paths to the source and destination folders
source_path = 'openlogo'
jpeg_images_path = os.path.join(source_path, 'JPEGImages')
labels_path = os.path.join(source_path, 'labels')
dest_path = 'yolov7/openlogo'

# Create the destination directories if they don't already exist
os.makedirs(os.path.join(dest_path, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(dest_path, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(dest_path, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(dest_path, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(dest_path, 'labels', 'val'), exist_ok=True)
os.makedirs(os.path.join(dest_path, 'labels', 'test'), exist_ok=True)

# Get the list of JPEG images
jpeg_images = [f for f in os.listdir(jpeg_images_path) if f.endswith('.jpg')]

# Shuffle the list of JPEG images
random.shuffle(jpeg_images)

# Split the list of JPEG images into train, val, and test sets
train_images = jpeg_images[:800]
val_images = jpeg_images[800:900]
test_images = jpeg_images[900:1000]

# Move the train images and labels
for f in train_images:
    # Move the image file
    src_file = os.path.join(jpeg_images_path, f)
    dest_file = os.path.join(dest_path, 'images', 'train', f)
    shutil.copy(src_file, dest_file)
    # Move the label file if it exists
    label_file = os.path.join(labels_path, f.replace('.jpg', '.txt'))
    if os.path.isfile(label_file):
        dest_file = os.path.join(dest_path, 'labels', 'train', f.replace('.jpg', '.txt'))
        shutil.copy(label_file, dest_file)

# Move the val images and labels
for f in val_images:
    # Move the image file
    src_file = os.path.join(jpeg_images_path, f)
    dest_file = os.path.join(dest_path, 'images', 'val', f)
    shutil.copy(src_file, dest_file)
    # Move the label file if it exists
    label_file = os.path.join(labels_path, f.replace('.jpg', '.txt'))
    if os.path.isfile(label_file):
        dest_file = os.path.join(dest_path, 'labels', 'val', f.replace('.jpg', '.txt'))
        shutil.copy(label_file, dest_file)

# Move the test images and labels
for f in test_images:
    # Move the image file
    src_file = os.path.join(jpeg_images_path, f)
    dest_file = os.path.join(dest_path, 'images', 'test', f)
    shutil.copy(src_file, dest_file)
    # Move the label file if it exists
    label_file = os.path.join(labels_path, f.replace('.jpg', '.txt'))
    if os.path.isfile(label_file):
        dest_file = os.path.join(dest_path, 'labels', 'test', f.replace('.jpg', '.txt'))
        shutil.copy(label_file, dest_file)