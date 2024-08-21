import os
from PIL import Image
import torch
from tqdm import tqdm
import cv2
import numpy as np

# Directory containing subfolders with the RGB images
root_dir = '/tmp/dataset'

n_pixels_a = 0
n_pixels_b = 0
sum_a = 0.0
sum_b = 0.0
sum_squared_a = 0.0
sum_squared_b = 0.0

# Collect all file paths
file_paths = []
for subdir, _, files in os.walk(root_dir):
    for file_name in files:
        if file_name.endswith(('.JPEG', '.jpg', '.jpeg')):
            file_paths.append(os.path.join(subdir, file_name))

for file_path in tqdm(file_paths, desc="Processing images"):
    image = Image.open(file_path).convert('RGB')
    
    image_lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    
    a_tensor = torch.tensor(a_channel, dtype=torch.float32)
    b_tensor = torch.tensor(b_channel, dtype=torch.float32)
    
    a_tensor = a_tensor / 255.0
    b_tensor = b_tensor / 255.0
    a_tensor = a_tensor.flatten()
    b_tensor = b_tensor.flatten()
    
    # Update the number of pixels processed for each channel
    n_a = a_tensor.numel()
    n_b = b_tensor.numel()
    n_pixels_a += n_a
    n_pixels_b += n_b
    
    # Get the sum of pixel values for A and B channels
    sum_a += a_tensor.sum().item()
    sum_b += b_tensor.sum().item()
    
    # Accumulate the sum of squared pixel values for A and B channels
    sum_squared_a += (a_tensor ** 2).sum().item()
    sum_squared_b += (b_tensor ** 2).sum().item()

mean_a = sum_a / n_pixels_a
mean_b = sum_b / n_pixels_b

variance_a = (sum_squared_a / n_pixels_a) - (mean_a ** 2)
variance_b = (sum_squared_b / n_pixels_b) - (mean_b ** 2)

std_a = torch.sqrt(torch.tensor(variance_a)).item()
std_b = torch.sqrt(torch.tensor(variance_b)).item()

print(f'Mean A channel: {mean_a}')
print(f'Variance A channel: {variance_a}')
print(f'Standard Deviation A channel: {std_a}')

print(f'Mean B channel: {mean_b}')
print(f'Variance B channel: {variance_b}')
print(f'Standard Deviation B channel: {std_b}')
