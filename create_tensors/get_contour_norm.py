import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm  

# Directory containing subfolders with the contour map images
root_dir = '/tmp/contour'

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

n_pixels = 0
sum_pixels = 0.0
sum_squared_pixels = 0.0

file_paths = []
for subdir, _, files in os.walk(root_dir):
    for file_name in files:
        if file_name.endswith('.png'):
            file_paths.append(os.path.join(subdir, file_name))

for file_path in tqdm(file_paths, desc="Processing images"):

    image = Image.open(file_path).convert('L')
    tensor = transform(image)
    tensor = tensor.flatten()

    # Update the number of pixels processed
    n = tensor.numel()
    n_pixels += n

    # Get the sum of pixel values
    sum_pixels += tensor.sum().item()

    # Accumulate the sum of squared pixel values
    sum_squared_pixels += (tensor ** 2).sum().item()

mean = sum_pixels / n_pixels
variance = (sum_squared_pixels / n_pixels) - (mean ** 2)
std = torch.sqrt(torch.tensor(variance)).item()

print(f'Mean: for contour map channel {mean}')
print(f'Standard Deviation for contour map channel: {std}')
print(f'Variance for contour map channel: {variance}')
