import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

class CustomDataset:
    def __init__(self, rgb_dir, contour_dir, transform_rgb=None, transform_contour=None, class_range=None):
        self.rgb_dir = rgb_dir
        self.contour_dir = contour_dir
        self.transform_rgb = transform_rgb
        self.transform_contour = transform_contour
        self.rgb_images = []
        self.contour_images = []

        # Collect images only from the specified class range
        for cls in sorted(os.listdir(rgb_dir))[class_range[0]:class_range[1]]:
            rgb_class_dir = os.path.join(rgb_dir, cls)
            contour_class_dir = os.path.join(contour_dir, cls)
            rgb_files = sorted(os.listdir(rgb_class_dir))
            contour_files = sorted(os.listdir(contour_class_dir))

            for rgb_file in rgb_files:
                contour_file = rgb_file.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.JPG', '.png').replace('.JPEG', '.png')
                if contour_file in contour_files:
                    self.rgb_images.append(os.path.join(rgb_class_dir, rgb_file))
                    self.contour_images.append(os.path.join(contour_class_dir, contour_file))

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = self.rgb_images[idx]
        contour_path = self.contour_images[idx]

        rgb_image = Image.open(rgb_path).convert('RGB')
        contour_image = Image.open(contour_path).convert('L')

        # Apply transformations to contour image
        if self.transform_contour:
            contour_image = self.transform_contour(contour_image)

        # Apply transformations to RGB image (resize and center crop)
        #if self.transform_rgb:
        #    rgb_image = self.transform_rgb(rgb_image)

        rgb_np = np.array(rgb_image)
        lab_image = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2Lab)
        #print(f"Dimensions after RGB to LAB conversion: {lab_image.shape}")  # Should still be (H, W, 3)

        # Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        #print(f"Dimensions of A channel: {a_channel.shape}")  # Should be (H, W)
        #print(f"Dimensions of B channel: {b_channel.shape}")  # Should be (H, W)

        a_channel_tensor = torch.tensor(a_channel, dtype=torch.float32).unsqueeze(0)
        b_channel_tensor = torch.tensor(b_channel, dtype=torch.float32).unsqueeze(0)
        #print(f"Dimensions of A channel tensor: {a_channel_tensor.shape}")  # Should be (1, H, W)
        #print(f"Dimensions of B channel tensor: {b_channel_tensor.shape}")  # Should be (1, H, W)

        # Normalize a and b channels
        a_channel_tensor = a_channel_tensor / 255.0
        b_channel_tensor = b_channel_tensor / 255.0

        image = torch.cat((contour_image, a_channel_tensor, b_channel_tensor), dim=0)
        #print(f"Final 3-channel input dimensions: {image.shape}")
        return image, rgb_path


# Define the transformations
#transform_rgb = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(224)
    # No ToTensor() or normalization here
#])

transform_contour = transforms.Compose([
    transforms.PILToTensor()
])

# Get arguments from command line
start_class = int(sys.argv[1])
end_class = int(sys.argv[2])
rgb_dir = '/tmp/dataset'
contour_dir = '/tmp/contour'
save_dir = '/tmp/tensor'

# Instantiate the dataset
dataset = CustomDataset(rgb_dir, contour_dir, None, transform_contour, class_range=(start_class, end_class))

# Save the tensors to disk
for idx in tqdm(range(len(dataset))):
    tensor, rgb_path = dataset[idx]
    relative_path = os.path.relpath(rgb_path, rgb_dir)
    tensor_save_path = os.path.join(save_dir, relative_path)
    tensor_save_dir = os.path.dirname(tensor_save_path)
    os.makedirs(tensor_save_dir, exist_ok=True)
    tensor_save_path = tensor_save_path.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.JPG', '.pt').replace('.JPEG', '.pt')
    #print(f"Saving tensor {tensor_save_path} with dimensions: {tensor.shape}")
    torch.save(tensor, tensor_save_path)
