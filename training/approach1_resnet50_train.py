import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from models.resnet50 import ResNet503Channel

# Define dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dir, class_to_idx, transform=None):
        self.tensor_dir = tensor_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.files = []
        self.labels = []

        for class_name, class_idx in class_to_idx.items():
            class_dir = os.path.join(tensor_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.pt'):
                    self.files.append(os.path.join(class_dir, file_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = torch.load(file_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class ApplyTransformToFirstChannel:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, tensor):
        # Split the tensor into three channels
        first_channel = tensor[0, :, :]
        other_channels = tensor[1:, :, :]

        # Apply the provided transform only to the first channel
        first_channel = self.transform(first_channel.unsqueeze(0))

        # Concatenate the transformed first channel back with the other channels
        transformed_tensor = torch.cat((first_channel, other_channels), dim=0)

        return transformed_tensor

def initialize_model_and_datasets(tensor_dir):
    # Create class_to_idx mapping from subfolder names
    classes = sorted(os.listdir(os.path.join(tensor_dir, 'train')))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        #transforms.ConvertImageDtype(torch.float),
        ApplyTransformToFirstChannel(transforms.ConvertImageDtype(torch.float)),
        transforms.Normalize(mean=[0.034, 0.507, 0.533], std=[0.087, 0.049, 0.070])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
        #transforms.ConvertImageDtype(torch.float),
        ApplyTransformToFirstChannel(transforms.ConvertImageDtype(torch.float)),
        transforms.Normalize(mean=[0.034, 0.507, 0.533], std=[0.087, 0.049, 0.070]),
    ])

    # Create datasets
    train_dataset = CustomDataset(tensor_dir=os.path.join(tensor_dir, 'train'), class_to_idx=class_to_idx, transform=train_transform)
    val_dataset = CustomDataset(tensor_dir=os.path.join(tensor_dir, 'val'), class_to_idx=class_to_idx, transform=val_transform)

        # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=3, pin_memory=True)

    # Initialize the model
    model = ResNet503Channel(num_classes=len(classes))

    # Use DataParallel to leverage multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"*****************Count GPU***************** {torch.cuda.device_count()}", )
    model = model.cuda()

    return model, train_loader, val_loader

def save_checkpoint(epoch, model, optimizer, scheduler, save_dir):
    save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(state, save_path)

tensor_dir = '/tmp/tensor'

model, train_loader, val_loader = initialize_model_and_datasets(tensor_dir)

criterion = nn.CrossEntropyLoss().to(torch.device("cuda"))
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training setup
num_epochs = 100

# Training loop
for epoch in range(0, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_err = 100. - (100. * correct / total)
    print(f"Epoch {epoch+1} Training Loss: {epoch_train_loss:.4f} | Training Top-1 Error: {epoch_train_err:.2f}%")


    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_error = 100. - (100.  * correct / total)
    print(f"Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f} | Validation Top-1 Error: {val_error}%")

    # Step the scheduler
    scheduler.step()

    if epoch >= 89:
        checkpoint_path = "/scratch/s5288843/approach1_resnet_2"
        save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")
    elif (epoch + 1) % 2 == 0 and epoch != 0:
        checkpoint_path = "/scratch/s5288843/approach1_resnet_2"
        save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

print(f"Training complete")
