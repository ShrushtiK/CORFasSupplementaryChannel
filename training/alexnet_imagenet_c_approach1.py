import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import numpy as np
import sys
from torchvision.transforms import InterpolationMode
from models.alexnet import AlexNet3Channel
import os
from torchvision import transforms
print("Python Version:", sys.version)
print("TORCH VERSION:", torch.__version__)

# Load the pre-trained AlexNet model
#net = models.alexnet(pretrained=True)

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



'''
net = models.AlexNet()
net.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
    )
)

net = models.alexnet()
net.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/alexnet-19c8e357.pth"
    )
)
'''

net = AlexNet3Channel(1000)


checkpoint_path = "alexnet_ap1/checkpoint_epoch_96.pth"
checkpoint = torch.load(checkpoint_path)
'''
state_dict = checkpoint['model_state_dict']
new_state_dict = {}

for key in state_dict:
    new_key = key.replace('module.', '')  # Remove the 'module.' prefix
    new_state_dict[new_key] = state_dict[key]

# Load the new state_dict into the model
net.load_state_dict(new_state_dict)
'''

# Load the saved state into the model
net.load_state_dict(checkpoint['model_state_dict'])

test_bs = 128
ngpu = 1
nworker = 9

if ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))

if ngpu > 0:
    net.cuda()

torch.manual_seed(1)
np.random.seed(1)
if ngpu > 0:
    torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders

mean = [0.034, 0.507, 0.533]
std = [0.087, 0.049, 0.070]

val_transform = trn.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
        #transforms.ConvertImageDtype(torch.float),
        ApplyTransformToFirstChannel(transforms.ConvertImageDtype(torch.float)),
        trn.Normalize(mean, std)
    ])

classes = sorted(os.listdir('/tmp/dataset'))
class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

val_dataset = CustomDataset(tensor_dir='/tmp/dataset', class_to_idx=class_to_idx, transform=val_transform)

clean_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=9, pin_memory=True)



correct = 0
with torch.no_grad():
    for data, target in clean_loader:
        if ngpu > 0:
            data = data.cuda()
            target = target.cuda()

        output = net(data)
        _, predicted = output.max(1)
        correct += (predicted == target).sum().item()

clean_error = 1 - correct / len(clean_loader.dataset)
print('Clean dataset error AlexNet (%): {:.4f}'.format(100 * clean_error))

def show_performance(distortion_name):
    errs = []

    for severity in range(1, 6):
        c_transform = trn.Compose([
            trn.CenterCrop(224),
            ApplyTransformToFirstChannel(transforms.ConvertImageDtype(torch.float)),
            #trn.ConvertImageDtype(torch.float),
            trn.Normalize(mean, std)
        ])

        distorted_dataset = CustomDataset(tensor_dir = "/tmp/imagenet_c/" + distortion_name + "/" + str(severity), class_to_idx=class_to_idx, transform=c_transform)
        distorted_dataset_loader = torch.utils.data.DataLoader(distorted_dataset, batch_size=128, shuffle=False, num_workers=9, pin_memory=True)

        correct = 0
        with torch.no_grad():
            for data, target in distorted_dataset_loader:
                if ngpu > 0:
                    data = data.cuda()
                    target = target.cuda()

                output = net(data)
                _, predicted = output.max(1)
                correct += (predicted == target).sum().item()

        print(f"ACC: {distortion_name} {severity}", 1.0 * correct / len(distorted_dataset))
        errs.append(1 - 1.0 * correct / len(distorted_dataset))

    print("\n=Average", tuple(errs))
    return np.mean(errs)

# List of corruption types
distortions = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]

error_rates = []
for distortion_name in distortions:
    rate = show_performance(distortion_name)
    error_rates.append(rate)
    print(
        "Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}".format(
            distortion_name, 100 * rate
        )
    )

print(
    "mCE (unnormalized by AlexNet errors) (%): {:.2f}".format(
        100 * np.mean(error_rates)
    )
)
