import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

class ImagenetteDataset(Dataset):
    def __init__(self, data_path, is_train=True, transform=None):
        self.data_path = data_path
        self.is_train = is_train
        self.transform = transform
        
        self.classes = {
            'n01440764': 0,  # tench
            'n02102040': 1,  # english springer
            'n02979186': 2,  # cassette player
            'n03000684': 3,  # chain saw
            'n03028079': 4,  # church
            'n03394916': 5,  # french horn
            'n03417042': 6,  # garbage truck
            'n03425413': 7,  # gas pump
            'n03445777': 8,  # golf ball
            'n03888257': 9   # parachute
        }

        self.data = []
        if is_train:
            folder = 'train'
        else:
            folder = 'val'
        base_path = os.path.join(self.data_path, folder)

        for classes_folder in self.classes.keys():
            class_path = os.path.join(base_path, classes_folder)
            if os.path.exists(class_path):
                for img in os.listdir(class_path):
                    img_path = os.path.join(class_path, img)
                    self.data.append((img_path, self.classes[class_path]))

    def __getitem__(self, index):
        img_path, label = self.data[index]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(Image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")

    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self, use_batchnorm=False, use_dropout=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout_rate = 0.3
        else:
            self.dropout_rate = 0.0

        # 1st convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_rate) if self.use_dropout else nn.Idetity()
        ) 


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = '/home/danyez87/Master AI/NN/HW1/imagenette2-160'

    # Create a directory to store the plots
    results_directory = 'results'
    os.makedirs(results_directory, exist_ok=True)

    # Create the configurations
    configs = [
        {'name' : 'base model', 'batch_normalization' : False, 'dropout' : False, 'data_augmentation': False},
        {'name' : 'with_batch_normalization', 'batch_normalization' : True, 'dropout' : False, 'data_augmentation': False},
        {'name' : 'with_dropout', 'batch_normalization' : False, 'dropout' : True, 'data_augmentation': False},
        {'name' : 'with_data_augmentation', 'batch_normalization' : False, 'dropout' : False, 'data_augmentation': True},
        {'name' : 'with_all', 'batch_normalization' : True, 'dropout' : True, 'data_augmentation': True}
    ]
    
    results = {}

    for config in configs:
        print(f"\nTraining model with configuration {config['name']}")

        # Base transform for all configurations
        base_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
        
        # When data_augmentation is True
        augmented_transform = transforms.Compose(
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        
        if config['data_augmentation']:
            trainset = ImagenetteDataset(data_path, is_train=True, transform=augmented_transform)
        else:
            trainset = ImagenetteDataset(data_path, is_train=True, transform=base_transform)
        
        testset = ImagenetteDataset(data_path, is_train=False, transform=base_transform)

        trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=32, num_workers= 2)

        model = Net(
            use_batchnorm = config['batch_normalization'],
            use_dropout = config['dropout']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        result = train_model(model, trainloader, testloader, criterion, optimizer, num_epochs = 20, device = device)
        results[config['name']] = result


if __name__ == "__main__":
    main()