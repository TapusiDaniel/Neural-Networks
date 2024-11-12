import torch
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
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        
        self.classes = {
            'n01440764': 0,  # tench
            'n02102040': 1,  # english springer
            'n02979186': 2,  # cassette player
            'n03000684': 3,  # chain saw
            'n03028079': 4,  # church
            'n03394916': 5,  # French horn
            'n03417042': 6,  # garbage truck
            'n03425413': 7,  # gas pump
            'n03445777': 8,  # golf ball
            'n03888257': 9   # parachute
        }
                
        self.data = []
        folder = 'train' if is_train else 'val'
        base_path = os.path.join(root_dir, folder)
        
        for class_folder in self.classes.keys():
            class_path = os.path.join(base_path, class_folder)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.JPEG', '.jpg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.data.append((img_path, self.classes[class_folder]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            if self.transform:
                return torch.zeros((3, 160, 160)), label
            return Image.new('RGB', (160, 160), 'black'), label

def get_transforms(is_train=True, use_augmentation=False):
    if is_train and use_augmentation:
        return transforms.Compose([
            transforms.Resize((160, 160)),  
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((160, 160)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
class ImagenetteModel(nn.Module):
    def __init__(self, activation='relu', use_batchnorm=False, use_dropout=False):
        super().__init__()
        
        # Selecting the activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Activation must be 'relu', 'leaky_relu', or 'sigmoid'")
        
        self.dropout_rate = 0.3 if use_dropout else 0.0
        
        # 1st layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32) if use_batchnorm else nn.Identity(),
            self.activation,
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_rate) if use_dropout else nn.Identity()
        )
        
        # 2nd layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            self.activation,
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_rate) if use_dropout else nn.Identity()
        )
        
        # 3rd layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batchnorm else nn.Identity(),
            self.activation,
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_rate) if use_dropout else nn.Identity()
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 20, 512),
            nn.BatchNorm1d(512) if use_batchnorm else nn.Identity(),
            self.activation,
            nn.Dropout(self.dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(512, 10)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if isinstance(self.activation, nn.ReLU) or isinstance(self.activation, nn.LeakyReLU):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(self.activation, nn.Sigmoid):
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    train_losses = []
    train_accuracies = []
    test_losses = []      
    test_accuracies = []  
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Test phase
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_running_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        epoch_test_loss = test_running_loss / len(test_loader)
        epoch_test_acc = 100. * test_correct / test_total
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {epoch_train_loss:.4f} '
              f'Train Acc: {epoch_train_acc:.2f}% '
              f'Test Loss: {epoch_test_loss:.4f} '
              f'Test Acc: {epoch_test_acc:.2f}%')
    
    # Confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'final_test_accuracy': epoch_test_acc,
        'confusion_matrix': conf_matrix
    }

# def save_learning_curves(config_name, results, save_dir):
#     # Loss Curves
#     plt.figure(figsize=(10, 6))
#     plt.plot(results['train_losses'], label='Training Loss')
#     plt.plot(results['test_losses'], label='Test Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'{config_name} - Training and Test Loss Over Time')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(os.path.join(save_dir, f'{config_name}_loss_curves.png'))
#     plt.close()
    
#     # Accuracy Curves
#     plt.figure(figsize=(10, 6))
#     plt.plot(results['train_accuracies'], label='Training Accuracy')
#     plt.plot(results['test_accuracies'], label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.title(f'{config_name} - Training and Test Accuracy Over Time')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(os.path.join(save_dir, f'{config_name}_accuracy_curves.png'))
#     plt.close()

def save_confusion_matrix(config_name, conf_matrix, save_dir):
    class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 
                  'church', 'French horn', 'garbage truck', 'gas pump', 
                  'golf ball', 'parachute']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{config_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_name}_confusion_matrix.png'))
    plt.close()

def plot_activation_comparison(results, save_dir):
    plt.figure(figsize=(12, 6))
    
    for config_name, result in results.items():
        plt.plot(result['test_accuracies'], label=config_name.replace('_activation', ''))
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Comparison of Activation Functions')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'activation_functions_comparison.png'))
    plt.close()

    # Loss plot
    plt.figure(figsize=(12, 6))
    
    for config_name, result in results.items():
        plt.plot(result['test_losses'], label=config_name.replace('_activation', ''))
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Comparison of Activation Functions - Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'activation_functions_comparison_loss.png'))
    plt.close()

def create_comparison_plots(results, method_name, save_dir):
    # Training plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(results['basemodel']['train_losses'], label='Base model Loss')
    ax1.plot(results[method_name]['train_losses'], label=f'{method_name.replace("with_", "").title()} Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss: Base model vs {method_name.replace("with_", "").title()}')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(results['basemodel']['train_accuracies'], label='Base model Accuracy')
    ax2.plot(results[method_name]['train_accuracies'], label=f'{method_name.replace("with_", "").title()} Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Training Accuracy: Base model vs {method_name.replace("with_", "").title()}')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'basemodel_vs_{method_name.replace("with_", "").lower()}_train.png'))
    plt.close()

    # Test plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(results['basemodel']['test_losses'], label='Base model Loss')
    ax1.plot(results[method_name]['test_losses'], label=f'{method_name.replace("with_", "").title()} Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Test Loss: Base model vs {method_name.replace("with_", "").title()}')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(results['basemodel']['test_accuracies'], label='Base model Accuracy')
    ax2.plot(results[method_name]['test_accuracies'], label=f'{method_name.replace("with_", "").title()} Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Test Accuracy: Base model vs {method_name.replace("with_", "").title()}')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'basemodel_vs_{method_name.replace("with_", "").lower()}_test.png'))
    plt.close()

def plot_comparison_curves(results, save_dir):
    methods = ['with_batchnorm', 'with_dropout', 'with_augment', 'all_methods']
    
    for method in methods:
        create_comparison_plots(results, method, save_dir)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = '/home/danyez87/Master AI/NN/HW1/imagenette2-160'
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    configs = [
        {'name': 'basemodel', 'activation': 'relu', 'augment': False, 'batchnorm': False, 'dropout': False},
        {'name': 'with_batchnorm', 'activation': 'relu', 'augment': False, 'batchnorm': True, 'dropout': False},
        {'name': 'with_dropout', 'activation': 'relu', 'augment': False, 'batchnorm': False, 'dropout': True},
        {'name': 'with_augment', 'activation': 'relu', 'augment': True, 'batchnorm': False, 'dropout': False},
        {'name': 'all_methods', 'activation': 'relu', 'augment': True, 'batchnorm': True, 'dropout': True}
        #{'name': 'relu_activation', 'activation': 'relu', 'augment': True, 'batchnorm': True, 'dropout': True},
        #{'name': 'leaky_relu_activation', 'activation': 'leaky_relu', 'augment': True, 'batchnorm': True, 'dropout': True},
        #{'name': 'sigmoid_activation', 'activation': 'sigmoid', 'augment': True, 'batchnorm': True, 'dropout': True}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTraining model with configuration: {config['name']}")
        train_transform = get_transforms(is_train=True, use_augmentation=config['augment'])
        test_transform = get_transforms(is_train=False, use_augmentation=False)
        train_dataset = ImagenetteDataset(data_path, is_train=True, transform=train_transform)
        test_dataset = ImagenetteDataset(data_path, is_train=False, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        model = ImagenetteModel(
            activation=config['activation'],
            use_batchnorm=config['batchnorm'],
            use_dropout=config['dropout']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        result = train_model(model, train_loader, test_loader, criterion, optimizer, 
                           num_epochs=20, device=device)
        results[config['name']] = result
        
        # save_learning_curves(config['name'], result, results_dir)
        save_confusion_matrix(config['name'], result['confusion_matrix'], results_dir)

    #plot_activation_comparison(results, results_dir)
    plot_comparison_curves(results, results_dir)
    
    return results

if __name__ == "__main__":
    main()