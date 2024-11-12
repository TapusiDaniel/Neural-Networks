import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ImagenetteDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        
        self.classes = {
            'n01440764': 0,  # tench
            'n02102040': 1,  # English springer
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
                return torch.zeros((3, 224, 224)), label
            return Image.new('RGB', (224, 224), 'black'), label

class ModelTrainer:
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.metrics = {
            'frozen': {
                'train_losses': [],
                'val_losses': [],
                'train_accuracies': [],
                'val_accuracies': []
            },
            'unfrozen': {
                'train_losses': [],
                'val_losses': [],
                'train_accuracies': [],
                'val_accuracies': []
            }
        }
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.setup_data_transforms()
        self.load_data()

    def setup_data_transforms(self):
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def load_data(self):
        self.train_dataset = ImagenetteDataset(
            self.data_dir, is_train=True, transform=self.data_transforms['train'])
        self.val_dataset = ImagenetteDataset(
            self.data_dir, is_train=False, transform=self.data_transforms['val'])
        
        self.dataloaders = {
            'train': DataLoader(self.train_dataset, batch_size=self.batch_size,
                              shuffle=True, num_workers=self.num_workers),
            'val': DataLoader(self.val_dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers)
        }
        
        self.dataset_sizes = {
            'train': len(self.train_dataset),
            'val': len(self.val_dataset)
        }
        self.num_classes = 10 

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=20):
        since = time.time()

        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
            torch.save(model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print('-' * 10)

                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    running_loss = 0.0
                    running_corrects = 0
                    all_preds = []
                    all_labels = []

                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                    if phase == 'train':
                        self.train_losses.append(epoch_loss)
                        self.train_accuracies.append(epoch_acc.cpu())
                    else:
                        self.val_losses.append(epoch_loss)
                        self.val_accuracies.append(epoch_acc.cpu())

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
            
        return model, all_preds, all_labels

    def plot_metrics(self):
        # Training metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(self.metrics['frozen']['train_losses'], label='Frozen BN - Training Loss')
        ax1.plot(self.metrics['unfrozen']['train_losses'], label='Unfrozen BN - Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(self.metrics['frozen']['train_accuracies'], label='Frozen BN - Training Accuracy')
        ax2.plot(self.metrics['unfrozen']['train_accuracies'], label='Unfrozen BN - Training Accuracy')
        ax2.set_title('Training Accuracy Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join('results', 'training_metrics.png'))
        plt.close()
        
        # Validation metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(self.metrics['frozen']['val_losses'], label='Frozen BN - Validation Loss')
        ax1.plot(self.metrics['unfrozen']['val_losses'], label='Unfrozen BN - Validation Loss')
        ax1.set_title('Validation Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(self.metrics['frozen']['val_accuracies'], label='Frozen BN - Validation Accuracy')
        ax2.plot(self.metrics['unfrozen']['val_accuracies'], label='Unfrozen BN - Validation Accuracy')
        ax2.set_title('Validation Accuracy Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join('results', 'validation_metrics.png'))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, title_prefix="", ax=None):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=range(self.num_classes))
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'{title_prefix}Confusion Matrix')

def create_model(unfreeze_bn=False, num_classes=10):
    """Create ResNet model with explicit control over layer freezing"""
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    for param in model.parameters():
        param.requires_grad = False
    
    if unfreeze_bn:
        print("Unfreezing BatchNorm layers...")
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                print(f"Unfreezing BatchNorm layer: {module}")
                for param in module.parameters():
                    param.requires_grad = True
                module.track_running_stats = True
                module.training = True
    else:
        print("Keeping BatchNorm layers frozen...")
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.training = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def main():
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    trainer = ModelTrainer('/home/danyez87/Master AI/NN/HW1/imagenette2-160')
    
    print("\nTraining model with frozen BatchNorm layers...")
    model_frozen_bn = create_model(unfreeze_bn=False, num_classes=10)
    model_frozen_bn = model_frozen_bn.to(trainer.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_frozen_bn.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model_frozen_bn, preds_frozen, labels_frozen = trainer.train_model(
        model_frozen_bn, criterion, optimizer, scheduler, num_epochs=20)
    
    trainer.metrics['frozen'] = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accuracies': trainer.train_accuracies,
        'val_accuracies': trainer.val_accuracies
    }
    
    trainer.train_losses = []
    trainer.val_losses = []
    trainer.train_accuracies = []
    trainer.val_accuracies = []
    
    print("\nTraining model with unfrozen BatchNorm layers...")
    model_unfrozen_bn = create_model(unfreeze_bn=True, num_classes=10)
    model_unfrozen_bn = model_unfrozen_bn.to(trainer.device)
    
    optimizer = optim.SGD(model_unfrozen_bn.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model_unfrozen_bn, preds_unfrozen, labels_unfrozen = trainer.train_model(
        model_unfrozen_bn, criterion, optimizer, scheduler, num_epochs=20)
    
    trainer.metrics['unfrozen'] = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accuracies': trainer.train_accuracies,
        'val_accuracies': trainer.val_accuracies
    }
    
    # Plot metrics
    trainer.plot_metrics()

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    trainer.plot_confusion_matrix(labels_frozen, preds_frozen, "Frozen BN - ", ax=ax1)
    trainer.plot_confusion_matrix(labels_unfrozen, preds_unfrozen, "Unfrozen BN - ", ax=ax2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Comparison of Confusion Matrices', fontsize=16, y=0.98)
    plt.savefig(os.path.join('results', 'confusion_matrices.png'), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()