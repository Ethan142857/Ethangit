import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, image_file, transform=None, use_image=True):
        self.data_dir = data_dir
        self.image_file = image_file
        self.transform = transform
        self.use_image = use_image
        # Read the CSV file with comma-separated values
        self.data = pd.read_csv(image_file, sep=',', skiprows=1, header=None, names=['guid', 'tag'])
        
    def __len__(self):
        return len(self.data)
    
    def read_image_file(self, file_path):
        try:
            img = Image.open(file_path).convert('RGB')
            return img
        except Exception as e:
            raise ValueError(f"Could not open image file {file_path}: {e}")
    
    def __getitem__(self, idx):
        guid = int(self.data.iloc[idx]['guid'])  # Ensure guid is an integer
        tag_str = self.data.iloc[idx]['tag']
        
        image_path = os.path.join(self.data_dir, f'{guid}.jpg')  # Assuming images are in JPG format
        
        if self.use_image:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = self.read_image_file(image_path)
            if self.transform:
                image = self.transform(image)
        else:
            image = None
            
        # Handle missing labels (for test set)
        if pd.isna(tag_str) or tag_str.strip() == 'null':
            label = -1
        else:
            label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
            label = label_map.get(tag_str.strip(), -1)
        
        return image, label

class ImageOnlyModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ImageOnlyModel, self).__init__()
        try:
            self.resnet = models.resnet50(pretrained=True)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = torch.nn.Linear(num_ftrs, num_classes)
        except Exception as e:
            print(f"Failed to load ResNet model: {e}")
            raise
        
    def forward(self, x):
        logits = self.resnet(x)
        return logits

def collate_fn_image(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    for batch in dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Filter out samples with label -1 for loss calculation
        valid_indices = (labels != -1)
        valid_outputs = outputs[valid_indices]
        valid_labels = labels[valid_indices]
        
        if valid_outputs.numel() > 0 and valid_labels.numel() > 0:
            loss = criterion(valid_outputs, valid_labels)
            _, preds = torch.max(outputs, dim=1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * valid_outputs.size(0)
            correct_predictions += torch.sum(preds[valid_indices] == valid_labels)
        else:
            print("No valid samples in this batch.")
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # Filter out samples with label -1 for loss calculation
            valid_indices = (labels != -1)
            valid_outputs = outputs[valid_indices]
            valid_labels = labels[valid_indices]
            
            if valid_outputs.numel() > 0 and valid_labels.numel() > 0:
                loss = criterion(valid_outputs, valid_labels)
                _, preds = torch.max(outputs, dim=1)
                
                running_loss += loss.item() * valid_outputs.size(0)
                correct_predictions += torch.sum(preds[valid_indices] == valid_labels)
            else:
                print("No valid samples in this batch.")
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

def predict(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch
            images = images.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            
    return predictions

# 数据路径
data_dir = 'data'
train_file = 'train.txt'
test_file = 'test_without_label.txt'

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 超参数
num_epochs = 10
batch_size = 8
learning_rate = 1e-5  # Adjusted learning rate

# 加载数据集
full_dataset = MultiModalDataset(data_dir, train_file, transform=transform, use_image=True)
train_data, val_data = train_test_split(full_dataset.data, test_size=0.2, random_state=42)

train_dataset_full = MultiModalDataset(data_dir, train_file, transform=transform, use_image=True)
val_dataset_full = MultiModalDataset(data_dir, train_file, transform=transform, use_image=True)

train_dataset_full.data = train_data.reset_index(drop=True)
val_dataset_full.data = val_data.reset_index(drop=True)

# 仅图像数据加载器
train_loader_image = DataLoader(
    train_dataset_full,
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn_image
)
val_loader_image = DataLoader(
    val_dataset_full,
    batch_size=batch_size, shuffle=False, collate_fn=collate_fn_image
)

# 测试集加载
test_dataset_full = MultiModalDataset(data_dir, test_file, transform=transform, use_image=True)
test_loader_full = DataLoader(test_dataset_full, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_image)

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 仅图像模型
print("Training Image Only Model...")
model_image = ImageOnlyModel(num_classes=3).to(device)
criterion_image = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer_image = torch.optim.AdamW(model_image.parameters(), lr=learning_rate)

best_val_acc_image = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model_image, train_loader_image, criterion_image, optimizer_image, device)
    val_loss, val_acc = evaluate_model(model_image, val_loader_image, criterion_image, device)
    
    print(f'Image Only Model | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    if val_acc > best_val_acc_image:
        best_val_acc_image = val_acc
        torch.save(model_image.state_dict(), 'best_image_only_model.pth')

# 加载最佳模型并预测测试集
if os.path.exists('best_image_only_model.pth'):
    model_image.load_state_dict(torch.load('best_image_only_model.pth', map_location=device))
    test_loader_image = DataLoader(
        test_dataset_full,
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn_image
    )
    predictions_image = predict(model_image, test_loader_image, device)
    
    # 将标签映射回字符串
    label_map_inv = {0: 'positive', 1: 'neutral', 2: 'negative'}
    predictions_image_str = [label_map_inv[pred] for pred in predictions_image]
    
    # 保存预测结果
    test_df_image = pd.read_csv(test_file, sep=',', skiprows=1, header=None, names=['guid', 'tag'])
    test_df_image['tag'] = predictions_image_str
    test_df_image.to_csv('test_with_label_image_only.txt', sep=',', index=False, header=False)
else:
    print("No best image-only model found. Skipping prediction.")



