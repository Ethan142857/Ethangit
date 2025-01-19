import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, text_file, transform=None):
        self.data_dir = data_dir
        self.text_file = text_file
        self.transform = transform
        # Read the CSV file with comma-separated values
        self.data = pd.read_csv(text_file, sep=',', skiprows=1, header=None, names=['guid', 'tag'])
        
    def __len__(self):
        return len(self.data)
    
    def read_text_file(self, file_path):
        encodings_to_try = ['utf-8', 'gbk', 'big5', 'latin1']
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode file {file_path} with any of the encodings: {encodings_to_try}")
    
    def __getitem__(self, idx):
        guid = int(self.data.iloc[idx]['guid'])  # Ensure guid is an integer
        tag_str = self.data.iloc[idx]['tag']
        
        text_path = os.path.join(self.data_dir, f'{guid}.txt')
        image_path = os.path.join(self.data_dir, f'{guid}.jpg')
        
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found: {text_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        text = self.read_text_file(text_path)
            
        image = plt.imread(image_path)
        if self.transform:
            image = self.transform(image.copy())  # Copy the array to make it writable
            
        # Handle missing labels (for test set)
        if pd.isna(tag_str) or tag_str.strip() == 'null':
            label = -1
        else:
            label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
            label = label_map.get(tag_str.strip(), -1)
        
        return text, image, label

class MultiModalFusionModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(MultiModalFusionModel, self).__init__()
        try:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"Failed to load BERT model: {e}")
            raise
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 256)
        self.fusion_layer = torch.nn.Linear(768 + 256, 256)
        self.classifier = torch.nn.Linear(256, num_classes)
        
    def forward(self, input_ids, attention_mask, image):
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_output = self.resnet(image)
        combined = torch.cat((text_output, image_output), dim=1)
        fused = torch.relu(self.fusion_layer(combined))
        logits = self.classifier(fused)
        return logits

def collate_fn(batch):
    texts, images, labels = zip(*batch)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return encoding['input_ids'], encoding['attention_mask'], images, labels

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    for batch in dataloader:
        input_ids, attention_mask, images, labels = [item.to(device) for item in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
        
        valid_indices = (labels != -1)
        valid_outputs = outputs[valid_indices]
        valid_labels = labels[valid_indices]
        
        if valid_outputs.numel() > 0 and valid_labels.numel() > 0:
            loss = criterion(valid_outputs, valid_labels)
            _, preds = torch.max(outputs, dim=1)
            
            # Debugging: Print outputs and labels
            print(f"Outputs: {outputs}")
            print(f"Labels: {labels}")
            
            # Debugging: Print loss value
            print(f"Loss: {loss.item()}")
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
            input_ids, attention_mask, images, labels = [item.to(device) for item in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            
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
            input_ids, attention_mask, images, _ = [item.to(device) for item in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            
    return predictions

# 数据路径
data_dir = 'data'
train_file = 'train.txt'
test_file = 'test_without_label.txt'

# 超参数
num_epochs = 10
batch_size = 8
learning_rate = 1e-5

# 图像变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
full_dataset = MultiModalDataset(data_dir, train_file, transform=transform)
train_data, val_data = train_test_split(full_dataset.data, test_size=0.2, random_state=42)

train_dataset = MultiModalDataset(data_dir, train_file, transform=transform)
val_dataset = MultiModalDataset(data_dir, train_file, transform=transform)

train_dataset.data = train_data.reset_index(drop=True)
val_dataset.data = val_data.reset_index(drop=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 测试集加载
test_dataset = MultiModalDataset(data_dir, test_file, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalFusionModel(num_classes=3).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练模型
best_val_acc = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# 加载最佳模型并预测测试集
if os.path.exists('best_model.pth'):
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    predictions = predict(model, test_loader, device)
    
    # 将数字标签转换为文本标签
    label_map_inverse = {0: 'positive', 1: 'neutral', 2: 'negative'}
    predictions_text = [label_map_inverse[pred] for pred in predictions]
    
    test_df = pd.read_csv(test_file, sep=',', skiprows=1, header=None, names=['guid', 'tag'])
    test_df['tag'] = predictions_text
    test_df.to_csv('test_with_label.txt', sep=',', index=False, header=False)
else:
    print("No best model found. Skipping prediction.")



