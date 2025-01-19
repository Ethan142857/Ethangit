import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, text_file, use_text=True):
        self.data_dir = data_dir
        self.text_file = text_file
        self.use_text = use_text
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
        
        if self.use_text:
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"Text file not found: {text_path}")
            text = self.read_text_file(text_path)
        else:
            text = None
            
        # Handle missing labels (for test set)
        if pd.isna(tag_str) or tag_str.strip() == 'null':
            label = -1
        else:
            label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
            label = label_map.get(tag_str.strip(), -1)
        
        return text, label

class TextOnlyModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(TextOnlyModel, self).__init__()
        try:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"Failed to load BERT model: {e}")
            raise
        self.classifier = torch.nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.classifier(text_output)
        return logits

def collate_fn_text(batch):
    texts, labels = zip(*batch)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')
    labels = torch.tensor(labels)
    
    return encoding['input_ids'], encoding['attention_mask'], labels

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
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
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            outputs = model(input_ids, attention_mask)
            
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
            input_ids, attention_mask, _ = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            outputs = model(input_ids, attention_mask)
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
learning_rate = 1e-5  # Adjusted learning rate

# 加载数据集
full_dataset = MultiModalDataset(data_dir, train_file, use_text=True)
train_data, val_data = train_test_split(full_dataset.data, test_size=0.2, random_state=42)

train_dataset_full = MultiModalDataset(data_dir, train_file, use_text=True)
val_dataset_full = MultiModalDataset(data_dir, train_file, use_text=True)

train_dataset_full.data = train_data.reset_index(drop=True)
val_dataset_full.data = val_data.reset_index(drop=True)

# 仅文本数据加载器
train_loader_text = DataLoader(
    train_dataset_full,
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn_text
)
val_loader_text = DataLoader(
    val_dataset_full,
    batch_size=batch_size, shuffle=False, collate_fn=collate_fn_text
)

# 测试集加载
test_dataset_full = MultiModalDataset(data_dir, test_file, use_text=True)
test_loader_full = DataLoader(test_dataset_full, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_text)

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 仅文本模型
print("Training Text Only Model...")
model_text = TextOnlyModel(num_classes=3).to(device)
criterion_text = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer_text = torch.optim.AdamW(model_text.parameters(), lr=learning_rate)

best_val_acc_text = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model_text, train_loader_text, criterion_text, optimizer_text, device)
    val_loss, val_acc = evaluate_model(model_text, val_loader_text, criterion_text, device)
    
    print(f'Text Only Model | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    if val_acc > best_val_acc_text:
        best_val_acc_text = val_acc
        torch.save(model_text.state_dict(), 'best_text_only_model.pth')

# 加载最佳模型并预测测试集
if os.path.exists('best_text_only_model.pth'):
    model_text.load_state_dict(torch.load('best_text_only_model.pth', map_location=device))
    test_loader_text = DataLoader(
        test_dataset_full,
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn_text
    )
    predictions_text = predict(model_text, test_loader_text, device)
    
    # 将标签映射回字符串
    label_map_inv = {0: 'positive', 1: 'neutral', 2: 'negative'}
    predictions_text_str = [label_map_inv[pred] for pred in predictions_text]
    
    # 保存预测结果
    test_df_text = pd.read_csv(test_file, sep=',', skiprows=1, header=None, names=['guid', 'tag'])
    test_df_text['tag'] = predictions_text_str
    test_df_text.to_csv('test_with_label_text_only.txt', sep=',', index=False, header=False)
else:
    print("No best text-only model found. Skipping prediction.")



