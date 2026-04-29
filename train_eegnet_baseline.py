import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class EEGNet(nn.Module):
    def __init__(self, channels=21, samples=2560, num_classes=2):
        super(EEGNet, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        
        self.conv1 = nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1, False)
        
        self.conv2 = nn.Conv2d(self.F1, self.F1 * self.D, (channels, 1), groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1 * self.D, False)
        self.pooling2 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding=(0, 8), groups=self.F1 * self.D, bias=False)
        self.conv4 = nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.F2, False)
        self.pooling3 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate size after pooling
        out_samples = samples // 32
        self.fc1 = nn.Linear(self.F2 * out_samples, num_classes)

    def forward(self, x):
        # x shape: [batch, channels, samples]
        x = x.unsqueeze(1) # [batch, 1, channels, samples]
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling3(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class BatchedDataset(Dataset):
    def __init__(self, preictal_data, interictal_data, indices, labels):
        self.preictal_data = preictal_data
        self.interictal_data = interictal_data
        self.preictal_size = preictal_data.shape[0]
        self.indices = indices
        self.labels = labels
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        if true_idx < self.preictal_size:
            sample = self.preictal_data[true_idx].astype(np.float32)
        else:
            sample = self.interictal_data[true_idx - self.preictal_size].astype(np.float32)
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def main():
    print("Loading data...")
    X_preictal = np.load('data/X_preictal.npy')
    X_interictal = np.load('data/X_interictal.npy')
    patient_ids_preictal = np.load('data/patient_ids_preictal.npy')
    patient_ids_interictal = np.load('data/patient_ids_interictal.npy')
    
    X_preictal = X_preictal.astype(np.float32)
    y_preictal = np.ones(X_preictal.shape[0], dtype=np.int64)
    y_interictal = np.zeros(X_interictal.shape[0], dtype=np.int64)
    
    all_indices = np.arange(X_preictal.shape[0] + X_interictal.shape[0])
    all_y = np.concatenate([y_preictal, y_interictal])
    all_patients = np.concatenate([patient_ids_preictal, patient_ids_interictal])
    
    print("Splitting data at patient level...")
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(all_indices, all_y, groups=all_patients))
    
    temp_indices = all_indices[temp_idx]
    temp_y = all_y[temp_idx]
    temp_patients = all_patients[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx_relative, test_idx_relative = next(gss2.split(temp_indices, temp_y, groups=temp_patients))
    
    val_idx = temp_indices[val_idx_relative]
    test_idx = temp_indices[test_idx_relative]
    
    y_train = all_y[train_idx]
    y_val = all_y[val_idx]
    y_test = all_y[test_idx]
    
    train_dataset = BatchedDataset(X_preictal, X_interictal, train_idx, y_train)
    val_dataset = BatchedDataset(X_preictal, X_interictal, val_idx, y_val)
    test_dataset = BatchedDataset(X_preictal, X_interictal, test_idx, y_test)
    
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print("Creating EEGNet model...")
    channels = X_preictal.shape[1]
    samples = X_preictal.shape[2]
    model = EEGNet(channels=channels, samples=samples).to(device)
    
    # Calculate class weights
    total_samples = len(y_train)
    num_preictal = sum(y_train == 1)
    num_interictal = sum(y_train == 0)
    weight_for_0 = total_samples / (2.0 * num_interictal)
    weight_for_1 = total_samples / (2.0 * num_preictal)
    
    # Smoothing
    weight_smoothing = 0.8
    weight_for_0 = weight_for_0 * weight_smoothing + (1 - weight_smoothing)
    weight_for_1 = weight_for_1 * weight_smoothing + (1 - weight_smoothing)
    class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("Training EEGNet...")
    best_val_f1 = 0
    for epoch in range(30): # EEGNet converges relatively quickly
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        print(f"Epoch {epoch+1}/30 | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'pretrained_models/eegnet_best.pt')
            
    print("Testing EEGNet...")
    model.load_state_dict(torch.load('pretrained_models/eegnet_best.pt'))
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs)
            
    test_acc = accuracy_score(y_true, y_pred)
    test_prec = precision_score(y_true, y_pred, zero_division=0)
    test_rec = recall_score(y_true, y_pred, zero_division=0)
    test_f1 = f1_score(y_true, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_true, y_prob)
    
    print("\nEEGNet Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUC: {test_auc:.4f}")
    
    os.makedirs('results/training_outputs', exist_ok=True)
    with open('results/training_outputs/eegnet_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Precision: {test_prec:.4f}\n")
        f.write(f"Recall: {test_rec:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"AUC: {test_auc:.4f}\n")

if __name__ == "__main__":
    main()
