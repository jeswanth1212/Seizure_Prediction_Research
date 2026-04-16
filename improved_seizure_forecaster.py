#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Seizure Forecasting Model using Spiking Neural Networks (SNN)
This script implements an SNN-based seizure forecasting model using pretrained encoder
and addresses class imbalance with several advanced techniques.
The model uses spiking neurons for more biologically plausible neural simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# snnTorch
import snntorch as snn
from snntorch import surrogate

# Add tqdm import at the top
from tqdm.auto import tqdm

# Set seed and use GPU
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)  # Use first GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print('CUDA not available, using CPU')

# EEG Dataset class
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Focal Loss for imbalanced classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Improved Spiking CNN Encoder with better architecture
class SpikingCNNEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128):
        super(SpikingCNNEncoder, self).__init__()
        
        # Configuration
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        
        # Spiking neuron parameters
        beta = 0.5  # Decay rate
        threshold = 1.0  # Firing threshold
        spike_grad = surrogate.fast_sigmoid(slope=25)  # Surrogate gradient
        
        # Initial convolution block
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Conv blocks with residual connections
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.lif2_1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.lif2_2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
        self.downsample2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(64)
        )
        
        # Global average pooling and projection
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embedding_dim)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        
    def forward(self, x, num_steps=25):
        batch_size = x.shape[0]
        
        # Initialize membrane potentials
        mem_1 = self.lif1.init_leaky()
        mem_2_1 = self.lif2_1.init_leaky()
        mem_2_2 = self.lif2_2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        # Output accumulator
        spk_out = torch.zeros((batch_size, self.embedding_dim), device=x.device)
        
        # Simulate network for num_steps time steps
        for step in range(num_steps):
            # Initial convolution block
            conv1 = self.conv1(x)
            conv1 = self.bn1(conv1)
            spk_1, mem_1 = self.lif1(conv1, mem_1)
            spk_1 = self.pool1(spk_1)
            
            # ResNet-style Block 2
            conv2_1 = self.conv2_1(spk_1)
            conv2_1 = self.bn2_1(conv2_1)
            spk_2_1, mem_2_1 = self.lif2_1(conv2_1, mem_2_1)
            
            conv2_2 = self.conv2_2(spk_2_1)
            conv2_2 = self.bn2_2(conv2_2)
            
            # Residual connection
            downsample2 = self.downsample2(spk_1)
            conv2_2 = conv2_2 + downsample2
            
            spk_2_2, mem_2_2 = self.lif2_2(conv2_2, mem_2_2)
            
            # Global average pooling
            out = self.gap(spk_2_2)
            out = out.view(out.size(0), -1)  # Flatten
            
            # Projection
            out = self.fc(out)
            
            # Final output neuron
            spk, mem_out = self.lif_out(out, mem_out)
            
            # Accumulate output spikes
            spk_out += spk
            
        # Return average output spike activity over all time steps
        return spk_out / num_steps

# Update the SimCLR model with correct dimensions in conv4 (256 channels)
class PretrainedSimCLR(nn.Module):
    def __init__(self, in_channels, pretrained_path):
        super(PretrainedSimCLR, self).__init__()
        self.embedding_dim = 64  # Final output dimension
        self.pretrained_path = pretrained_path
        self.in_channels = in_channels
        
        # Create the exact structure matching the pretrained model
        self.encoder = nn.Module()
        
        # First convolutional layer with 32 filters (from error message)
        self.encoder.conv1 = nn.Sequential(
            nn.Conv1d(self.in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        self.encoder.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-style blocks with correct dimensions
        self.encoder.conv2 = nn.ModuleList([
            self._create_resblock(32, 64),  # First block: 32->64
            self._create_resblock(64, 64)   # Second block: 64->64
        ])
        
        self.encoder.conv3 = nn.ModuleList([
            self._create_resblock(64, 128), # First block: 64->128
            self._create_resblock(128, 128) # Second block: 128->128
        ])
        
        # Conv4 should be 128->256 not 128->128 according to error message
        self.encoder.conv4 = nn.ModuleList([
            self._create_resblock(128, 256), # First block: 128->256
            self._create_resblock(256, 256)  # Second block: 256->256
        ])
        
        self.encoder.avgpool = nn.AdaptiveAvgPool1d(1)
        # FC layer should take 256 input
        self.encoder.fc = nn.Linear(256, 128)
        
        # Create projection head with correct dimensions
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Load the pretrained model
        print(f"Loading SimCLR pretrained model from {pretrained_path}")
        self.pretrained_weights = torch.load(pretrained_path, map_location=device)
        
        # Try to directly load weights with special handling to avoid errors
        try:
            self._load_pretrained_weights()
        except Exception as e:
            print(f"Warning: Error during weight loading: {e}")
            print("Continuing with partial weight loading...")
    
    def _load_pretrained_weights(self):
        """Special function to load weights directly"""
        print("Loading SimCLR weights with special handling...")
        
        try:
            # Try direct loading first with non-strict mode
            missing, unexpected = self.load_state_dict(self.pretrained_weights, strict=False)
            print(f"Initial load completed with {len(missing)} missing and {len(unexpected)} unexpected params")
            
            # Report troublesome parameters
            if len(missing) > 0:
                print(f"Missing keys sample: {missing[:5]}")
            if len(unexpected) > 0:
                print(f"Unexpected keys sample: {unexpected[:5]}")
            
            # Now load parameters selectively, only those with matching shapes
            print("Performing selective parameter loading...")
            model_dict = self.state_dict()
            
            # Filter the pretrained weights to only include items with matching shapes
            pretrained_dict = {k: v for k, v in self.pretrained_weights.items() 
                            if k in model_dict and v.shape == model_dict[k].shape}
            
            # Update the model with compatible parameters
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            
            print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
            
        except Exception as e:
            print(f"Error during weight loading: {e}")
            print("Using parameter-by-parameter loading as fallback...")
            
            # Fallback to extremely cautious loading
            loaded = 0
            total = len(self.state_dict())
            
            for name, param in self.named_parameters():
                try:
                    if name in self.pretrained_weights and self.pretrained_weights[name].shape == param.shape:
                        param.data.copy_(self.pretrained_weights[name])
                        loaded += 1
                except Exception as ex:
                    print(f"Error loading {name}: {ex}")
                    
            print(f"Fallback loading: loaded {loaded}/{total} parameters")
    
    def _create_resblock(self, in_channels, out_channels):
        """Create a ResNet block with the right structure"""
        block = nn.Module()
        
        block.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        block.bn1 = nn.BatchNorm1d(out_channels)
        block.relu = nn.ReLU(inplace=True)
        
        block.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        block.bn2 = nn.BatchNorm1d(out_channels)
        
        # Downsample if needed
        if in_channels != out_channels:
            block.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        return block
        
    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.maxpool(x)
        
        # ResNet blocks
        for block in self.encoder.conv2:
            identity = x
            x = block.conv1(x)
            x = block.bn1(x)
            x = block.relu(x)
            
            x = block.conv2(x)
            x = block.bn2(x)
            
            if hasattr(block, 'downsample'):
                identity = block.downsample(identity)
                
            x += identity
            x = block.relu(x)
        
        for block in self.encoder.conv3:
            identity = x
            x = block.conv1(x)
            x = block.bn1(x)
            x = block.relu(x)
            
            x = block.conv2(x)
            x = block.bn2(x)
            
            if hasattr(block, 'downsample'):
                identity = block.downsample(identity)
                
            x += identity
            x = block.relu(x)
            
        for block in self.encoder.conv4:
            identity = x
            x = block.conv1(x)
            x = block.bn1(x)
            x = block.relu(x)
            
            x = block.conv2(x)
            x = block.bn2(x)
            
            if hasattr(block, 'downsample'):
                identity = block.downsample(identity)
                
            x += identity
            x = block.relu(x)
        
        x = self.encoder.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder.fc(x)
        
        # Projection head
        embeddings = self.projection(x)
        
        return embeddings, x

# Pretrained Encoder model with exact architecture
class PretrainedEncoder(nn.Module):
    def __init__(self, in_channels, pretrained_path):
        super(PretrainedEncoder, self).__init__()
        self.embedding_dim = 128
        self.pretrained_path = pretrained_path
        self.in_channels = in_channels
        
        # Examine the pretrained weights to determine exact architecture
        print(f"Loading encoder pretrained model from {pretrained_path}")
        self.pretrained_weights = torch.load(pretrained_path, map_location=device)
        
        # Create the EXACT structure matching the pretrained model
        # Initial conv with 32 filters (not 64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # First ResNet block: 32->64
        self.conv2 = nn.ModuleList()
        
        # First block: 32->64
        block1 = nn.Module()
        block1.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        block1.bn1 = nn.BatchNorm1d(64)
        block1.relu1 = nn.ReLU(inplace=True)
        block1.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        block1.bn2 = nn.BatchNorm1d(64)
        block1.relu2 = nn.ReLU(inplace=True)
        block1.downsample = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(64)
        )
        self.conv2.append(block1)
        
        # Second block: 64->64
        block2 = nn.Module()
        block2.conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        block2.bn1 = nn.BatchNorm1d(64)
        block2.relu1 = nn.ReLU(inplace=True)
        block2.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        block2.bn2 = nn.BatchNorm1d(64)
        block2.relu2 = nn.ReLU(inplace=True)
        self.conv2.append(block2)
        
        # Second ResNet block: 64->128
        self.conv3 = nn.ModuleList()
        
        # First block: 64->128
        block1 = nn.Module()
        block1.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        block1.bn1 = nn.BatchNorm1d(128)
        block1.relu1 = nn.ReLU(inplace=True)
        block1.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        block1.bn2 = nn.BatchNorm1d(128)
        block1.relu2 = nn.ReLU(inplace=True)
        block1.downsample = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(128)
        )
        self.conv3.append(block1)
        
        # Second block: 128->128
        block2 = nn.Module()
        block2.conv1 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        block2.bn1 = nn.BatchNorm1d(128)
        block2.relu1 = nn.ReLU(inplace=True)
        block2.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        block2.bn2 = nn.BatchNorm1d(128)
        block2.relu2 = nn.ReLU(inplace=True)
        self.conv3.append(block2)
        
        # Third ResNet block: 128->256
        self.conv4 = nn.ModuleList()
        
        # First block: 128->256
        block1 = nn.Module()
        block1.conv1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        block1.bn1 = nn.BatchNorm1d(256)
        block1.relu1 = nn.ReLU(inplace=True)
        block1.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        block1.bn2 = nn.BatchNorm1d(256)
        block1.relu2 = nn.ReLU(inplace=True)
        block1.downsample = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(256)
        )
        self.conv4.append(block1)
        
        # Second block: 256->256
        block2 = nn.Module()
        block2.conv1 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        block2.bn1 = nn.BatchNorm1d(256)
        block2.relu1 = nn.ReLU(inplace=True)
        block2.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        block2.bn2 = nn.BatchNorm1d(256)
        block2.relu2 = nn.ReLU(inplace=True)
        self.conv4.append(block2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 128)
        
        # Load the pretrained weights with perfect matching
        try:
            # Print some keys to understand the structure
            print("First few keys in pretrained weights:")
            keys = list(self.pretrained_weights.keys())
            for k in keys[:10]:
                print(f"{k}: {self.pretrained_weights[k].shape}")
                
            self._load_pretrained_weights()
        except Exception as e:
            print(f"Warning: Error during weight loading: {e}")
            print("Continuing with partial weight loading...")
            
            # Fallback to extremely cautious loading
            loaded = 0
            total = len(self.state_dict())
            
            for name, param in self.named_parameters():
                try:
                    if name in self.pretrained_weights and self.pretrained_weights[name].shape == param.shape:
                        param.data.copy_(self.pretrained_weights[name])
                        loaded += 1
                except Exception as ex:
                    print(f"Error loading {name}: {ex}")
                    
            print(f"Fallback loading: loaded {loaded}/{total} parameters")
    
    def _load_pretrained_weights(self):
        """Special function to load weights with perfect matching"""
        print("Loading encoder weights with perfect matching...")
        
        # First try direct loading with non-strict mode to see what's missing
        missing, unexpected = self.load_state_dict(self.pretrained_weights, strict=False)
        
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"Initial check: {len(missing)} missing and {len(unexpected)} unexpected params")
            
            # Get our model's state dict
            model_dict = self.state_dict()
            
            # Print some keys from both dictionaries for debugging
            print("First 5 keys in pretrained weights:", list(self.pretrained_weights.keys())[:5])
            print("First 5 keys in model state dict:", list(model_dict.keys())[:5])
            
            # Check for key name differences
            pretrained_keys = set(self.pretrained_weights.keys())
            model_keys = set(model_dict.keys())
            
            # Find keys that exist in both dictionaries
            common_keys = pretrained_keys.intersection(model_keys)
            print(f"Found {len(common_keys)} common keys")
            
            # Create a new state dict with only matching keys and shapes
            matched_dict = {}
            for k in common_keys:
                if self.pretrained_weights[k].shape == model_dict[k].shape:
                    matched_dict[k] = self.pretrained_weights[k]
                else:
                    print(f"Shape mismatch for {k}: pretrained {self.pretrained_weights[k].shape} vs model {model_dict[k].shape}")
            
            # Load the matched parameters
            self.load_state_dict(matched_dict, strict=False)
            print(f"Loaded {len(matched_dict)}/{len(model_dict)} parameters with matching shapes")
            
            # Try to fix any remaining mismatches by direct parameter assignment
            for name, param in self.named_parameters():
                if name in self.pretrained_weights and name not in matched_dict:
                    try:
                        # Try to adapt the parameter shape if possible
                        pretrained_param = self.pretrained_weights[name]
                        if param.dim() == pretrained_param.dim():
                            print(f"Attempting to adapt parameter {name}")
                            # For each dimension, copy what we can
                            if all(a <= b for a, b in zip(pretrained_param.shape, param.shape)):
                                # Pretrained is smaller, we can copy it directly into a subset
                                param_view = param.view(*pretrained_param.shape)
                                param_view.copy_(pretrained_param)
                                print(f"Adapted parameter {name}")
                    except Exception as e:
                        print(f"Failed to adapt {name}: {e}")
        else:
            print("Perfect match! All parameters loaded successfully.")
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        for block in self.conv2:
            identity = x
            x = block.conv1(x)
            x = block.bn1(x)
            x = block.relu1(x)
            
            x = block.conv2(x)
            x = block.bn2(x)
            
            if hasattr(block, 'downsample'):
                identity = block.downsample(identity)
                
            x += identity
            x = block.relu2(x)
        
        for block in self.conv3:
            identity = x
            x = block.conv1(x)
            x = block.bn1(x)
            x = block.relu1(x)
            
            x = block.conv2(x)
            x = block.bn2(x)
            
            if hasattr(block, 'downsample'):
                identity = block.downsample(identity)
                
            x += identity
            x = block.relu2(x)
            
        for block in self.conv4:
            identity = x
            x = block.conv1(x)
            x = block.bn1(x)
            x = block.relu1(x)
            
            x = block.conv2(x)
            x = block.bn2(x)
            
            if hasattr(block, 'downsample'):
                identity = block.downsample(identity)
                
            x += identity
            x = block.relu2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x

# Update CombinedPretrained to match correct dimensions
class CombinedPretrained(nn.Module):
    def __init__(self, simclr_model, encoder_model):
        super(CombinedPretrained, self).__init__()
        self.simclr_model = simclr_model
        self.encoder_model = encoder_model
        self.embedding_dim = 64 + 128  # SimCLR (64) + Encoder (128) = 192
        
        # Add in_channels attribute for compatibility with downstream models
        if hasattr(simclr_model, 'in_channels'):
            self.in_channels = simclr_model.in_channels
        elif hasattr(encoder_model, 'in_channels'):
            self.in_channels = encoder_model.in_channels
        else:
            # Default to standard EEG channels if can't be determined
            self.in_channels = 21
        
    def forward(self, x):
        # Get embeddings from both models
        simclr_embeddings, _ = self.simclr_model(x)
        encoder_embeddings = self.encoder_model(x)
        
        # Combine the embeddings
        combined = torch.cat([simclr_embeddings, encoder_embeddings], dim=1)
        return combined

# SNN-based Seizure Classifier to replace the ANN-based version
class SNNCombinedSeizureClassifier(nn.Module):
    def __init__(self, combined_encoder, hidden_size=192, dropout=0.6):
        super(SNNCombinedSeizureClassifier, self).__init__()
        self.encoder = combined_encoder
        self.embedding_dim = combined_encoder.embedding_dim
        
        # Set up the spiking neuron parameters
        beta = 0.5  # Decay rate
        spike_grad = surrogate.fast_sigmoid(slope=25)  # Surrogate gradient
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout)
        
        # First SNN layer
        self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Second SNN layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Third SNN layer
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size // 4, 2)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x, num_steps=25):
        # Get ANN embeddings from the encoder first
        embeddings = self.encoder(x)
        
        # Apply dropout
        embeddings = self.dropout1(embeddings)
        
        # Initialize output accumulator
        spk_out = torch.zeros((x.size(0), 2), device=x.device)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        # Simulate spiking neurons for multiple time steps
        for step in range(num_steps):
            # First layer
            cur = self.fc1(embeddings)
            cur = self.bn1(cur)
            spk1, mem1 = self.lif1(cur, mem1)
            
            # Second layer
            cur = self.fc2(spk1)
            cur = self.bn2(cur)
            spk2, mem2 = self.lif2(cur, mem2)
            
            # Third layer
            cur = self.fc3(spk2)
            cur = self.bn3(cur)
            spk3, mem3 = self.lif3(cur, mem3)
            
            # Output layer
            cur = self.fc_out(spk3)
            spk, mem_out = self.lif_out(cur, mem_out)
            
            # Accumulate output spikes
            spk_out += spk
        
        # Return average spike rate
        return spk_out / num_steps

# Enhanced SNN with Attention Mechanism
class AttentionEnhancedSNN(nn.Module):
    def __init__(self, combined_encoder, hidden_size=192, dropout=0.5):
        super(AttentionEnhancedSNN, self).__init__()
        self.encoder = combined_encoder
        self.embedding_dim = combined_encoder.embedding_dim
        
        # Get number of input channels from the encoder model
        # The CombinedPretrained doesn't have in_channels attribute directly, 
        # but we can get it from one of its component models
        if hasattr(combined_encoder, 'simclr_model') and hasattr(combined_encoder.simclr_model, 'in_channels'):
            in_channels = combined_encoder.simclr_model.in_channels
        elif hasattr(combined_encoder, 'encoder_model') and hasattr(combined_encoder.encoder_model, 'in_channels'):
            in_channels = combined_encoder.encoder_model.in_channels
        else:
            # Default to 21 channels if we can't determine it from the model
            print("Warning: Could not determine number of input channels from model. Using default of 21.")
            in_channels = 21
            
        # Add frequency feature dimension to embedding (simpler approach)
        freq_feature_dim = 5 * in_channels  # 5 bands per channel
        self.freq_proj = nn.Linear(freq_feature_dim, 32)  # Smaller projection
        self.total_dim = self.embedding_dim + 32
        
        # Lightweight attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim),
            nn.Tanh(),
            nn.Linear(self.total_dim, 1)
        )
        
        # Enhanced SNN layers with stronger gradients but fewer parameters
        beta = 0.5  # Decay rate
        spike_grad = surrogate.fast_sigmoid(slope=30)
        
        # More efficient network architecture
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.total_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.dropout2 = nn.Dropout(dropout * 0.8)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_out = nn.Linear(hidden_size // 2, 2)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def compute_attention(self, x):
        """Apply simple attention to the input"""
        # Simple, efficient attention mechanism
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = x * attn_weights
        return context.sum(dim=1)
        
    def forward(self, x, freq_features=None, num_steps=50):  # Reduced from 100 to 50
        # Get embeddings from encoder
        embeddings = self.encoder(x)
        batch_size = embeddings.size(0)
        
        # Process frequency features if provided
        if freq_features is None:
            # Extract frequency features on-the-fly
            in_channels = x.shape[1]  # Get number of channels from input directly
            freq_features = torch.zeros((batch_size, 5 * in_channels), device=x.device)
            for i in range(batch_size):
                freq_feats = extract_frequency_features(x[i].cpu().numpy())
                freq_features[i] = torch.tensor(freq_feats, device=x.device)
        
        # Project frequency features
        freq_proj = self.freq_proj(freq_features)
        
        # Combine embeddings with frequency features
        combined_features = torch.cat([embeddings, freq_proj], dim=1)
        
        # Apply dropout
        features = self.dropout1(combined_features)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        # Initialize output accumulator
        spk_out = torch.zeros((batch_size, 2), device=x.device)
        
        # Simulate for fewer time steps
        for step in range(num_steps):
            # Layer 1
            cur = self.fc1(features)
            cur = self.bn1(cur)
            spk1, mem1 = self.lif1(cur, mem1)
            spk1 = self.dropout2(spk1)
            
            # Layer 2
            cur = self.fc2(spk1)
            cur = self.bn2(cur)
            spk2, mem2 = self.lif2(cur, mem2)
            
            # Output layer
            cur = self.fc_out(spk2)
            spk, mem_out = self.lif_out(cur, mem_out)
            
            # Accumulate output spikes
            spk_out += spk
            
        # Return average spike rate
        return spk_out / num_steps

# Original ANN-based classifier kept for backward compatibility
class CombinedSeizureClassifier(nn.Module):
    def __init__(self, combined_encoder, hidden_size=192, dropout=0.6):
        super(CombinedSeizureClassifier, self).__init__()
        self.encoder = combined_encoder
        self.embedding_dim = combined_encoder.embedding_dim
        
        # Add stronger dropout and batch normalization for regularization
        self.dropout1 = nn.Dropout(dropout)
        
        # Add a more complex classifier with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_size // 4, 2)
        )
        
    def forward(self, x):
        # Get combined embeddings from encoder
        embeddings = self.encoder(x)
        
        # Apply dropout
        embeddings = self.dropout1(embeddings)
        
        # Apply classifier
        outputs = self.classifier(embeddings)
        
        return outputs

# Ensemble of Multiple SNN Models
class SNNEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        super(SNNEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Initialize weights if not provided (equal weighting)
        if weights is None:
            weights = torch.ones(len(models))
            
        # Normalize weights
        self.register_buffer('weights', weights / weights.sum())
        
    def forward(self, x, num_steps=100):
        """Forward pass through ensemble of models
        
        Args:
            x: Input data
            num_steps: Number of simulation steps for SNN models
            
        Returns:
            Combined output from all models
        """
        outputs = []
        
        # Get output from each model
        for i, model in enumerate(self.models):
            # Check if model has num_steps parameter (SNN) or not (ANN)
            if isinstance(model, (SNNCombinedSeizureClassifier, AttentionEnhancedSNN)):
                outputs.append(model(x, num_steps) * self.weights[i])
            else:
                outputs.append(model(x) * self.weights[i])
                
        # Average predictions
        return sum(outputs)
    
    def train_models(self):
        """Set all models to training mode"""
        for model in self.models:
            model.train()
    
    def eval_models(self):
        """Set all models to evaluation mode"""
        for model in self.models:
            model.eval()

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
        
    return metrics

# Function to extract frequency domain features
def extract_frequency_features(eeg_segment, fs=256.0):
    """Extract frequency domain features from EEG segment
    
    Args:
        eeg_segment: EEG segment of shape [channels, samples]
        fs: Sampling frequency
        
    Returns:
        Array of frequency domain features
    """
    from scipy import signal
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    features = []
    
    # For each channel
    for ch in range(eeg_segment.shape[0]):
        ch_data = eeg_segment[ch]
        
        # Calculate power spectral density
        freqs, psd = signal.welch(ch_data, fs, nperseg=min(512, len(ch_data)))
        
        # Extract band powers
        band_powers = {}
        for band, (low, high) in bands.items():
            # Find indices corresponding to frequency band
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Calculate average power in band
            band_powers[band] = np.mean(psd[idx_band]) if np.any(idx_band) else 0
            
        # Add normalized band powers
        total_power = sum(band_powers.values())
        if total_power > 0:
            for band in bands:
                features.append(band_powers[band] / total_power)
        else:
            # If no power, add zeros
            for _ in bands:
                features.append(0.0)
    
    return np.array(features)

# Function to train the model with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
               num_epochs=100, num_steps=100, patience=15, monitor='f1', is_pretrained=False):
    """Train the model with early stopping based on validation metrics"""
    best_val_metric = 0.0 if monitor != 'loss' else float('inf')
    is_better = lambda x, y: x > y if monitor != 'loss' else x < y
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_f1': [], 'val_f1': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_auc': [], 'val_auc': []
    }
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(total=num_epochs, desc="Training Progress", position=0)
    
    for epoch in range(num_epochs):
        print("-" * 50)
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train, y_prob_train = [], [], []
        
        # Create progress bar for batches
        batch_pbar = tqdm(total=len(train_loader), desc=f"Training", position=1, leave=False)
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass depends on model type
            if is_pretrained:
                outputs = model(inputs)
            else:
                outputs = model(inputs, num_steps)
                
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            
            y_true_train.extend(targets.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())
            y_prob_train.extend(probs)
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
            batch_pbar.update(1)
        
        batch_pbar.close()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(y_true_train, y_pred_train, y_prob_train)
        
        # Validation
        model.eval()
        running_loss = 0.0
        y_true_val, y_pred_val, y_prob_val = [], [], []
        
        with torch.no_grad():
            val_pbar = tqdm(total=len(val_loader), desc="Validation", position=1, leave=False)
            
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass depends on model type
                if is_pretrained:
                    outputs = model(inputs)
                else:
                    outputs = model(inputs, num_steps)
                    
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                
                y_true_val.extend(targets.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())
                y_prob_val.extend(probs)
                
                val_pbar.update(1)
                
            val_pbar.close()
        
        val_loss = running_loss / len(val_loader.dataset)
        val_metrics = calculate_metrics(y_true_val, y_pred_val, y_prob_val)
        
        # Print statistics with more detailed metrics
        print(f"Train - Loss: {train_loss:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"AUC: {train_metrics.get('auc', 0):.4f}")
        
        print(f"Val - Loss: {val_loss:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics.get('auc', 0):.4f}")
        
        # Update history with all metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['train_auc'].append(train_metrics.get('auc', 0))
        history['val_auc'].append(val_metrics.get('auc', 0))
        
        # Early stopping check
        val_metric_value = val_metrics[monitor] if monitor != 'loss' else val_loss
        
        if is_better(val_metric_value, best_val_metric):
            best_val_metric = val_metric_value
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Best model saved with val_{monitor}: {val_metric_value:.4f}")
        else:
            patience_counter += 1
            print(f"EarlyStopping: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metric_value)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
        
        # Update epoch progress bar
        epoch_pbar.update(1)
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            val_f1=f"{val_metrics['f1']:.4f}"
        )
    
    epoch_pbar.close()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def prepare_seizure_data(X_preictal, X_interictal):
    """Prepare data for training with balancing techniques"""
    print(f"Preparing data - preictal: {X_preictal.shape}, interictal: {X_interictal.shape}")
    
    print("1. Converting preictal data to float32...")
    # Convert to float32 to save memory
    X_preictal = X_preictal.astype(np.float32)
    
    print("2. Processing data in batches to avoid memory error...")
    # Process interictal samples in batches to avoid memory error
    batch_size = 10000  # Process 10K samples at a time
    num_batches = (X_interictal.shape[0] + batch_size - 1) // batch_size
    
    print("3. Creating labels...")
    # Create labels
    y_preictal = np.ones(X_preictal.shape[0], dtype=np.int64)
    y_interictal = np.zeros(X_interictal.shape[0], dtype=np.int64)
    
    print("4. Creating dataset indices...")
    # Create dataset indices first
    all_indices = np.arange(X_preictal.shape[0] + X_interictal.shape[0])
    all_y = np.concatenate([y_preictal, y_interictal])
    
    print("5. Splitting data into train/val/test sets...")
    # Split indices first
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        all_indices, all_y, test_size=0.3, random_state=SEED, stratify=all_y)
    
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
    
    print(f"Training set: {len(train_idx)} samples")
    print(f"Validation set: {len(val_idx)} samples")
    print(f"Test set: {len(test_idx)} samples")
    
    # Create custom dataset classes that load data on-demand
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
            
            # Check if it's a preictal or interictal sample
            if true_idx < self.preictal_size:
                # It's a preictal sample
                sample = self.preictal_data[true_idx].astype(np.float32)
            else:
                # It's an interictal sample - adjust index
                sample = self.interictal_data[true_idx - self.preictal_size].astype(np.float32)
            
            # Convert to tensor
            sample_tensor = torch.tensor(sample, dtype=torch.float32)
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return sample_tensor, label_tensor
    
    # Create custom datasets
    train_dataset = BatchedDataset(X_preictal, X_interictal, train_idx, y_train)
    val_dataset = BatchedDataset(X_preictal, X_interictal, val_idx, y_val)
    test_dataset = BatchedDataset(X_preictal, X_interictal, test_idx, y_test)
    
    return train_dataset, val_dataset, test_dataset

def main():
    """Main training function"""
    # Ensure output directory exists
    os.makedirs('results/training_outputs', exist_ok=True)
    OUT = 'results/training_outputs'

    try:
        print("Loading data files (this may take a while)...")
        print("Loading data/X_preictal.npy...")
        X_preictal = np.load('data/X_preictal.npy')
        print(f"Loaded X_preictal: {X_preictal.shape}")
        
        print("Loading data/X_interictal.npy...")
        X_interictal = np.load('data/X_interictal.npy')
        print(f"Loaded X_interictal: {X_interictal.shape}")
    except FileNotFoundError:
        print("Error: Data files not found. Please run seizure_forecasting_pipeline.ipynb first.")
        return
    except KeyboardInterrupt:
        print("\nData loading interrupted. Try again or check your data files.")
        return
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = prepare_seizure_data(X_preictal, X_interictal)
    
    # Create dataloaders
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Step 1: Load pretrained encoder models
    in_channels = X_preictal.shape[1]  # Number of EEG channels
    
    print("Step 1: Loading pretrained encoder models...")
    
    # Verify that both files exist
    if not os.path.exists('pretrained_models/simclr_pretrained.pt'):
        raise FileNotFoundError("pretrained_models/simclr_pretrained.pt not found!")
    
    if not os.path.exists('pretrained_models/encoder_pretrained.pt'):
        raise FileNotFoundError("pretrained_models/encoder_pretrained.pt not found!")
    
    # Step 1.1: Load SimCLR model with exact architecture matching
    print("Loading SimCLR model with exact architecture matching...")
    simclr_model = PretrainedSimCLR(in_channels, 'pretrained_models/simclr_pretrained.pt')
    simclr_model = simclr_model.to(device)
    
    # Step 1.2: Load Encoder model with exact architecture matching
    print("Loading Encoder model with exact architecture matching...")
    encoder_model = PretrainedEncoder(in_channels, 'pretrained_models/encoder_pretrained.pt')
    encoder_model = encoder_model.to(device)
    
    # Step 1.3: Create combined model using both pretrained models
    print("Creating combined model using both pretrained models together...")
    combined_model = CombinedPretrained(simclr_model, encoder_model)
    combined_model = combined_model.to(device)
    
    # Step 2: Create the efficient enhanced SNN model
    print("Step 2: Creating lightweight enhanced SNN model...")
    
    # Create a single enhanced model instead of an ensemble
    model = AttentionEnhancedSNN(combined_model, hidden_size=192, dropout=0.5)
    model = model.to(device)
    
    # Set to pretrained mode for the training function
    is_pretrained = True
    
    # Calculate class weights for the complete dataset with slightly adjusted weighting
    num_preictal = X_preictal.shape[0]
    num_interictal = X_interictal.shape[0]
    total_samples = num_preictal + num_interictal
    
    # Calculate class weights manually based on inverse frequency
    weight_for_0 = total_samples / (2.0 * num_interictal)  # weight for interictal class
    weight_for_1 = total_samples / (2.0 * num_preictal)    # weight for preictal class
    
    # Add weight smoothing to prevent overemphasizing rare class
    weight_smoothing = 0.8
    weight_for_0 = weight_for_0 * weight_smoothing + (1 - weight_smoothing)
    weight_for_1 = weight_for_1 * weight_smoothing + (1 - weight_smoothing)
    
    class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32).to(device)
    print(f"Using class weights: {class_weights} to balance the dataset")
    
    # Create focal loss with class weights
    criterion = FocalLoss(alpha=0.75, gamma=2.0, weight=class_weights)
    
    # Create optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # Use cosine annealing scheduler for better convergence without restarts
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Train model with early stopping
    print("Step 3: Training lightweight enhanced SNN model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,   # Back to original number of epochs
        num_steps=50,     # Moderate increase in simulation steps
        patience=15,      # Standard patience
        monitor='f1',
        is_pretrained=is_pretrained
    )
    
    # Save the model
    os.makedirs('trained_models', exist_ok=True)
    torch.save(model.state_dict(), 'trained_models/lightweight_enhanced_snn_model.pt')
    print("Lightweight Enhanced SNN Model saved to trained_models/lightweight_enhanced_snn_model.pt")
    
    # Evaluate on test set
    print("Step 4: Evaluating on test set...")
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for inputs, targets in test_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Use SNN model for inference with moderate num_steps
            outputs = model(inputs, num_steps=50)
            
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs)
    
    # Calculate test metrics
    test_metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    print("\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Interictal', 'Preictal'])
    plt.yticks([0, 1], ['Interictal', 'Preictal'])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{OUT}/confusion_matrix.png')
    print(f"Saved confusion matrix to {OUT}/confusion_matrix.png")
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'{OUT}/roc_curve.png')
    print(f"Saved ROC curve to {OUT}/roc_curve.png")
    
    # Save training history to CSV
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{OUT}/training_history.csv', index=False)
    print(f"Training history saved to {OUT}/training_history.csv")
    
    # Save metrics to file
    with open(f'{OUT}/test_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {test_metrics['f1']:.4f}\n")
        f.write(f"AUC: {test_metrics['auc']:.4f}\n")
    print(f"Test metrics saved to {OUT}/test_metrics.txt")
    
    # Additional visualizations
    
    # 1. Training History Plots
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # F1 Score plot
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Training F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # AUC plot
    plt.subplot(2, 2, 4)
    plt.plot(history['train_auc'], label='Training AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/training_history.png', dpi=300)
    print(f"Saved training history plots to {OUT}/training_history.png")
    
    # 2. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'{OUT}/precision_recall_curve.png', dpi=300)
    print(f"Saved Precision-Recall curve to {OUT}/precision_recall_curve.png")
    
    # 3. Class Distribution Visualization
    plt.figure(figsize=(10, 6))
    class_names = ['Interictal', 'Preictal']
    class_counts = [len(y_true) - sum(y_true), sum(y_true)]
    plt.bar(class_names, class_counts, color=['blue', 'orange'])
    plt.title('Test Set Class Distribution')
    plt.ylabel('Count')
    plt.grid(axis='y')
    for i, count in enumerate(class_counts):
        plt.text(i, count + 5, str(count), ha='center')
    plt.savefig(f'{OUT}/class_distribution.png', dpi=300)
    print(f"Saved class distribution to {OUT}/class_distribution.png")
    
    # 4. Prediction Confidence Histogram
    plt.figure(figsize=(12, 6))
    
    # Separate probabilities for correct and incorrect predictions
    correct_indices = np.where(np.array(y_pred) == np.array(y_true))[0]
    incorrect_indices = np.where(np.array(y_pred) != np.array(y_true))[0]
    
    correct_probs = [y_prob[i] if y_pred[i] == 1 else 1-y_prob[i] for i in correct_indices]
    incorrect_probs = [y_prob[i] if y_pred[i] == 1 else 1-y_prob[i] for i in incorrect_indices]
    
    plt.subplot(1, 2, 1)
    plt.hist(correct_probs, bins=20, alpha=0.7, color='green', label='Correct Predictions')
    plt.title('Confidence Distribution - Correct Predictions')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(incorrect_probs) > 0:
        plt.hist(incorrect_probs, bins=20, alpha=0.7, color='red', label='Incorrect Predictions')
        plt.title('Confidence Distribution - Incorrect Predictions')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No incorrect predictions", ha='center', va='center')
        plt.title('Confidence Distribution - Incorrect Predictions')
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/confidence_distribution.png', dpi=300)
    print(f"Saved confidence distribution to {OUT}/confidence_distribution.png")
    
    # 5. Detailed Classification Report
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, target_names=['Interictal', 'Preictal'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{OUT}/classification_report.csv')
    print(f"Saved detailed classification report to {OUT}/classification_report.csv")
    
    # 6. Save all results as a comprehensive report
    from datetime import datetime
    
    with open(f'{OUT}/evaluation_report.md', 'w') as f:
        f.write(f"# Lightweight Enhanced Seizure Forecasting Model Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Dataset Information\n\n")
        f.write(f"- Preictal samples: {X_preictal.shape[0]}\n")
        f.write(f"- Interictal samples: {X_interictal.shape[0]}\n")
        f.write(f"- Class imbalance ratio: {X_interictal.shape[0] / X_preictal.shape[0]:.2f}\n")
        f.write(f"- Input channels: {in_channels}\n\n")
        
        f.write(f"## Lightweight Enhanced Model Architecture\n\n")
        f.write(f"- Feature extractors: Combined SimCLR and Encoder models\n")
        f.write(f"- Simple attention mechanism on feature embeddings\n")
        f.write(f"- Frequency domain features: 5 frequency bands per channel\n")
        f.write(f"- SNN simulation steps: 50 (moderate increase from original 25)\n")
        f.write(f"- Optimized network depth for efficiency\n\n")
        
        f.write(f"## Training Parameters\n\n")
        f.write(f"- Optimizer: AdamW (lr=5e-5, weight_decay=1e-4)\n")
        f.write(f"- Scheduler: CosineAnnealingLR (T_max=20)\n")
        f.write(f"- Loss function: Focal Loss (alpha=0.75, gamma=2.0)\n")
        f.write(f"- Class weights: [{weight_for_0:.4f}, {weight_for_1:.4f}]\n")
        f.write(f"- Early stopping: patience=15, monitor='f1'\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n\n")
        
        f.write(f"## Test Results\n\n")
        f.write(f"- Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"- Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"- Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {test_metrics['f1']:.4f}\n")
        f.write(f"- AUC: {test_metrics['auc']:.4f}\n\n")
        
        f.write(f"## Generated Visualizations\n\n")
        f.write(f"1. [Confusion Matrix](confusion_matrix.png)\n")
        f.write(f"2. [ROC Curve](roc_curve.png)\n")
        f.write(f"3. [Precision-Recall Curve](precision_recall_curve.png)\n")
        f.write(f"4. [Training History](training_history.png)\n")
        f.write(f"5. [Class Distribution](class_distribution.png)\n")
        f.write(f"6. [Confidence Distribution](confidence_distribution.png)\n\n")
        
        f.write(f"## Confusion Matrix\n\n")
        f.write(f"```\n{cm}\n```\n\n")
        
        f.write(f"## Classification Report\n\n")
        f.write(f"```\n{classification_report(y_true, y_pred, target_names=['Interictal', 'Preictal'])}\n```\n\n")
        
        f.write(f"## Conclusion\n\n")
        if test_metrics['accuracy'] > 0.95:
            f.write(f"The lightweight enhanced model successfully achieves the target accuracy of >95% with {test_metrics['accuracy']:.2%} accuracy, while keeping computational requirements moderate.\n")
        elif test_metrics['f1'] > 0.7:
            f.write(f"The model shows strong performance in seizure forecasting with high F1 score and AUC, approaching but not quite reaching the target 95% accuracy.\n")
        else:
            f.write(f"The model shows moderate performance in seizure forecasting. Further improvements could help reach the target accuracy of 95%.\n")
        
        # Add improvement suggestions regardless of performance
        f.write(f"\n## Potential Future Improvements\n\n")
        f.write(f"1. Experiment with additional frequency domain features\n")
        f.write(f"2. Fine-tune SNN parameters (beta, threshold) for better temporal dynamics\n")
        f.write(f"3. Optimize attention mechanism for better feature focus\n")
        f.write(f"4. Apply moderate data augmentation to improve generalization\n")
    
    print(f"Comprehensive evaluation report saved to {OUT}/evaluation_report.md")

if __name__ == "__main__":
    main() 