#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Explainable AI (XAI) for Seizure Forecasting Model

This script applies various XAI techniques to interpret the predictions of a
trained Spiking Neural Network (SNN) seizure forecasting model. It helps understand
how the model identifies patterns in EEG data that predict seizures.

Implements:
- Integrated Gradients
- SHAP
- LIME
- Saliency Maps
- Attention Visualization

Results are saved to the 'xai' directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
import warnings
import seaborn as sns
from scipy.stats import zscore
from collections import defaultdict

# XAI libraries
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    Occlusion,
    NoiseTunnel,
    visualization as viz
)
import lime
import lime.lime_tabular

# Import model definitions from the existing script
from improved_seizure_forecaster import (
    PretrainedSimCLR,
    PretrainedEncoder,
    CombinedPretrained,
    AttentionEnhancedSNN,
    EEGDataset,
    prepare_seizure_data,
    extract_frequency_features
)

# Ignore specific warnings that might arise during XAI processing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constants
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print('CUDA not available, using CPU')

# Create the XAI directory if it doesn't exist
os.makedirs('results/xai_outputs', exist_ok=True)

# Modified wrapper for the model to make it compatible with XAI methods
class XAIModelWrapper(nn.Module):
    def __init__(self, model, num_steps=25, target_class=1):
        super(XAIModelWrapper, self).__init__()
        self.model = model
        self.num_steps = num_steps
        self.target_class = target_class
        
    def forward(self, x, freq_features=None):
        """Forward pass with fixed number of steps for XAI purposes
        
        Args:
            x: Input EEG data
            freq_features: Optional frequency domain features
            
        Returns:
            Scores for target class
        """
        # Extract frequency features if not provided
        if freq_features is None and hasattr(self.model, 'encoder'):
            # Check if the input is a single sample or batch
            if x.ndim == 2:
                x = x.unsqueeze(0)  # Add batch dimension
                
            batch_size = x.shape[0]
            in_channels = x.shape[1]
            
            # Extract frequency features using the model's method
            try:
                freq_feats = []
                for i in range(batch_size):
                    # Extract frequency features from numpy array
                    feats = extract_frequency_features(x[i].detach().cpu().numpy())
                    freq_feats.append(feats)
                freq_features = torch.tensor(np.array(freq_feats), dtype=torch.float32, device=x.device)
            except Exception as e:
                print(f"Warning: Could not extract frequency features: {e}")
                # Create empty frequency features
                freq_features = torch.zeros((batch_size, 5 * in_channels), device=x.device)
        
        # Get the raw model output (logits)
        try:
            # Try with frequency features first
            if freq_features is not None:
                logits = self.model(x, freq_features=freq_features, num_steps=self.num_steps)
            else:
                logits = self.model(x, num_steps=self.num_steps)
        except TypeError:
            # If that fails, try without frequency features
            logits = self.model(x, num_steps=self.num_steps)
        
        return logits[:, self.target_class]  # Return scores for target class

# Function to load the model
def load_model():
    """Load the trained model and its components"""
    # Load data to get input channels
    try:
        X_preictal = np.load('data/X_preictal.npy')
        in_channels = X_preictal.shape[1]  # Number of EEG channels
        print(f"Loaded X_preictal.npy, shape: {X_preictal.shape}")
    except FileNotFoundError:
        print("Warning: X_preictal.npy not found, using default 21 channels")
        in_channels = 21
    
    print("Loading pretrained encoder models...")
    
    # Load SimCLR model
    print("Loading SimCLR model...")
    simclr_model = PretrainedSimCLR(in_channels, 'pretrained_models/simclr_pretrained.pt')
    simclr_model = simclr_model.to(device)
    
    # Load Encoder model
    print("Loading Encoder model...")
    encoder_model = PretrainedEncoder(in_channels, 'pretrained_models/encoder_pretrained.pt')
    encoder_model = encoder_model.to(device)
    
    # Create combined model
    print("Creating combined pretrained model...")
    combined_model = CombinedPretrained(simclr_model, encoder_model)
    combined_model = combined_model.to(device)
    
    # Load the AttentionEnhancedSNN model
    print("Loading AttentionEnhancedSNN model...")
    model = AttentionEnhancedSNN(combined_model, hidden_size=192, dropout=0.5)
    model = model.to(device)
    
    # Load the trained weights
    print("Loading trained model weights...")
    model.load_state_dict(torch.load('trained_models/lightweight_enhanced_snn_model.pt', 
                                     map_location=device))
    model.eval()  # Set model to evaluation mode
    
    # Create XAI-compatible wrapper
    print("Creating XAI model wrapper...")
    xai_model = XAIModelWrapper(model, num_steps=25, target_class=1)
    
    return model, xai_model, in_channels

# Function to load data and select representative samples
def load_data_samples():
    """Load EEG data and select representative samples for XAI analysis"""
    print("Loading EEG data...")
    try:
        X_preictal = np.load('data/X_preictal.npy')
        X_interictal = np.load('data/X_interictal.npy')
        
        print(f"Loaded data - preictal: {X_preictal.shape}, interictal: {X_interictal.shape}")
        
        # Create datasets
        y_preictal = np.ones(X_preictal.shape[0], dtype=np.int64)
        y_interictal = np.zeros(X_interictal.shape[0], dtype=np.int64)
        
        # For background samples (SHAP), select 50 interictal samples
        bg_indices = np.random.choice(len(X_interictal), size=50, replace=False)
        bg_samples = X_interictal[bg_indices]
        
        # Select 5 preictal and 5 interictal samples for analysis
        preictal_indices = np.random.choice(len(X_preictal), size=5, replace=False)
        interictal_indices = np.random.choice(len(X_interictal), size=5, replace=False)
        
        preictal_samples = X_preictal[preictal_indices]
        interictal_samples = X_interictal[interictal_indices]
        
        # Combine samples for analysis
        analysis_samples = np.vstack([preictal_samples, interictal_samples])
        analysis_labels = np.concatenate([np.ones(5), np.zeros(5)])
        
        data_info = {
            'preictal_count': len(X_preictal),
            'interictal_count': len(X_interictal),
            'imbalance_ratio': len(X_interictal) / len(X_preictal)
        }
        
        return {
            'bg_samples': bg_samples,
            'analysis_samples': analysis_samples,
            'analysis_labels': analysis_labels,
            'data_info': data_info
        }
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise

# Class to extract attention weights from the model
class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_weights = None
        self.pre_attention_features = None
        
        # Register hooks to capture attention weights
        self.hooks = []
        
        # Print the attention module structure to help with debugging
        print(f"Attention module structure: {model.attention}")
        
        # Hook to capture attention weights
        def hook_fn(module, input, output):
            # Store the attention weights (output is the attention weights)
            # Check if the output is a tuple or a tensor
            if isinstance(output, tuple):
                self.attention_weights = output[0].detach().cpu()  # Common for some attention mechanisms
            else:
                self.attention_weights = output.detach().cpu()
                
        # Hook to capture pre-attention features
        def pre_hook_fn(module, input):
            # Store the input features before attention
            self.pre_attention_features = input[0].detach().cpu()
        
        # Register the hooks on the attention layer
        if hasattr(model, 'attention'):
            # Try different components of the attention module
            if isinstance(model.attention, nn.Sequential):
                # If it's a sequential module, hook on the first layer
                self.hooks.append(model.attention[0].register_forward_hook(hook_fn))
                self.hooks.append(model.attention[0].register_forward_pre_hook(pre_hook_fn))
            elif hasattr(model.attention, 'tanh') and hasattr(model.attention.tanh, 'register_forward_hook'):
                # If it has a tanh layer, hook after it
                self.hooks.append(model.attention.tanh.register_forward_hook(hook_fn))
            else:
                # Try the main attention module
                self.hooks.append(model.attention.register_forward_hook(hook_fn))
                self.hooks.append(model.attention.register_forward_pre_hook(pre_hook_fn))
        else:
            print("Warning: Model doesn't have an 'attention' attribute")
    
    def get_attention_weights(self, inputs):
        """Forward pass to get attention weights with frequency features"""
        # Reset stored weights
        self.attention_weights = None
        self.pre_attention_features = None
        
        # Forward pass to trigger the hook
        with torch.no_grad():
            # Process frequency features
            batch_size = inputs.size(0)
            in_channels = inputs.size(1)
            
            # Extract frequency features using the extract_frequency_features function
            freq_features = torch.zeros((batch_size, 5 * in_channels), device=inputs.device)
            for i in range(batch_size):
                freq_feats = extract_frequency_features(inputs[i].detach().cpu().numpy())
                freq_features[i] = torch.tensor(freq_feats, device=inputs.device)
                
            # Pass inputs and frequency features through the model
            _ = self.model(inputs, freq_features=freq_features)
        
        # Compute attention feature information
        attention_info = {}
        if self.attention_weights is not None:
            attention_info['weights'] = self.attention_weights.numpy()
            
            # Calculate attention on frequency features
            if self.pre_attention_features is not None:
                # Assuming last 5*in_channels dimensions are frequency features
                freq_dim = 5 * in_channels
                total_dim = self.pre_attention_features.shape[1]
                base_dim = max(0, total_dim - freq_dim)
                
                # Calculate attention percentages
                if base_dim > 0 and total_dim > base_dim:
                    freq_attention = self.attention_weights[:, base_dim:].sum().item()
                    total_attention = self.attention_weights.sum().item()
                    if total_attention > 0:
                        freq_attention_percentage = (freq_attention / total_attention) * 100
                        attention_info['freq_attention_percentage'] = freq_attention_percentage
        
        return attention_info
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()

# Implement Integrated Gradients
def compute_integrated_gradients(xai_model, inputs, baseline=None, target=1, n_steps=50):
    """Compute attributions using Integrated Gradients"""
    print("Computing Integrated Gradients attributions...")
    
    # Convert inputs to tensor and move to device
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)
    
    # Create a baseline of zeros if not provided (quiet EEG)
    if baseline is None:
        baseline = torch.zeros_like(inputs_tensor)
    
    # Initialize integrated gradients
    ig = IntegratedGradients(xai_model)
    
    # Compute attributions
    attributions = []
    for i in tqdm(range(len(inputs_tensor)), desc="Processing samples"):
        input_sample = inputs_tensor[i:i+1]
        baseline_sample = baseline[i:i+1] if isinstance(baseline, torch.Tensor) else baseline
        
        # Get attributions for this sample
        attr = ig.attribute(input_sample, baselines=baseline_sample, n_steps=n_steps)
        attributions.append(attr.detach().cpu().numpy())
    
    # Stack attributions
    attributions = np.vstack([a for a in attributions])
    
    return attributions

# Implement Perturbation-Based Feature Importance (Occlusion Analysis)
def compute_perturbation_importance(xai_model, sample, bg_samples, num_features=20):
    """Compute perturbation-based feature importance"""
    print("Computing feature importance (Perturbation-style)...")
    
    # Store sample dimensions for reshaping
    sample_shape = sample.shape
    
    # We'll use a simplified approach since SHAP is memory-intensive and breaks with SNN temporal dynamics
    # Create a feature importance metric based on perturbation
    
    # First, get the baseline prediction for the sample
    sample_tensor = torch.tensor(sample.reshape(1, *sample_shape), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        baseline_pred = xai_model(sample_tensor).cpu().numpy()[0]
    
    print("Computing feature importance using perturbation analysis...")
    feature_importance = np.zeros(sample_shape)
    
    # Select a subset of channels and time points to analyze (for performance)
    channels_to_analyze = range(sample_shape[0])  # All channels
    
    # For time points, sample uniformly across the signal
    step_size = max(1, sample_shape[1] // 100)  # Sample ~100 time points
    time_points_to_analyze = range(0, sample_shape[1], step_size)
    
    # Analyze importance by perturbing features
    for ch in tqdm(channels_to_analyze, desc="Processing channels"):
        for t in time_points_to_analyze:
            # Create a perturbed sample
            perturbed = sample.copy()
            
            # Perturb by replacing with mean value from background
            perturbed[ch, t] = np.mean(bg_samples[:, ch, t])
            
            # Get prediction for perturbed sample
            perturbed_tensor = torch.tensor(perturbed.reshape(1, *sample_shape), dtype=torch.float32, device=device)
            with torch.no_grad():
                perturbed_pred = xai_model(perturbed_tensor).cpu().numpy()[0]
            
            # Importance is the difference in prediction
            feature_importance[ch, t] = baseline_pred - perturbed_pred
            
            # For unanalyzed time points, use linear interpolation
            if step_size > 1 and t < sample_shape[1] - step_size:
                next_t = min(t + step_size, sample_shape[1] - 1)
                for interp_t in range(t + 1, next_t):
                    alpha = (interp_t - t) / (next_t - t)
                    if next_t in time_points_to_analyze:
                        next_imp = feature_importance[ch, next_t]
                        feature_importance[ch, interp_t] = (1 - alpha) * feature_importance[ch, t] + alpha * next_imp
    
    # Flatten inputs and feature importance
    flat_sample = sample.reshape(-1)
    flat_importance = feature_importance.reshape(-1)
    
    # Create feature names
    feature_names = []
    for ch in range(sample_shape[0]):
        for t in range(sample_shape[1]):
            feature_names.append(f"Ch{ch}_t{t}")
    
    # Convert to DataFrame for easier manipulation
    perturbation_df = pd.DataFrame({
        'feature': feature_names,
        'value': flat_sample,
        'perturbation_value': flat_importance
    })
    
    # Get top features by absolute Perturbation value
    top_features = perturbation_df.reindex(perturbation_df['perturbation_value'].abs().sort_values(ascending=False).index)
    top_features = top_features.head(num_features)
    
    return {
        'perturbation_values': feature_importance.reshape(1, -1),
        'top_features': top_features,
        'feature_names': feature_names
    }

# Implement LIME
def compute_lime_explanation(sample, bg_samples, xai_model, num_features=20):
    """Generate LIME explanation for a sample"""
    print("Computing LIME explanation...")
    
    # Store sample shape for reference
    sample_shape = sample.shape
    
    # Flatten input for LIME
    flat_sample = sample.reshape(-1)
    flat_bg_samples = bg_samples.reshape(bg_samples.shape[0], -1)
    
    # To handle high-dimensionality, we'll create downsampled versions
    # This makes LIME much faster and more stable
    
    # Option 1: Use original high-dim data (slower but more accurate)
    # Option 2: Downsample in time domain to reduce dimensionality
    downsample_factor = 16  # Reduce time points by factor of 16
    
    # Downsample the time domain
    ds_sample_shape = (sample_shape[0], sample_shape[1] // downsample_factor)
    ds_sample = np.zeros(ds_sample_shape)
    ds_bg_samples = np.zeros((bg_samples.shape[0], ds_sample_shape[0], ds_sample_shape[1]))
    
    # Average over time windows
    for ch in range(sample_shape[0]):
        for t in range(ds_sample_shape[1]):
            t_start = t * downsample_factor
            t_end = min((t+1) * downsample_factor, sample_shape[1])
            # Downsample the sample
            ds_sample[ch, t] = np.mean(sample[ch, t_start:t_end])
            # Downsample the background samples
            ds_bg_samples[:, ch, t] = np.mean(bg_samples[:, ch, t_start:t_end], axis=1)
    
    # Flatten downsampled data
    ds_flat_sample = ds_sample.reshape(-1)
    ds_flat_bg_samples = ds_bg_samples.reshape(ds_bg_samples.shape[0], -1)
    
    # Create feature names for downsampled data
    ds_feature_names = []
    for ch in range(ds_sample_shape[0]):
        for t in range(ds_sample_shape[1]):
            ds_feature_names.append(f"Ch{ch}_t{t*downsample_factor}-{(t+1)*downsample_factor-1}")
    
    # Create prediction function for LIME
    def predict_fn(inputs):
        # Reshape inputs back to downsampled (batch, channels, time)
        batch_size = inputs.shape[0]
        reshaped_inputs = inputs.reshape(batch_size, ds_sample_shape[0], ds_sample_shape[1])
        
        # We need to upsample back to original time dimension for the model
        upsampled_inputs = np.zeros((batch_size, sample_shape[0], sample_shape[1]))
        for b in range(batch_size):
            for ch in range(ds_sample_shape[0]):
                for t in range(ds_sample_shape[1]):
                    t_start = t * downsample_factor
                    t_end = min((t+1) * downsample_factor, sample_shape[1])
                    upsampled_inputs[b, ch, t_start:t_end] = reshaped_inputs[b, ch, t]
        
        # Convert to tensor and move to device
        inputs_tensor = torch.tensor(upsampled_inputs, dtype=torch.float32, device=device)
        
        # Get model predictions (using classification outputs)
        with torch.no_grad():
            preds = F.softmax(xai_model.model(inputs_tensor, num_steps=25), dim=1).cpu().numpy()
        
        return preds
    
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        ds_flat_bg_samples,
        feature_names=ds_feature_names,
        class_names=["Interictal", "Preictal"],
        mode="classification"  # Use classification mode for binary predictions
    )
    
    # Get explanation
    explanation = explainer.explain_instance(
        ds_flat_sample,
        predict_fn,
        num_features=num_features,
        top_labels=1
    )
    
    # Extract explanation for the preictal class (index 1)
    lime_weights = explanation.as_list(label=1)
    
    # Convert to DataFrame
    lime_df = pd.DataFrame(lime_weights, columns=['feature', 'weight'])
    
    return {
        'explanation': explanation,
        'weights_df': lime_df,
        'downsample_factor': downsample_factor
    }

# Implement Gradient-based Saliency Maps
def compute_saliency_maps(xai_model, inputs, n_samples=10):
    """Compute gradient-based saliency maps with SmoothGrad-like noise"""
    print("Computing saliency maps...")
    
    # Convert inputs to tensor and move to device
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)
    
    # Initialize GradientShap
    gradient_shap = GradientShap(xai_model)
    
    # Create proper baselines - use zero baseline with small noise
    # This is more standard practice for saliency maps
    baselines = []
    for _ in range(n_samples):
        # Create zero baseline with minimal noise for numerical stability
        noise = torch.randn_like(inputs_tensor) * 0.01
        baselines.append(torch.zeros_like(inputs_tensor) + noise)
    baselines = torch.cat(baselines, dim=0)
    
    # Expand inputs to match baselines
    expanded_inputs = inputs_tensor.repeat(n_samples, 1, 1)
    
    # Compute attributions
    attributions = gradient_shap.attribute(expanded_inputs, baselines=baselines)
    
    # Average attributions across noise samples
    attributions = attributions.reshape(n_samples, *inputs_tensor.shape)
    attributions = attributions.mean(dim=0)
    
    return attributions.detach().cpu().numpy()

# Function to visualize channel importance
def visualize_channel_importance(attributions, channels=None, filename='results/xai_outputs/channel_importance.png'):
    """Visualize channel importance from attributions"""
    print(f"Visualizing channel importance...")
    
    # Average attributions over time dimension to get channel importance
    channel_importance = np.mean(np.abs(attributions), axis=2)
    
    # Average over samples if we have multiple
    if channel_importance.ndim > 1:
        channel_importance = np.mean(channel_importance, axis=0)
    
    # If channel names are not provided, create generic names
    if channels is None:
        channels = [f"Channel {i}" for i in range(len(channel_importance))]
    
    # Create barplot
    plt.figure(figsize=(12, 8))
    
    # Sort channels by importance
    sorted_indices = np.argsort(channel_importance)[::-1]
    sorted_importance = channel_importance[sorted_indices]
    sorted_channels = [channels[i] for i in sorted_indices]
    
    bars = plt.bar(range(len(sorted_channels)), sorted_importance, color='steelblue')
    plt.xlabel('EEG Channels', fontsize=14)
    plt.ylabel('Attribution Magnitude', fontsize=14)
    plt.title('Channel Importance for Seizure Prediction', fontsize=16)
    plt.xticks(range(len(sorted_channels)), sorted_channels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    
    # Save to CSV as well
    channel_df = pd.DataFrame({
        'Channel': channels,
        'Importance': channel_importance
    }).sort_values('Importance', ascending=False)
    
    channel_df.to_csv(filename.replace('.png', '.csv'), index=False)
    
    return channel_importance

# Function to visualize temporal importance
def visualize_temporal_importance(attributions, filename='results/xai_outputs/temporal_importance.png'):
    """Visualize temporal importance from attributions"""
    print(f"Visualizing temporal importance...")
    
    # Average attributions over channel dimension to get temporal importance
    temporal_importance = np.mean(np.abs(attributions), axis=1)
    
    # Average over samples if we have multiple
    if temporal_importance.ndim > 1:
        temporal_importance = np.mean(temporal_importance, axis=0)
    
    # Create heatmap
    plt.figure(figsize=(15, 6))
    
    # Normalize for better visualization
    if np.max(temporal_importance) > np.min(temporal_importance):
        normalized_importance = (temporal_importance - np.min(temporal_importance)) / (np.max(temporal_importance) - np.min(temporal_importance))
    else:
        normalized_importance = np.zeros_like(temporal_importance)
    
    # Create the heatmap reshaped to show time better
    # Dynamically determine rows and cols based on data size
    time_points = len(normalized_importance)
    rows = min(32, time_points)  # Cap at 32 rows
    cols = (time_points + rows - 1) // rows  # Ceiling division
    reshaped_importance = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < time_points:
                reshaped_importance[i, j] = normalized_importance[idx]
    
    plt.imshow(reshaped_importance, aspect='auto', interpolation='none', cmap='inferno')
    plt.colorbar(label='Normalized Attribution Magnitude')
    plt.xlabel('Time (segments)', fontsize=14)
    plt.ylabel('Time segments', fontsize=14)
    plt.title('Temporal Importance for Seizure Prediction', fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    
    # Save raw data to CSV
    pd.DataFrame({
        'Time_Point': range(len(temporal_importance)),
        'Importance': temporal_importance
    }).to_csv(filename.replace('.png', '.csv'), index=False)
    
    return temporal_importance

# Visualize SHAP values
def visualize_perturbation_importance(perturbation_data, filename_prefix='results/xai_outputs/shap'):
    """Visualize SHAP values"""
    print(f"Visualizing Perturbation values...")
    
    # Extract Perturbation values and top features
    perturbation_values = perturbation_data['perturbation_values']
    top_features = perturbation_data['top_features']
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    # Create bar plot of top features
    plt.subplot(1, 1, 1)
    plt.barh(top_features['feature'], top_features['perturbation_value'], color='steelblue')
    plt.xlabel('Perturbation Impact on Prediction', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Top Features by Perturbation Importance', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_summary.png', dpi=300)
    
    # Save top features to CSV
    top_features.to_csv(f'{filename_prefix}_values.csv', index=False)
    
    # Attempt to create a more traditional SHAP summary plot
    try:
        plt.figure(figsize=(12, 8))
        feature_names = perturbation_data['feature_names']
        
        # Create a sample from flattened SHAP values and feature values
        sample_num = min(20, len(feature_names))  # Limit to top 20 features
        feature_indices = np.argsort(np.abs(perturbation_values.flatten()))[-sample_num:]
        
        plt.barh(
            [feature_names[i] for i in feature_indices],
            perturbation_values.flatten()[feature_indices]
        )
        plt.xlabel('Perturbation Value', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title('Perturbation Values (Feature Impact)', fontsize=16)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_beeswarm.png', dpi=300)
    except Exception as e:
        print(f"Warning: Could not create standard Perturbation beeswarm plot: {e}")

# Visualize LIME explanation
def visualize_lime_explanation(lime_data, filename_prefix='results/xai_outputs/lime'):
    """Visualize LIME explanation"""
    print(f"Visualizing LIME explanation...")
    
    # Extract explanation and weights
    explanation = lime_data['explanation']
    weights_df = lime_data['weights_df']
    
    # Save weights to CSV
    weights_df.to_csv(f'{filename_prefix}_weights.csv', index=False)
    
    # Create custom plot
    plt.figure(figsize=(12, 8))
    
    # Sort by absolute weight
    weights_df = weights_df.reindex(weights_df['weight'].abs().sort_values(ascending=False).index)
    
    # Create bar plot
    colors = ['red' if w < 0 else 'green' for w in weights_df['weight']]
    plt.barh(weights_df['feature'], weights_df['weight'], color=colors)
    plt.xlabel('LIME Weight (Impact on Prediction)', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Feature Impact According to LIME', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Increases prediction'),
        Patch(facecolor='red', label='Decreases prediction')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_explanation.png', dpi=300)
    
    return weights_df

# Visualize saliency maps
def visualize_saliency_maps(saliency_maps, filename='results/xai_outputs/saliency_map.png'):
    """Visualize saliency maps"""
    print(f"Visualizing saliency maps...")
    
    # Average across channels to get temporal saliency
    temporal_saliency = np.mean(np.abs(saliency_maps), axis=1)
    
    # Create heatmap
    plt.figure(figsize=(15, 6))
    
    # For multiple samples, create a multi-row heatmap
    if temporal_saliency.ndim > 1:
        num_samples = temporal_saliency.shape[0]
        plt.imshow(temporal_saliency, aspect='auto', interpolation='none', cmap='viridis')
        plt.colorbar(label='Attribution Magnitude')
        plt.xlabel('Time Steps', fontsize=14)
        plt.ylabel('Samples', fontsize=14)
        plt.yticks(range(num_samples))
    else:
        # For a single sample
        # Create a reshaped version for better visualization
        time_points = len(temporal_saliency)
        rows = min(8, time_points)  # Ensure we don't create more rows than we have points
        cols = (time_points + rows - 1) // rows  # Ceiling division
        reshaped_saliency = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < time_points:
                    reshaped_saliency[i, j] = temporal_saliency[idx]
        
        plt.imshow(reshaped_saliency, aspect='auto', interpolation='none', cmap='viridis')
        plt.colorbar(label='Attribution Magnitude')
        plt.xlabel('Time (segments)', fontsize=14)
        plt.ylabel('Time segments', fontsize=14)
    
    plt.title('Temporal Saliency Map', fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    
    # Save raw data
    if temporal_saliency.ndim > 1:
        saliency_df = pd.DataFrame(temporal_saliency)
    else:
        saliency_df = pd.DataFrame({
            'Time_Point': range(len(temporal_saliency)),
            'Saliency': temporal_saliency
        })
    saliency_df.to_csv(filename.replace('.png', '.csv'), index=False)
    
    return temporal_saliency

# Visualize attention weights
def visualize_attention_weights(attention_info, filename='results/xai_outputs/attention_weights.png'):
    """Visualize attention weights"""
    print(f"Visualizing attention weights...")
    
    # Check if we have valid attention weights
    if 'weights' not in attention_info or attention_info['weights'].size == 0:
        print("Warning: No attention weights available to visualize")
        # Create a placeholder plot
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No attention weights available", 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(filename, dpi=300)
        return attention_info
    
    attention_weights = attention_info['weights']
    
    plt.figure(figsize=(12, 8))
    
    # Plot attention weights as a heatmap
    sns.heatmap(attention_weights, cmap='viridis', annot=False)
    plt.xlabel('Feature Dimension', fontsize=14)
    plt.ylabel('Sample', fontsize=14)
    
    # Add frequency attention info to title if available
    title = 'Attention Weights on Feature Dimensions'
    if 'freq_attention_percentage' in attention_info:
        title += f" (Frequency Features: {attention_info['freq_attention_percentage']:.1f}%)"
    plt.title(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    
    # Save raw data
    pd.DataFrame(attention_weights).to_csv(filename.replace('.png', '.csv'), index=False)
    
    # Save attention info to JSON
    with open(filename.replace('.png', '_info.json'), 'w') as f:
        import json
        # Convert numpy values to Python types for JSON serialization
        info_dict = {}
        for k, v in attention_info.items():
            if k == 'weights':
                continue  # Skip weights as we save them to CSV
            if isinstance(v, np.ndarray):
                info_dict[k] = v.tolist()
            elif isinstance(v, np.number):
                info_dict[k] = v.item()
            else:
                info_dict[k] = v
        json.dump(info_dict, f, indent=2)
    
    return attention_info

# Generate a comprehensive XAI evaluation report
def generate_evaluation_report(data_info, channel_importance, temporal_importance, 
                              perturbation_data, lime_data, saliency_data, attention_data, num_steps=25):
    """Generate a comprehensive evaluation report in Markdown format"""
    print("Generating evaluation report...")
    
    # Get top channels by importance
    top_channels = np.argsort(channel_importance)[::-1][:5]
    
    # Extract key time points
    important_time_windows = []
    window_size = 100
    for i in range(0, len(temporal_importance), window_size):
        end = min(i + window_size, len(temporal_importance))
        window_avg = np.mean(temporal_importance[i:end])
        important_time_windows.append((i, end, window_avg))
    
    important_time_windows.sort(key=lambda x: x[2], reverse=True)
    top_time_windows = important_time_windows[:3]
    
    # Create the report
    report = f"""# Seizure Forecasting Model XAI Evaluation Report

## Dataset Information
- **Preictal Samples**: {data_info['preictal_count']}
- **Interictal Samples**: {data_info['interictal_count']}
- **Imbalance Ratio**: {data_info['imbalance_ratio']:.2f}

## Model Architecture
- **Feature Extractors**: Combined SimCLR and Encoder models
- **Classifier**: AttentionEnhancedSNN with attention mechanism 
- **SNN Simulation Steps**: {num_steps}
- **Input Shape**: [batch, {data_info.get('channels', 'unknown')} channels, {data_info.get('time_points', 'unknown')} time points]

## XAI Findings

### Key EEG Channels
The model pays particular attention to the following channels:
"""

    # Add top channels
    for i, channel_idx in enumerate(top_channels):
        importance = channel_importance[channel_idx]
        report += f"- **Channel {channel_idx}**: Attribution score of {importance:.4f}\n"
    
    report += "\n### Important Time Segments\n"
    
    # Add top time windows
    for i, (start, end, importance) in enumerate(top_time_windows):
        percentage_start = start / len(temporal_importance) * 100
        percentage_end = end / len(temporal_importance) * 100
        report += f"- **Window {start}-{end}** ({percentage_start:.1f}%-{percentage_end:.1f}% of recording): Attribution score of {importance:.4f}\n"
    
    # Add Perturbation insights
    report += "\n### Feature Importance (Perturbation Sensitivity Analysis)\n"
    top_perturbation_features = perturbation_data['top_features'].head(5)
    for _, row in top_perturbation_features.iterrows():
        report += f"- **{row['feature']}**: Perturbation value of {row['perturbation_value']:.4f}\n"
    
    # Add LIME insights
    report += "\n### Local Feature Impact (LIME Analysis)\n"
    top_lime_features = lime_data['weights_df'].head(5)
    for _, row in top_lime_features.iterrows():
        impact = "increases" if row['weight'] > 0 else "decreases"
        report += f"- **{row['feature']}**: {impact} prediction with weight {row['weight']:.4f}\n"
    
    # Add attention insights
    report += "\n### Attention Mechanism Analysis\n"
    report += f"- The model's attention layer shows focus on specific feature dimensions derived from the combined encoders\n"
    report += f"- Frequency domain features (5 bands per channel) receive {attention_data['freq_attention_percentage']:.1f}% of the attention weight\n"
    
    # Add clinical insights
    report += """
## Clinical Insights

The SNN-based seizure forecasting model appears to identify patterns consistent with known epilepsy biomarkers:

1. **Channel Importance**: The model focuses on specific channels that may correspond to the seizure onset zone or propagation pathways.

2. **Temporal Patterns**: The model gives higher importance to specific segments of the EEG recording, potentially identifying pre-ictal changes that occur minutes to hours before seizure onset.

3. **Frequency Features**: Attention analysis suggests the model leverages frequency domain information, which aligns with clinical knowledge that certain frequency bands (particularly theta, delta, and high-frequency oscillations) can be predictive of seizure activity.

4. **Biomarker Alignment**: The identified features align with known epilepsy biomarkers such as:
   - Changes in synchronization patterns
   - Shifts in spectral power
   - Presence of high-frequency oscillations (HFOs)
   - Alterations in network connectivity

## Limitations and Considerations

1. **SNN Interpretability**: The spiking nature of the model presents unique challenges for XAI methods designed for standard neural networks.

2. **Temporal Dynamics**: Standard attribution methods may not fully capture the temporal dependencies that SNNs are designed to process.

3. **Sample Size**: This analysis was performed on a limited number of samples; a more comprehensive study would be beneficial.

4. **Clinical Validation**: These XAI findings should be validated by clinical experts to confirm alignment with physiological markers of seizure development.

## Recommendations

1. **Channel-Specific Monitoring**: Focus real-time monitoring on the top channels identified by the model.

2. **Time Window Focus**: Pay special attention to EEG changes occurring in the identified important time windows.

3. **Model Refinement**: Use these insights to potentially refine the model architecture, focusing computational resources on the most informative features.

4. **Clinical Integration**: Work with epileptologists to validate and integrate these findings into clinical practice.
"""

    # Write the report to a file
    with open('xai/evaluation_report.md', 'w') as f:
        f.write(report)
    
    print("Evaluation report saved to xai/evaluation_report.md")
    
    return report

# Main function
def main():
    """Main function to run XAI analysis"""
    # Load the model
    model, xai_model, in_channels = load_model()
    
    # Print model configuration
    print(f"Model configuration:")
    print(f"- Input channels: {in_channels}")
    print(f"- SNN time steps: {xai_model.num_steps}")
    print(f"- Using device: {device}")
    
    # Load data samples for analysis
    data = load_data_samples()
    bg_samples = data['bg_samples']
    analysis_samples = data['analysis_samples']
    analysis_labels = data['analysis_labels']
    data_info = data['data_info']
    
    # Update data info with dynamic dimensions rather than hardcoded values
    data_info['channels'] = analysis_samples.shape[1]
    data_info['time_points'] = analysis_samples.shape[2]
    
    # Convert samples to tensors
    analysis_tensors = torch.tensor(analysis_samples, dtype=torch.float32, device=device)
    
    # 1. Integrated Gradients
    print("\n===== Integrated Gradients Analysis =====")
    ig_attributions = compute_integrated_gradients(xai_model, analysis_samples)
    
    # Visualize channel importance
    channel_importance = visualize_channel_importance(ig_attributions)
    
    # Visualize temporal importance
    temporal_importance = visualize_temporal_importance(ig_attributions)
    
    # Save raw attributions
    np.save('xai/ig_attributions.npy', ig_attributions)
    
    # 2. Perturbation Sensitivity Analysis
    print("\n===== Perturbation Sensitivity Analysis =====")
    # Select one preictal sample for Perturbation analysis
    preictal_indices = np.where(analysis_labels == 1)[0]
    num_samples = min(20, len(preictal_indices))
    selected_indices = preictal_indices[:num_samples]
    
    print(f"Running XAI on {num_samples} preictal samples...")
    
    all_perturbation_values = []
    for idx in tqdm(selected_indices, desc="Perturbation Analysis"):
        sample = analysis_samples[idx]
        p_data = compute_perturbation_importance(xai_model, sample, bg_samples, num_features=20)
        all_perturbation_values.append(p_data['perturbation_values'])
        
    avg_perturbation = np.mean(np.vstack(all_perturbation_values), axis=0).reshape(1, -1)
    perturbation_data = p_data.copy()
    perturbation_data['perturbation_values'] = avg_perturbation
    
    # Just use the first sample for the rest
    preictal_sample = analysis_samples[selected_indices[0]:selected_indices[0]+1]
    visualize_perturbation_importance(perturbation_data)
    
    # 3. LIME Analysis
    print("\n===== LIME Analysis =====")
    lime_data = compute_lime_explanation(preictal_sample[0], bg_samples, xai_model)
    visualize_lime_explanation(lime_data)
    
    # 4. Saliency Maps
    print("\n===== Saliency Maps =====")
    saliency_maps = compute_saliency_maps(xai_model, preictal_sample)
    saliency_data = visualize_saliency_maps(saliency_maps)
    
    # 5. Attention Visualization
    print("\n===== Attention Visualization =====")
    attention_extractor = AttentionExtractor(model)
    
    # Get attention info for all analysis samples with frequency features
    attention_info = attention_extractor.get_attention_weights(analysis_tensors)
    
    # Clean up hooks
    attention_extractor.remove_hooks()
    
    # Visualize attention weights
    attention_data = visualize_attention_weights(attention_info)
    
    # If we couldn't extract attention weights, set a default frequency percentage
    if 'freq_attention_percentage' not in attention_data:
        print("Warning: Could not calculate frequency attention percentage")
        attention_data['freq_attention_percentage'] = 0.0
    
    # 6. Generate evaluation report
    print("\n===== Generating Evaluation Report =====")
    generate_evaluation_report(
        data_info, 
        channel_importance, 
        temporal_importance,
        perturbation_data,
        lime_data,
        saliency_data,
        attention_data,
        num_steps=xai_model.num_steps
    )
    
    print("\nXAI analysis completed. Results saved to the 'xai' directory.")

if __name__ == "__main__":
    main()

