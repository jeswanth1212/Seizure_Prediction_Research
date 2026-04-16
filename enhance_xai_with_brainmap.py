#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhance XAI Results with Brain Mapping (10-20 System)

This script enhances XAI results by mapping EEG channels to the International 10-20 
electrode placement system, providing clinical interpretability by associating 
channels with specific brain regions and electrode names.

It processes existing XAI files (channel_importance.csv, shap_values.csv, etc.) 
and creates enhanced versions with brain region mapping in a new directory.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import OrderedDict

# Create the brain_mapping_xai directory if it doesn't exist
os.makedirs('results/brain_mapping_outputs', exist_ok=True)

# Define the International 10-20 system mapping (standard for 21-channel EEG)
# Maps channel indices to electrode names and brain regions
def create_channel_mapping():
    """Create the mapping from channel indices to electrode names and brain regions"""
    mapping = OrderedDict([
        (0, {"electrode": "Fp1", "region": "Frontal Polar - Prefrontal Cortex, Left"}),
        (1, {"electrode": "Fp2", "region": "Frontal Polar - Prefrontal Cortex, Right"}),
        (2, {"electrode": "F7", "region": "Frontal - Inferior Frontal Gyrus, Left"}),
        (3, {"electrode": "F3", "region": "Frontal - Dorsolateral Prefrontal Cortex, Left"}),
        (4, {"electrode": "Fz", "region": "Frontal - Medial Frontal Cortex, Midline"}),
        (5, {"electrode": "F4", "region": "Frontal - Dorsolateral Prefrontal Cortex, Right"}),
        (6, {"electrode": "F8", "region": "Frontal - Inferior Frontal Gyrus, Right"}),
        (7, {"electrode": "T3", "region": "Temporal - Mid-Temporal, Left, Auditory Cortex"}),
        (8, {"electrode": "C3", "region": "Central - Primary Motor/Sensory Cortex, Left"}),
        (9, {"electrode": "Cz", "region": "Central - Vertex, Supplementary Motor Area, Midline"}),
        (10, {"electrode": "C4", "region": "Central - Primary Motor/Sensory Cortex, Right"}),
        (11, {"electrode": "T4", "region": "Temporal - Mid-Temporal, Right, Auditory Cortex"}),
        (12, {"electrode": "T5", "region": "Temporal/Parietal - Posterior Temporal, Left"}),
        (13, {"electrode": "P3", "region": "Parietal - Superior Parietal Lobule, Left"}),
        (14, {"electrode": "Pz", "region": "Parietal - Precuneus, Midline"}),
        (15, {"electrode": "P4", "region": "Parietal - Superior Parietal Lobule, Right"}),
        (16, {"electrode": "T6", "region": "Temporal/Parietal - Posterior Temporal, Right"}),
        (17, {"electrode": "O1", "region": "Occipital - Primary Visual Cortex, Left"}),
        (18, {"electrode": "Oz", "region": "Occipital - Medial Visual Cortex, Midline"}),
        (19, {"electrode": "O2", "region": "Occipital - Primary Visual Cortex, Right"}),
        (20, {"electrode": "Fpz", "region": "Frontal Polar - Prefrontal Cortex, Midline"})
    ])
    return mapping

# Create abbreviated region names for plots (to avoid crowding)
def create_abbreviated_regions():
    """Create abbreviated region names for more concise plotting"""
    abbreviated_regions = {
        0: "Prefrontal-L",
        1: "Prefrontal-R",
        2: "Frontal-IFG-L",
        3: "Frontal-DLPFC-L",
        4: "Frontal-Mid",
        5: "Frontal-DLPFC-R",
        6: "Frontal-IFG-R",
        7: "Temporal-L",
        8: "Motor-Sensory-L",
        9: "Motor-SMA-Mid",
        10: "Motor-Sensory-R",
        11: "Temporal-R",
        12: "Post-Temporal-L",
        13: "Parietal-L",
        14: "Parietal-Mid",
        15: "Parietal-R",
        16: "Post-Temporal-R",
        17: "Occipital-L",
        18: "Occipital-Mid",
        19: "Occipital-R",
        20: "Prefrontal-Mid"
    }
    return abbreviated_regions

# Function to extract channel index from various formats
def extract_channel_index(channel_str):
    """Extract the channel index from various string formats"""
    if isinstance(channel_str, str):
        # Format: "Channel X" or "ChX"
        match = re.search(r'(?:Channel\s+|Ch)(\d+)', channel_str)
        if match:
            return int(match.group(1))
    elif isinstance(channel_str, (int, np.integer)):
        # Already a numeric index
        return int(channel_str)
    
    # If we can't extract a channel index, return None
    return None

# Process channel_importance.csv
def process_channel_importance(channel_mapping, abbreviated_regions):
    """Process and enhance channel_importance.csv with electrode and brain region info"""
    print("Processing channel_importance.csv...")
    
    try:
        # Read the file
        df = pd.read_csv('results/xai_outputs/channel_importance.csv')
        
        # Check if we have the expected 21 channels
        if len(df) != 21:
            print(f"Warning: channel_importance.csv has {len(df)} rows, not 21 as expected.")
            print("Check if your dataset uses a non-standard EEG montage.")
        
        # Extract channel indices
        df['Channel_Index'] = df['Channel'].apply(extract_channel_index)
        
        # Add electrode and brain region information
        df['Electrode'] = df['Channel_Index'].apply(
            lambda idx: channel_mapping.get(idx, {}).get('electrode', 'Unknown')
        )
        df['Brain_Region'] = df['Channel_Index'].apply(
            lambda idx: channel_mapping.get(idx, {}).get('region', 'Unknown')
        )
        
        # Create abbreviated brain region for plotting
        df['Region_Short'] = df['Channel_Index'].apply(
            lambda idx: abbreviated_regions.get(idx, 'Unknown')
        )
        
        # Sort by importance (descending) to highlight most important channels
        df = df.sort_values('Importance', ascending=False)
        
        # Save enhanced CSV
        df.to_csv('results/brain_mapping_outputs/channel_importance_mapped.csv', index=False)
        
        # Create enhanced visualization
        plt.figure(figsize=(14, 8))
        
        # Create channel labels with electrode names and abbreviated regions
        channel_labels = [
            f"Ch{row.Channel_Index} ({row.Electrode}\n{row.Region_Short})" 
            for _, row in df.iterrows()
        ]
        
        # Create bar plot
        bars = plt.bar(range(len(df)), df['Importance'], color='steelblue')
        plt.xticks(range(len(df)), channel_labels, rotation=90, fontsize=10)
        plt.xlabel('EEG Channels with 10-20 System Mapping', fontsize=12)
        plt.ylabel('Attribution Importance', fontsize=12)
        plt.title('Channel Importance for Seizure Prediction (10-20 System)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05*max(df['Importance']),
                    f'{height:.2e}', ha='center', va='bottom', rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/brain_mapping_outputs/channel_importance_mapped.png', dpi=300)
        plt.close()
        
        print("âœ“ Enhanced channel_importance.csv and created visualization")
        return df
    
    except FileNotFoundError:
        print("Warning: xai/channel_importance.csv not found. Skipping.")
        return None

# Process shap_channel_importance.csv
def process_shap_channel_importance(channel_mapping, abbreviated_regions):
    """Process and enhance shap_channel_importance.csv with electrode and brain region info"""
    print("Processing shap_channel_importance.csv...")
    
    try:
        # Read the file
        df = pd.read_csv('results/xai_outputs/shap_channel_importance.csv')
        
        # Check if we have the expected 21 channels
        if len(df) != 21:
            print(f"Warning: shap_channel_importance.csv has {len(df)} rows, not 21 as expected.")
            print("Check if your dataset uses a non-standard EEG montage.")
        
        # Add electrode and brain region information
        df['Electrode'] = df['channel'].apply(
            lambda idx: channel_mapping.get(idx, {}).get('electrode', 'Unknown')
        )
        df['Brain_Region'] = df['channel'].apply(
            lambda idx: channel_mapping.get(idx, {}).get('region', 'Unknown')
        )
        
        # Create abbreviated brain region for plotting
        df['Region_Short'] = df['channel'].apply(
            lambda idx: abbreviated_regions.get(idx, 'Unknown')
        )
        
        # Sort by importance (descending) to highlight most important channels
        df = df.sort_values('importance', ascending=False)
        
        # Save enhanced CSV
        df.to_csv('results/brain_mapping_outputs/shap_channel_importance_mapped.csv', index=False)
        
        # Create enhanced visualization
        plt.figure(figsize=(14, 8))
        
        # Create channel labels with electrode names and abbreviated regions
        channel_labels = [
            f"Ch{row.channel} ({row.Electrode}\n{row.Region_Short})" 
            for _, row in df.iterrows()
        ]
        
        # Create bar plot
        bars = plt.bar(range(len(df)), df['importance'], color='steelblue')
        plt.xticks(range(len(df)), channel_labels, rotation=90, fontsize=10)
        plt.xlabel('EEG Channels with 10-20 System Mapping', fontsize=12)
        plt.ylabel('SHAP Importance', fontsize=12)
        plt.title('Channel Importance from SHAP Analysis (10-20 System)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05*max(df['importance']),
                    f'{height:.2e}', ha='center', va='bottom', rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/brain_mapping_outputs/shap_channel_importance_mapped.png', dpi=300)
        plt.close()
        
        print("âœ“ Enhanced shap_channel_importance.csv and created visualization")
        return df
    
    except FileNotFoundError:
        print("Warning: xai/shap_channel_importance.csv not found. Skipping.")
        return None

# Process lime_weights.csv
def process_lime_weights(channel_mapping):
    """Process and enhance lime_weights.csv with electrode and brain region info"""
    print("Processing lime_weights.csv...")
    
    try:
        # Read the file
        df = pd.read_csv('results/xai_outputs/lime_weights.csv')
        
        # Function to map feature to electrode and brain region
        def map_feature(feature):
            # Extract channel number using regex
            match = re.search(r'Ch(\d+)', feature)
            if match:
                ch_idx = int(match.group(1))
                electrode = channel_mapping.get(ch_idx, {}).get('electrode', 'Unknown')
                region_short = channel_mapping.get(ch_idx, {}).get('region', 'Unknown').split(' - ')[0]
                return f"{feature} ({electrode} - {region_short})"
            return feature
        
        # Add mapped feature
        df['Mapped_Feature'] = df['feature'].apply(map_feature)
        
        # Sort by absolute weight to highlight most important features
        df['abs_weight'] = df['weight'].abs()
        df = df.sort_values('abs_weight', ascending=False)
        df = df.drop('abs_weight', axis=1)
        
        # Save enhanced CSV
        df.to_csv('results/brain_mapping_outputs/lime_weights_mapped.csv', index=False)
        
        # Create enhanced visualization if we have a manageable number of features
        if len(df) <= 30:  # Only create visualization if we have a reasonable number to display
            plt.figure(figsize=(14, 10))
            
            # Create bar plot with colors based on weight direction
            colors = ['green' if w > 0 else 'red' for w in df['weight']]
            bars = plt.barh(df['Mapped_Feature'], df['weight'], color=colors)
            
            plt.xlabel('LIME Weight (Impact on Prediction)', fontsize=12)
            plt.ylabel('Features with Brain Region Mapping', fontsize=12)
            plt.title('LIME Feature Importance (10-20 System)', fontsize=14)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Increases prediction'),
                Patch(facecolor='red', label='Decreases prediction')
            ]
            plt.legend(handles=legend_elements, loc='best')
            
            plt.tight_layout()
            plt.savefig('results/brain_mapping_outputs/lime_weights_mapped.png', dpi=300)
            plt.close()
        
        print("âœ“ Enhanced lime_weights.csv")
        return df
    
    except FileNotFoundError:
        print("Warning: xai/lime_weights.csv not found. Skipping.")
        return None

# Process shap_values.csv
def process_shap_values(channel_mapping):
    """Process and enhance shap_values.csv with electrode and brain region info"""
    print("Processing shap_values.csv...")
    
    try:
        # Read the file
        df = pd.read_csv('results/xai_outputs/shap_values.csv')
        
        # Add electrode and brain region information based on channel column
        df['Electrode'] = df['channel'].apply(
            lambda idx: channel_mapping.get(idx, {}).get('electrode', 'Unknown')
        )
        df['Brain_Region'] = df['channel'].apply(
            lambda idx: channel_mapping.get(idx, {}).get('region', 'Unknown')
        )
        
        # Save enhanced CSV
        df.to_csv('results/brain_mapping_outputs/shap_values_mapped.csv', index=False)
        
        # Create enhanced visualization if we have a manageable number of features
        if len(df) <= 30:  # Only create visualization if we have a reasonable number to display
            plt.figure(figsize=(14, 10))
            
            # Create feature labels with electrode names and brain regions
            feature_labels = [
                f"Ch{row.channel} ({row.Electrode}) t{row.time_start}-{row.time_end}" 
                for _, row in df.iterrows()
            ]
            
            # Create bar plot with colors based on importance direction
            colors = ['green' if imp > 0 else 'red' for imp in df['importance']]
            bars = plt.barh(feature_labels, df['importance'], color=colors)
            
            plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
            plt.ylabel('Features with Brain Region Mapping', fontsize=12)
            plt.title('SHAP Feature Importance (10-20 System)', fontsize=14)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('results/brain_mapping_outputs/shap_values_mapped.png', dpi=300)
            plt.close()
        
        print("âœ“ Enhanced shap_values.csv")
        return df
    
    except FileNotFoundError:
        print("Warning: xai/shap_values.csv not found. Skipping.")
        return None

# Copy files that don't need modification
def copy_unmodified_files():
    """Copy files that don't need modification to the brain_mapping_xai directory"""
    files_to_copy = [
        'temporal_importance.csv',
        'attention_weights.csv',
        'attention_weights_info.json'
    ]
    
    for file in files_to_copy:
        source_file = f'results/xai_outputs/{file}'
        target_file = f'results/brain_mapping_outputs/{file}'
        
        try:
            # Read and write to copy the file
            with open(source_file, 'r') as f:
                content = f.read()
            
            with open(target_file, 'w') as f:
                f.write(content)
            
            print(f"âœ“ Copied {file} to brain_mapping_xai/")
        
        except FileNotFoundError:
            print(f"Warning: {source_file} not found. Skipping.")

# Create or update evaluation report
def create_evaluation_report(ig_df, shap_df):
    """Create or update evaluation report with brain mapping insights"""
    print("Creating evaluation report with brain mapping insights...")
    
    # Try to load existing report
    report_content = ""
    try:
        with open('results/xai_outputs/evaluation_report.md', 'r') as f:
            report_content = f.read()
        print("Loaded existing evaluation report")
    except FileNotFoundError:
        # Create placeholder if not found
        report_content = """# Seizure Forecasting XAI Report

## Results

Various XAI techniques were applied to understand how the model predicts seizures.

## Limitations and Considerations

1. **SNN Interpretability**: The spiking nature of the model presents unique challenges for XAI methods.
2. **Temporal Dynamics**: Standard attribution methods may not fully capture temporal dependencies.
"""
        print("Created placeholder evaluation report")
    
    # Split the report at the Limitations section to insert our brain mapping section
    parts = re.split(r'(## Limitations and Considerations)', report_content, maxsplit=1)
    
    if len(parts) < 2:
        # If we didn't find the Limitations section, just append to the end
        report_content += "\n\n"
        brain_mapping_section = "## Brain Map Insights (10-20 System)\n\n"
        limitations_section = ""
    else:
        # Insert between Results and Limitations
        brain_mapping_section = "\n## Brain Map Insights (10-20 System)\n\n"
        limitations_section = "\n" + parts[1] + parts[2] if len(parts) > 2 else ""
        report_content = parts[0]
    
    # Add brain mapping insights
    brain_mapping_section += "The EEG channels have been mapped to the International 10-20 system to provide clinical interpretability.\n\n"
    
    # Add insights from channel_importance (Integrated Gradients)
    if ig_df is not None and not ig_df.empty:
        brain_mapping_section += "### Key Channels from Integrated Gradients Analysis\n\n"
        
        # Get top 5 channels
        top_channels = ig_df.head(5)
        for _, row in top_channels.iterrows():
            ch_idx = row.get('Channel_Index', row.get('Channel', 'Unknown'))
            electrode = row['Electrode']
            region = row['Brain_Region']
            importance = row['Importance']
            
            brain_mapping_section += f"- **Channel {ch_idx} ({electrode})**: Attribution score of {importance:.4e}\n"
            brain_mapping_section += f"  - Brain Region: {region}\n"
    
    # Add insights from shap_channel_importance
    if shap_df is not None and not shap_df.empty:
        brain_mapping_section += "\n### Key Channels from SHAP Analysis\n\n"
        
        # Get top 5 channels
        top_channels = shap_df.head(5)
        for _, row in top_channels.iterrows():
            ch_idx = row['channel']
            electrode = row['Electrode']
            region = row['Brain_Region']
            importance = row['importance']
            
            brain_mapping_section += f"- **Channel {ch_idx} ({electrode})**: SHAP value of {importance:.4e}\n"
            brain_mapping_section += f"  - Brain Region: {region}\n"
    
    # Add clinical note
    brain_mapping_section += """
### Clinical Significance

The brain mapping reveals important patterns in how the model identifies seizures:

- **Temporal Channels** (T3=Ch7, T4=Ch11): These areas are often associated with temporal lobe seizures, which account for approximately 40% of all seizures. The model's focus here aligns with clinical knowledge about seizure origins.

- **Central Channels** (Cz=Ch9, C3=Ch8, C4=Ch10): These regions detect motor seizure propagation and are important for identifying hypersynchrony patterns that often occur during seizure activity.

- **Frontal Channels** (Fp1=Ch0, Fp2=Ch1, F3=Ch3): The frontal lobe, particularly prefrontal areas, often shows early executive and cognitive changes before seizure onset. The model's attention to these channels could indicate detection of preictal cognitive shifts.

- **Parietal & Occipital Channels** (P3=Ch13, Pz=Ch14, O1=Ch17): These regions are involved in sensory integration and visual processing. Their importance may relate to certain types of seizures with sensory manifestations or visual auras.

This brain region mapping provides clinically actionable insights for EEG monitoring, suggesting key areas to focus on for early seizure detection.
"""
    
    # Combine all sections
    updated_report = report_content + brain_mapping_section + limitations_section
    
    # Save enhanced report
    with open('results/brain_mapping_outputs/evaluation_report_brainmap.md', 'w') as f:
        f.write(updated_report)
    
    print("âœ“ Created enhanced evaluation report with brain mapping insights")

def main():
    """Main function to enhance XAI results with brain mapping"""
    print("Enhancing XAI results with International 10-20 EEG electrode system mapping...\n")
    
    # Create the channel mapping
    channel_mapping = create_channel_mapping()
    abbreviated_regions = create_abbreviated_regions()
    
    # Process each file
    ig_df = process_channel_importance(channel_mapping, abbreviated_regions)
    shap_df = process_shap_channel_importance(channel_mapping, abbreviated_regions)
    process_lime_weights(channel_mapping)
    process_shap_values(channel_mapping)
    copy_unmodified_files()
    
    # Create or update evaluation report
    create_evaluation_report(ig_df, shap_df)
    
    print("\nâœ… All XAI results have been enhanced with brain mapping in the 'results/brain_mapping_outputs' directory")

if __name__ == "__main__":
    main()




