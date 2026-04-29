import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Use regular tqdm instead of tqdm.notebook

# Data processing
import pyedflib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# snnTorch
import snntorch as snn
from snntorch import surrogate

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# --- Cell 4 ---
## 2. Data Loading and Preprocessing



# --- Cell 5 ---
# Define paths
DATA_DIR = 'dataset'
EEG_DIR = DATA_DIR
ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'annotations_2017_A_fixed.csv')  # Using the available annotation file
CLINICAL_INFO_FILE = os.path.join(DATA_DIR, 'clinical_information.csv')



# --- Cell 6 ---
# Function to load annotations
def load_annotations(annotations_file):
    """
    Load seizure annotations from CSV file
    """
    try:
        annotations_df = pd.read_csv(annotations_file)
        print(f"Annotations loaded successfully. Shape: {annotations_df.shape}")
        print(f"Column names: {annotations_df.columns.tolist()}")
        return annotations_df
    except Exception as e:
        print(f"Error loading annotations: {str(e)}")
        # Try to load with different delimiter or encoding
        try:
            annotations_df = pd.read_csv(annotations_file, delimiter=',', encoding='utf-8')
            print(f"Annotations loaded with custom params. Shape: {annotations_df.shape}")
            return annotations_df
        except Exception as e2:
            print(f"Error with alternative loading method: {str(e2)}")
            return None



# --- Cell 7 ---
# Function to load clinical information
def load_clinical_info(clinical_info_file):
    """
    Load clinical information from CSV file
    """
    try:
        clinical_info_df = pd.read_csv(clinical_info_file)
        print(f"Clinical info loaded successfully. Shape: {clinical_info_df.shape}")
        return clinical_info_df
    except Exception as e:
        print(f"Error loading clinical info: {str(e)}")
        return None



# --- Cell 8 ---
# Load annotations and clinical info
annotations_df = load_annotations(ANNOTATIONS_FILE)
clinical_info_df = load_clinical_info(CLINICAL_INFO_FILE)



# --- Cell 9 ---
# Display first few rows of annotations if available
if annotations_df is not None:
    print("\nFirst few rows of annotations:")
    print(annotations_df.head())
    
# Display first few rows of clinical info if available
if clinical_info_df is not None:
    print("\nFirst few rows of clinical information:")
    print(clinical_info_df.head())



# --- Cell 10 ---
# Function to load EEG from EDF file
def load_eeg_file(edf_file):
    """
    Load EEG data from an EDF file
    """
    try:
        f = pyedflib.EdfReader(edf_file)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in range(n):
            sigbufs[i, :] = f.readSignal(i)
            
        # Get sampling frequency
        fs = f.getSampleFrequency(0)
        f.close()
        
        return sigbufs, signal_labels, fs
    except Exception as e:
        print(f"Error loading EEG file {edf_file}: {str(e)}")
        return None, None, None



# --- Cell 11 ---
# Function to get all EEG files
def get_eeg_files(eeg_dir):
    """
    Get all EEG files from the directory
    """
    eeg_files = [os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith('.edf')]
    print(f"Found {len(eeg_files)} EEG files")
    return eeg_files



# --- Cell 12 ---
# Get list of EEG files
eeg_files = get_eeg_files(EEG_DIR)

# Test load of one EEG file to check structure
if eeg_files:
    print(f"\nLoading sample EEG file: {os.path.basename(eeg_files[0])}")
    sigbufs, signal_labels, fs = load_eeg_file(eeg_files[0])
    if sigbufs is not None:
        print(f"Signal shape: {sigbufs.shape}")
        print(f"Signal labels: {signal_labels}")
        print(f"Sampling frequency: {fs} Hz")
        
        # Plot first few seconds of first few channels
        plt.figure(figsize=(15, 8))
        seconds_to_plot = 5
        channels_to_plot = min(5, sigbufs.shape[0])
        
        for i in range(channels_to_plot):
            plt.subplot(channels_to_plot, 1, i+1)
            plt.plot(sigbufs[i, :int(seconds_to_plot * fs)])
            plt.title(f"Channel: {signal_labels[i]}")
        
        plt.tight_layout()
        plt.show()



# --- Cell 13 ---
## 3. EEG Preprocessing and Segment Extraction



# --- Cell 14 ---
# Define constants for segmentation
WINDOW_SIZE_SEC = 10  # 10-second windows for EEG segments
PRE_ICTAL_WINDOW = 4 * 60  # 4 minutes (in seconds) before seizure onset
INTER_ICTAL_MIN_DIST = 5 * 60  # At least 5 minutes (in seconds) away from any seizure



# --- Cell 15 ---
# Function to extract segments from EEG data
def extract_segments(eeg_data, fs, window_size_sec=WINDOW_SIZE_SEC, overlap=0.5):
    """
    Extract segments from EEG data using sliding window
    
    Args:
        eeg_data: EEG data array [channels, samples]
        fs: Sampling frequency
        window_size_sec: Size of window in seconds
        overlap: Overlap between consecutive windows (0 to 1)
        
    Returns:
        List of segments, each of shape [channels, samples_in_window]
    """
    n_channels, n_samples = eeg_data.shape
    samples_per_window = int(window_size_sec * fs)
    step_size = int(samples_per_window * (1 - overlap))
    
    segments = []
    for start in range(0, n_samples - samples_per_window + 1, step_size):
        end = start + samples_per_window
        segment = eeg_data[:, start:end]
        segments.append(segment)
    
    return segments



# --- Cell 16 ---
# Function to normalize EEG data
def normalize_segment(segment, per_channel=True):
    """
    Normalize EEG segment
    
    Args:
        segment: EEG segment [channels, samples]
        per_channel: If True, normalize each channel independently
        
    Returns:
        Normalized EEG segment
    """
    if per_channel:
        # Normalize each channel independently
        for i in range(segment.shape[0]):
            # Skip normalization if channel is flat or near-flat
            if np.std(segment[i]) > 1e-6:  # Avoid division by near-zero std
                segment[i] = (segment[i] - np.mean(segment[i])) / np.std(segment[i])
    else:
        # Global normalization
        if np.std(segment) > 1e-6:
            segment = (segment - np.mean(segment)) / np.std(segment)
    
    return segment



# --- Cell 17 ---
# Parse annotations to identify seizure time points
def parse_seizure_annotations(annotations_df, eeg_files):
    """
    Parse annotations to identify seizure time points for each EEG file
    
    Returns:
        Dictionary mapping EEG file paths to list of seizure onset times (in seconds)
    """
    seizure_info = {}
    
    # Initialize empty lists for all EEG files
    for eeg_file in eeg_files:
        seizure_info[eeg_file] = []
    
    # Process annotations dataframe
    # We assume that annotations_df has columns 1-79 corresponding to EEG file IDs
    # and values indicate seizure events (1.0) or no seizure (0.0)
    if annotations_df is not None:
        # Iterate through each row in annotations (representing time points)
        for index, row in annotations_df.iterrows():
            # Each column represents a patient/file
            for col_idx in range(1, 80):  # Columns 1 to 79
                col_name = str(col_idx)
                if col_name in row.index and row[col_name] == 1.0:
                    # This indicates a seizure for this file at this time point
                    # Find the corresponding EEG file
                    eeg_filename = f"eeg{col_idx}.edf"
                    eeg_file_path = next((f for f in eeg_files if os.path.basename(f) == eeg_filename), None)
                    
                    if eeg_file_path:
                        # For simplicity, we'll use the row index as the seizure time (in seconds)
                        # In a real scenario, you might have actual timestamps
                        seizure_time = index * 30  # Using row index * 30 as a proxy for time in seconds (assuming 30s epochs)
                        seizure_info[eeg_file_path].append(seizure_time)
    
    # Print summary
    total_seizures = sum(len(times) for times in seizure_info.values())
    files_with_seizures = sum(1 for times in seizure_info.values() if len(times) > 0)
    print(f"Parsed seizure information for {len(seizure_info)} EEG files")
    print(f"Found {total_seizures} seizure events across {files_with_seizures} files")
    
    return seizure_info



# --- Cell 18 ---
# Function to identify preictal and interictal segments
def classify_segments(eeg_file, segments, seizure_times, fs, window_size_sec=WINDOW_SIZE_SEC, 
                     preictal_window=PRE_ICTAL_WINDOW, interictal_min_dist=INTER_ICTAL_MIN_DIST):
    """
    Classify segments as preictal or interictal
    
    Args:
        eeg_file: Path to EEG file
        segments: List of EEG segments
        seizure_times: List of seizure onset times (in seconds) for this file
        fs: Sampling frequency
        window_size_sec: Size of segment window in seconds
        preictal_window: Time before seizure to label as preictal (seconds)
        interictal_min_dist: Minimum distance from seizures to label as interictal (seconds)
        
    Returns:
        preictal_segments: List of preictal segments
        interictal_segments: List of interictal segments
    """
    preictal_segments = []
    interictal_segments = []
    unlabeled_segments = []
    
    step_size = window_size_sec // 2  # Assuming 50% overlap
    
    # Go through each segment and classify
    for i, segment in enumerate(segments):
        segment_start_sec = i * step_size
        segment_end_sec = segment_start_sec + window_size_sec
        
        # Check if segment is preictal (4 min before seizure)
        is_preictal = False
        for seizure_time in seizure_times:
            if (seizure_time - preictal_window) <= segment_end_sec <= seizure_time:
                preictal_segments.append(segment)
                is_preictal = True
                break
                
        if is_preictal:
            continue
            
        # Check if segment is interictal (≥5 min away from any seizure)
        is_interictal = True
        for seizure_time in seizure_times:
            dist_before = segment_end_sec - seizure_time
            dist_after = seizure_time - segment_start_sec
            
            # If segment is too close to any seizure, it's not interictal
            if -interictal_min_dist < dist_before < interictal_min_dist or \
               -interictal_min_dist < dist_after < interictal_min_dist:
                is_interictal = False
                break
                
        if is_interictal:
            interictal_segments.append(segment)
        else:
            unlabeled_segments.append(segment)
            
    print(f"File {os.path.basename(eeg_file)}: Found {len(preictal_segments)} preictal, "
          f"{len(interictal_segments)} interictal, {len(unlabeled_segments)} unlabeled segments")
    
    return preictal_segments, interictal_segments



# --- Cell 19 ---
# Process all EEG files to extract labeled segments
def process_eeg_data(eeg_files, seizure_info, max_files=None, debug=False):
    """
    Process all EEG files to extract labeled segments
    
    Args:
        eeg_files: List of EEG file paths
        seizure_info: Dictionary mapping EEG file paths to seizure onset times
        max_files: Maximum number of files to process (for debugging)
        debug: If True, print debug information
        
    Returns:
        all_preictal_segments: List of preictal segments
        all_interictal_segments: List of interictal segments
        all_preictal_patient_ids: List of patient IDs for preictal segments
        all_interictal_patient_ids: List of patient IDs for interictal segments
    """
    all_preictal_segments = []
    all_interictal_segments = []
    all_preictal_patient_ids = []
    all_interictal_patient_ids = []
    
    if max_files:
        eeg_files = eeg_files[:max_files]
    
    total_files = len(eeg_files)
    print(f"Processing {total_files} EEG files...")
    
    # Manually track progress instead of using tqdm
    for i, eeg_file in enumerate(eeg_files):
        patient_id = i # Use file index as patient ID
        # Report progress every ~10% of files
        if i % max(1, total_files // 10) == 0 or i == total_files - 1:
            print(f"Processing file {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")
        
        if debug:
            print(f"\nProcessing {os.path.basename(eeg_file)}")
        
        # Load EEG data
        sigbufs, signal_labels, fs = load_eeg_file(eeg_file)
        if sigbufs is None:
            print(f"Skipping {os.path.basename(eeg_file)} due to loading error")
            continue
            
        # Extract segments
        segments = extract_segments(sigbufs, fs)
        
        # Normalize segments
        normalized_segments = [normalize_segment(segment.copy()) for segment in segments]
        
        # Get seizure times for this file
        seizure_times = seizure_info.get(eeg_file, [])
        
        # Classify segments
        preictal_segments, interictal_segments = classify_segments(
            eeg_file, normalized_segments, seizure_times, fs)
        
        all_preictal_segments.extend(preictal_segments)
        all_interictal_segments.extend(interictal_segments)
        all_preictal_patient_ids.extend([patient_id] * len(preictal_segments))
        all_interictal_patient_ids.extend([patient_id] * len(interictal_segments))
    
    print(f"\nTotal segments extracted: {len(all_preictal_segments)} preictal, "
          f"{len(all_interictal_segments)} interictal")
    
    return all_preictal_segments, all_interictal_segments, all_preictal_patient_ids, all_interictal_patient_ids



# --- Cell 20 ---
# Parse seizure annotations and process EEG data
seizure_info = parse_seizure_annotations(annotations_df, eeg_files)

# Process all files in the dataset
print(f"Processing all {len(eeg_files)} EEG files...")
all_preictal_segments, all_interictal_segments, all_preictal_patient_ids, all_interictal_patient_ids = process_eeg_data(eeg_files, seizure_info, max_files=None)

# Convert segments to numpy arrays
if all_preictal_segments and all_interictal_segments:
    # Stack numpy arrays
    X_preictal = np.stack(all_preictal_segments)
    X_interictal = np.stack(all_interictal_segments)
    
    print(f"Preictal data shape: {X_preictal.shape}")
    print(f"Interictal data shape: {X_interictal.shape}")
else:
    print("No segments were extracted. Check the seizure annotations parsing and segment extraction.")
    # We want to work with real data, so we'll raise an error rather than create dummy data
    raise ValueError("No preictal or interictal segments were found. Please check the annotations and data processing.")



# --- Cell 34 ---
import numpy as np
import os
X_preictal = np.stack(all_preictal_segments).astype(np.float32)
print(f"Created X_preictal with shape: {X_preictal.shape}, dtype: {X_preictal.dtype}")
os.makedirs('data', exist_ok=True)
np.save('data/X_preictal.npy', X_preictal)
X_interictal = np.stack(all_interictal_segments).astype(np.float32)
print(f"Created X_interictal with shape: {X_interictal.shape}, dtype: {X_interictal.dtype}")
np.save('data/X_interictal.npy', X_interictal)

np.save('data/patient_ids_preictal.npy', np.array(all_preictal_patient_ids))
np.save('data/patient_ids_interictal.npy', np.array(all_interictal_patient_ids))
print('Saved patient ID arrays successfully.')


