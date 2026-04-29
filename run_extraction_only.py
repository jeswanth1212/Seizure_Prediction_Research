import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Data processing
import pyedflib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
DATA_DIR = 'dataset'
EEG_DIR = DATA_DIR
ANNOTATIONS_A = os.path.join(DATA_DIR, 'annotations_2017_A.csv')
ANNOTATIONS_B = os.path.join(DATA_DIR, 'annotations_2017_B.csv')
ANNOTATIONS_C = os.path.join(DATA_DIR, 'annotations_2017_C.csv')
CLINICAL_INFO_FILE = os.path.join(DATA_DIR, 'clinical_information (1).csv')

# Function to load consensus annotations
def load_consensus_annotations(file_a, file_b, file_c):
    """
    Load annotations from 3 experts and create a majority-vote consensus
    """
    print(f"Loading annotations from Experts A, B, and C...")
    try:
        df_a = pd.read_csv(file_a)
        df_b = pd.read_csv(file_b)
        df_c = pd.read_csv(file_c)
        
        # Ensure they have the same shape
        if not (df_a.shape == df_b.shape == df_c.shape):
            print("Warning: Annotation files have different shapes. Attempting to align...")
            # For this dataset, they should be 1-79. We'll just take the intersection of columns and rows
            common_cols = sorted(list(set(df_a.columns) & set(df_b.columns) & set(df_c.columns)))
            min_rows = min(len(df_a), len(df_b), len(df_c))
            df_a = df_a[common_cols].iloc[:min_rows]
            df_b = df_b[common_cols].iloc[:min_rows]
            df_c = df_c[common_cols].iloc[:min_rows]

        consensus_df = df_a.copy()
        
        # Majority vote: (A + B + C) >= 2
        for col in df_a.columns:
            votes = df_a[col].fillna(0) + df_b[col].fillna(0) + df_c[col].fillna(0)
            consensus_df[col] = (votes >= 2).astype(float)
            
        print(f"Consensus created successfully. Shape: {consensus_df.shape}")
        return consensus_df
    except Exception as e:
        print(f"Error creating consensus: {str(e)}")
        return None

def load_clinical_info(clinical_info_file):
    try:
        clinical_info_df = pd.read_csv(clinical_info_file)
        print(f"Clinical info loaded. Shape: {clinical_info_df.shape}")
        return clinical_info_df
    except Exception as e:
        print(f"Error loading clinical info: {str(e)}")
        return None

# Load annotations and clinical info
annotations_df = load_consensus_annotations(ANNOTATIONS_A, ANNOTATIONS_B, ANNOTATIONS_C)
clinical_info_df = load_clinical_info(CLINICAL_INFO_FILE)

# Function to load EEG from EDF file
def load_eeg_file(edf_file):
    try:
        f = pyedflib.EdfReader(edf_file)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in range(n):
            sigbufs[i, :] = f.readSignal(i)
        fs = f.getSampleFrequency(0)
        f.close()
        return sigbufs, signal_labels, fs
    except Exception as e:
        print(f"Error loading {edf_file}: {str(e)}")
        return None, None, None

def get_eeg_files(eeg_dir):
    eeg_files = sorted(
        [os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith('.edf')],
        key=lambda x: int(os.path.basename(x).replace('eeg', '').replace('.edf', ''))
    )
    print(f"Found {len(eeg_files)} EEG files")
    return eeg_files

eeg_files = get_eeg_files(EEG_DIR)

# Define constants for segmentation
WINDOW_SIZE_SEC = 10
PRE_ICTAL_WINDOW = 4 * 60
INTER_ICTAL_MIN_DIST = 5 * 60

def extract_segments(eeg_data, fs, window_size_sec=WINDOW_SIZE_SEC, overlap=0.5):
    n_channels, n_samples = eeg_data.shape
    samples_per_window = int(window_size_sec * fs)
    step_size = int(samples_per_window * (1 - overlap))
    segments = []
    for start in range(0, n_samples - samples_per_window + 1, step_size):
        end = start + samples_per_window
        segment = eeg_data[:, start:end]
        segments.append(segment)
    return segments

def normalize_segment(segment, per_channel=True):
    if per_channel:
        for i in range(segment.shape[0]):
            if np.std(segment[i]) > 1e-6:
                segment[i] = (segment[i] - np.mean(segment[i])) / np.std(segment[i])
    else:
        if np.std(segment) > 1e-6:
            segment = (segment - np.mean(segment)) / np.std(segment)
    return segment

def parse_seizure_annotations(annotations_df, eeg_files):
    seizure_info = {}
    for eeg_file in eeg_files:
        seizure_info[eeg_file] = []
    if annotations_df is not None:
        for index, row in annotations_df.iterrows():
            for col_name in annotations_df.columns:
                if str(col_name).isdigit() and row[col_name] == 1.0:
                    col_idx = int(col_name)
                    eeg_filename = f"eeg{col_idx}.edf"
                    eeg_file_path = next((f for f in eeg_files if os.path.basename(f) == eeg_filename), None)
                    if eeg_file_path:
                        seizure_time = index * 30  # assuming annotation resolution = 30 seconds 
                        seizure_info[eeg_file_path].append(seizure_time)
    total_seizures = sum(len(times) for times in seizure_info.values())
    print(f"Found {total_seizures} consensus seizure events")
    return seizure_info

def classify_segments(eeg_file, segments, seizure_times, fs, window_size_sec=WINDOW_SIZE_SEC, 
                     preictal_window=PRE_ICTAL_WINDOW, interictal_min_dist=INTER_ICTAL_MIN_DIST, overlap=0.5):
    preictal_segments = []
    interictal_segments = []
    step_size = window_size_sec * (1 - overlap)
    for i, segment in enumerate(segments):
        segment_start_sec = i * step_size
        segment_end_sec = segment_start_sec + window_size_sec
        is_preictal = False
        for seizure_time in seizure_times:
            if (seizure_time - preictal_window) <= segment_end_sec <= seizure_time:
                preictal_segments.append(segment)
                is_preictal = True
                break
        if is_preictal: continue
        is_interictal = True
        for seizure_time in seizure_times:
            dist_before = segment_end_sec - seizure_time
            dist_after = seizure_time - segment_start_sec
            if -interictal_min_dist < dist_before < interictal_min_dist or \
               -interictal_min_dist < dist_after < interictal_min_dist:
                is_interictal = False
                break
        if is_interictal: interictal_segments.append(segment)
    return preictal_segments, interictal_segments

def process_eeg_data(eeg_files, seizure_info, max_files=None):
    all_preictal_segments = []
    all_interictal_segments = []
    all_preictal_patient_ids = []
    all_interictal_patient_ids = []
    if max_files: eeg_files = eeg_files[:max_files]
    total_files = len(eeg_files)
    for i, eeg_file in enumerate(eeg_files):
        # Extract patient ID from filename (e.g., 'eeg1.edf' -> 1)
        patient_id = int(os.path.basename(eeg_file).replace('eeg', '').replace('.edf', ''))
        if i % max(1, total_files // 10) == 0 or i == total_files - 1:
            print(f"Processing file {i+1}/{total_files}")
        sigbufs, signal_labels, fs = load_eeg_file(eeg_file)
        if sigbufs is None: continue
        segments = extract_segments(sigbufs, fs)
        normalized_segments = [normalize_segment(segment.copy()) for segment in segments]
        seizure_times = seizure_info.get(eeg_file, [])
        preictal_segments, interictal_segments = classify_segments(eeg_file, normalized_segments, seizure_times, fs)
        all_preictal_segments.extend(preictal_segments)
        all_interictal_segments.extend(interictal_segments)
        all_preictal_patient_ids.extend([patient_id] * len(preictal_segments))
        all_interictal_patient_ids.extend([patient_id] * len(interictal_segments))
    return all_preictal_segments, all_interictal_segments, all_preictal_patient_ids, all_interictal_patient_ids

seizure_info = parse_seizure_annotations(annotations_df, eeg_files)
all_preictal_segments, all_interictal_segments, all_preictal_patient_ids, all_interictal_patient_ids = process_eeg_data(eeg_files, seizure_info)

if all_preictal_segments and all_interictal_segments:
    X_preictal = np.stack(all_preictal_segments).astype(np.float32)
    X_interictal = np.stack(all_interictal_segments).astype(np.float32)
    os.makedirs('data', exist_ok=True)
    np.save('data/X_preictal.npy', X_preictal)
    np.save('data/X_interictal.npy', X_interictal)
    np.save('data/patient_ids_preictal.npy', np.array(all_preictal_patient_ids))
    np.save('data/patient_ids_interictal.npy', np.array(all_interictal_patient_ids))
    print(f"Success! Saved {len(all_preictal_segments)} preictal and {len(all_interictal_segments)} interictal segments.")
else:
    print("Error: No segments found.")
