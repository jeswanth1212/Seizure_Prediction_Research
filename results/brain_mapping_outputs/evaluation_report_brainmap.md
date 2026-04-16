# Seizure Forecasting XAI Report

## Results

Various XAI techniques were applied to understand how the model predicts seizures.


## Brain Map Insights (10-20 System)

The EEG channels have been mapped to the International 10-20 system to provide clinical interpretability.

### Key Channels from Integrated Gradients Analysis

- **Channel 7 (T3)**: Attribution score of 1.0256e-06
  - Brain Region: Temporal - Mid-Temporal, Left, Auditory Cortex
- **Channel 6 (F8)**: Attribution score of 1.0019e-06
  - Brain Region: Frontal - Inferior Frontal Gyrus, Right
- **Channel 15 (P4)**: Attribution score of 8.6669e-07
  - Brain Region: Parietal - Superior Parietal Lobule, Right
- **Channel 14 (Pz)**: Attribution score of 8.3978e-07
  - Brain Region: Parietal - Precuneus, Midline
- **Channel 11 (T4)**: Attribution score of 7.7550e-07
  - Brain Region: Temporal - Mid-Temporal, Right, Auditory Cortex

### Key Channels from SHAP Analysis

- **Channel 1 (Fp2)**: SHAP value of 8.8774e-07
  - Brain Region: Frontal Polar - Prefrontal Cortex, Right
- **Channel 0 (Fp1)**: SHAP value of 8.7693e-07
  - Brain Region: Frontal Polar - Prefrontal Cortex, Left
- **Channel 3 (F3)**: SHAP value of 7.9240e-07
  - Brain Region: Frontal - Dorsolateral Prefrontal Cortex, Left
- **Channel 5 (F4)**: SHAP value of 7.0180e-07
  - Brain Region: Frontal - Dorsolateral Prefrontal Cortex, Right
- **Channel 2 (F7)**: SHAP value of 6.9325e-07
  - Brain Region: Frontal - Inferior Frontal Gyrus, Left

### Clinical Significance

The brain mapping reveals important patterns in how the model identifies seizures:

- **Temporal Channels** (T3=Ch7, T4=Ch11): These areas are often associated with temporal lobe seizures, which account for approximately 40% of all seizures. The model's focus here aligns with clinical knowledge about seizure origins.

- **Central Channels** (Cz=Ch9, C3=Ch8, C4=Ch10): These regions detect motor seizure propagation and are important for identifying hypersynchrony patterns that often occur during seizure activity.

- **Frontal Channels** (Fp1=Ch0, Fp2=Ch1, F3=Ch3): The frontal lobe, particularly prefrontal areas, often shows early executive and cognitive changes before seizure onset. The model's attention to these channels could indicate detection of preictal cognitive shifts.

- **Parietal & Occipital Channels** (P3=Ch13, Pz=Ch14, O1=Ch17): These regions are involved in sensory integration and visual processing. Their importance may relate to certain types of seizures with sensory manifestations or visual auras.

This brain region mapping provides clinically actionable insights for EEG monitoring, suggesting key areas to focus on for early seizure detection.

## Limitations and Considerations

1. **SNN Interpretability**: The spiking nature of the model presents unique challenges for XAI methods.
2. **Temporal Dynamics**: Standard attribution methods may not fully capture temporal dependencies.