# Master Research Workflow

This document provides a comprehensive guide to executing the neonatal seizure forecasting pipeline. The codebase has been consolidated into a flat structure for ease of use.

---

## 🏗️ The 4-Step Pipeline

### Step 1: Data Acquisition & Pre-training
**Primary Script:** `seizure_forecasting_pipeline.ipynb`

This is the most compute-intensive part of the research. It must be run first and requires the raw dataset.

1.  **EDF Segmenting**: Raw EDF files from the Helsinki dataset are parsed using `mne`.
2.  **Labeling**: Segments are meticulously labeled as **Preictal** (4-minute window before onset) or **Interictal** (baseline).
3.  **Contrastive Pre-training (SimCLR)**:
    *   A 1D-ResNet encoder is trained on unlabeled interictal segments.
    *   This forces the model to learn useful temporal features (e.g., rhythmic activity vs. artifacts) before even seeing a label.
4.  **Outputs**: 
    *   `data/X_preictal.npy` & `data/X_interictal.npy`
    *   `pretrained_models/simclr_pretrained.pt`
    *   `pretrained_models/encoder_pretrained.pt`

---

### Step 2: SNN Architecture & Training
**Primary Script:** `improved_seizure_forecaster.py`

Once you have the pretrained backbone from Step 1, you can train the classification head.

1.  **Hybrid Input**: The model combines the 192-dim representation from the SimCLR/Encoder backbone with 32-dim spectral features (Delta, Theta, Alpha, Beta, Gamma bands).
2.  **Attention Mechanism**: Focuses the network on relevant frequency bins and temporal segments.
3.  **SNN Dynamics**: Uses **snntorch**'s Leaky Integrate-and-Fire (LIF) neurons over 50 simulation steps.
4.  **Imbalance Handling**: Uses **Focal Loss** (α=0.75, γ=2.0) and class weighting to ensure preictal recall is prioritized.
5.  **Outputs**:
    *   `trained_models/lightweight_enhanced_snn_model.pt`
    *   Performance reports in `results/training_outputs/`.

---

### Step 3: Interpretability Analysis (XAI)
**Primary Script:** `xai_explain_seizure_forecaster.py`

This step answers: *"Why did the model say a seizure is coming?"*

1.  **Integrated Gradients**: Calculates the formal attribution of each electrode to the classification.
2.  **SHAP Values**: Provides a game-theoretic approach to feature importance (Channel vs. Time).
3.  **LIME**: Creates a local linear explanation around specific preictal segments.
4.  **Outputs**: Detailed visualizations in `results/xai_outputs/`.

---

### Step 4: Neurological Brain Mapping
**Primary Script:** `enhance_xai_with_brainmap.py`

The final step for clinical translation. It translates "Channel 7" into "T3 - Left Temporal Lobe".

1.  **10-20 system mapping**: All attributions from Step 3 are mapped to the International 10-20 EEG system.
2.  **Regional Summary**: Aggregates importance by lobes (Frontal, Temporal, Central, etc.).
3.  **Outputs**: 10-20 system heatmaps in `results/brain_mapping_outputs/`.

---

## 🛠️ Typical Execution Order

1.  Open `seizure_forecasting_pipeline.ipynb` in Jupyter and "Run All".
2.  Run `python improved_seizure_forecaster.py` to train the SNN.
3.  Run `python xai_explain_seizure_forecaster.py` to generate attributions.
4.  Run `python enhance_xai_with_brainmap.py` to generate the clinical brain maps.

## ⚠️ Notes for Reproducibility

*   **Random Seed**: All scripts use `SEED = 42` for consistency.
*   **Device**: Scripts automatically detect CUDA (GPU) or fallback to CPU.
*   **Paths**: All scripts expect files to be in the folder structure defined in the `README.md`.
