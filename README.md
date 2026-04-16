# Hybrid SNN for Interpretable Neonatal Seizure Forecasting

Implementation of the paper:  
**"A Hybrid Spiking Neural Network with Contrastive Pretraining for Interpretable Seizure Forecasting Using Explainable AI"**

Uses the [Helsinki University Hospital Neonatal EEG Seizure Dataset](https://doi.org/10.1038/s41582-019-0200-7) — 79 neonates, 5,800 hours of EEG.

---

## 📁 Project Structure

```
seizure_prediction_1/
│
├── seizure_forecasting_pipeline.ipynb    # Step 1 — Data prep (EDF → .npy)
├── improved_seizure_forecaster.py        # Step 2 — Train the SNN model
├── xai_explain_seizure_forecaster.py     # Step 3 — XAI analysis
├── enhance_xai_with_brainmap.py          # Step 4 — Brain region mapping
├── run_xai_analysis.py                   # (Wrapper for Step 3)
│
├── data/
│   ├── X_preictal.npy                    # (10086, 21, 2560) — output of Step 1
│   └── X_interictal.npy                  # (65402, 21, 2560) — output of Step 1
│
├── pretrained_models/
│   ├── simclr_pretrained.pt              # SimCLR encoder weights (required for Step 2)
│   └── encoder_pretrained.pt             # Standard encoder weights (required for Step 2)
│
├── trained_models/
│   └── lightweight_enhanced_snn_model.pt # Output of Step 2, required for Step 3
│
├── results/
│   ├── training_outputs/                 # Outputs of Step 2
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── training_history.csv
│   │   ├── training_history.png
│   │   ├── precision_recall_curve.png
│   │   ├── class_distribution.png
│   │   ├── confidence_distribution.png
│   │   ├── classification_report.csv
│   │   ├── test_metrics.txt
│   │   └── evaluation_report.md
│   │
│   ├── xai_outputs/                      # Outputs of Step 3
│   │   ├── channel_importance.csv / .png
│   │   ├── temporal_importance.csv / .png
│   │   ├── shap_summary.png / shap_values.csv
│   │   ├── shap_beeswarm.png / shap_channel_importance.csv
│   │   ├── lime_explanation.png / lime_weights.csv
│   │   ├── saliency_map.png / .csv
│   │   └── attention_weights.png / .csv
│   │
│   └── brain_mapping_outputs/            # Outputs of Step 4
│       ├── channel_importance_mapped.csv / .png
│       ├── shap_channel_importance_mapped.csv / .png
│       ├── lime_weights_mapped.csv / .png
│       ├── shap_values_mapped.csv / .png
│       └── evaluation_report_brainmap.md
│
├── dataset/                              # Raw EDF files and annotations (not tracked in git)
│   └── annotations_2017_A_fixed.csv
│
├── requirements.txt
├── requirements_xai.txt
└── docs/
    └── WORKFLOWS.md
```

---

## 🚀 Detailed Research Pipeline

The complete research workflow consists of 4 main steps. **Step 1 is the most critical foundation.**

### Step 1: Data Prep & Unsupervised Pretraining
**Script:** `seizure_forecasting_pipeline.ipynb`

This notebook performs two massive tasks:
1.  **Data Preprocessing**: Converts raw EDF/CSV files into cleaned NumPy tensors (`data/X_preictal.npy`, `data/X_interictal.npy`).
2.  **Unsupervised Pretraining**: Implements **SimCLR (Contrastive Learning)** on the interictal data. It trains a 1D ResNet encoder to learn robust EEG representations without labels.
    *   **Output 1**: `pretrained_models/simclr_pretrained.pt` (Full SimCLR weights)
    *   **Output 2**: `pretrained_models/encoder_pretrained.pt` (Clean encoder weights for the SNN)

> [!IMPORTANT]
> You **cannot** run the training script without running this notebook first, as it generates the backbone weights for the classifier.

---

### Step 2: SNN Classifier Training
**Script:** `improved_seizure_forecaster.py`

This script takes the pretrained weights from Step 1 and builds a **Hybrid Spiking Neural Network**.
*   **Input**: `data/`, `pretrained_models/`
*   **Architecture**: Combines SimCLR embeddings + Spectral features → Attention SNN (LIF neurons).
*   **Output**: `trained_models/lightweight_enhanced_snn_model.pt` and detailed metrics in `results/training_outputs/`.

---

### Step 3: Explainable AI (XAI)
**Script:** `xai_explain_seizure_forecaster.py` (or `run_xai_analysis.py`)

Applies **SHAP, LIME, Integrated Gradients, and Saliency Maps** to explain *why* the model predicts a seizure.
*   **Input**: `trained_models/`, `data/`, `pretrained_models/`
*   **Output**: Comprehensive visualizations in `results/xai_outputs/`.

---

### Step 4: Brain Region Mapping
**Script:** `enhance_xai_with_brainmap.py`

Maps the technical electrode importances (Ch0-Ch20) to the **International 10-20 system**.
*   **Output**: Clinical reports and brain heatmaps in `results/brain_mapping_outputs/`.

---

## 📊 Technical Specifications

*   **Architecture**: Hybrid SNN with Contrastive Pre-training.
*   **Simulation**: 50 LIF Timesteps (β=0.5).
*   **Features**: 192D (Learned) + 32D (Spectral) = 224D Input.
*   **Performance**: ~99.5% Preictal Recall, 0.97 AUC.

---

## 📋 Requirements

```bash
# Core Machine Learning
pip install torch snntorch numpy scikit-learn pandas tqdm mne scipy imbalanced-learn

# Explainable AI & Visualization
pip install captum shap lime seaborn matplotlib
```

---

## 📄 License

MIT License — Research implementation only. Not validated for clinical use.
