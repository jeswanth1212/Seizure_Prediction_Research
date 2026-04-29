# Implementation Plan: BRB-D-26-00510 Major Revision

This document serves as the master tracking plan for addressing the major revision of the manuscript "A Hybrid Spiking Neural Network with Contrastive Pretraining for Interpretable Seizure Forecasting Using Explainable AI" submitted to Brain Research Bulletin. 

It contains the full context of the project, the critical flaws identified by reviewers, and a step-by-step task list to resolve them. **Any AI assistant continuing this work should read this document first.**

---

## 📖 Part 1: Project Context & Objectives

### The Paper
We have developed a hybrid framework for detecting preictal (pre-seizure) states in neonatal EEG data (Helsinki dataset). The architecture combines:
1. **SimCLR-inspired Contrastive Pretraining:** An unsupervised encoder trained on interictal data to learn robust representations.
2. **Attention-Enhanced SNN:** A supervised SNN classifier head using Leaky Integrate-and-Fire (LIF) neurons and spectral features.
3. **XAI Suite:** Explainability tools to map model decisions back to neurophysiologically relevant brain regions.

### The Problem (Reviewer Comments)
The paper received a "Major Revision" decision. The reviewer identified several critical flaws that threaten the scientific validity of the work. If these are not fixed, the paper will be rejected. 

**The 7 Critical Blockers:**
1. **Data Leakage:** The current split (`train_test_split`) divides individual 10-second segments randomly. Because segments overlap by 50%, the train and test sets contain highly correlated segments from the *same patient*, artificially inflating performance (AUC 0.97). We MUST implement a **patient-level split**.
2. **Dead Attention Module:** Code analysis reveals that the `AttentionEnhancedSNN.compute_attention()` method is *never called* in the `forward()` pass. The attention weights are dead parameters. Fig 16 in the paper is visualizing unused weights.
3. **SHAP Misrepresentation:** The code implements a simple feature perturbation/occlusion method but the paper claims it is using `shap.KernelExplainer` with Shapley values. 
4. **Parameter Mismatch:** The paper text contradicts the code (e.g., paper says SNN $\beta=0.95$ and dropout $0.3$; code uses $\beta=0.5$ and dropout $0.5$. Paper says 3 ResNet blocks; code uses 4).
5. **False XAI Claims:** The paper claims "85% concordance with expert annotations", but no expert evaluation was ever conducted.
6. **Double-Weighting in Loss:** The `FocalLoss` implementation double-counts class imbalance by applying both a manual `alpha` multiplier *and* passing `class_weights` to `CrossEntropyLoss`.
7. **"Forecasting" Terminology:** The system classifies fixed segments, but "forecasting" implies temporal prediction of future events. This must be reframed.

### The Strategy
To fix Blocker 1 without waiting 14 hours for the SimCLR encoder to retrain, we will **freeze the pretrained encoder** (which was trained unsupervised on interictal data, so it doesn't leak class boundaries) and **retrain only the SNN classifier head** using the new patient-level splits. This is scientifically valid and fast (~2-3 hours).

---

## 🛠️ Part 2: Task Execution Tracking

### Phase 1: Data Pipeline Fix (CPU Bound)
We cannot do patient-level splits without patient IDs. The original pipeline discarded patient IDs when saving the segments.

- [ ] **Task 1.1: Edit `seizure_forecasting_pipeline.ipynb`**
  - Modify `process_eeg_data()` to track patient IDs (using the file index as the ID).
  - Add logic to save `data/patient_ids_preictal.npy` and `data/patient_ids_interictal.npy`.
- [ ] **Task 1.2: Execute Notebook**
  - Run the entire preprocessing pipeline to generate the new `.npy` files. 
  - *Expected Time: 1-3 hours (CPU bound).*

### Phase 2: Core Model Fixes (`improved_seizure_forecaster.py`)
- [ ] **Task 2.1: Fix Dead Attention Module**
  - In `AttentionEnhancedSNN.forward()`, insert `features = self.compute_attention(features)` immediately before the SNN loop (around line 768).
- [ ] **Task 2.2: Implement Patient-Level Split**
  - Create `prepare_seizure_data_patient_level()` to replace the old split function.
  - Load the new patient ID arrays.
  - Use `sklearn.model_selection.GroupShuffleSplit` to create train/val/test splits grouped by patient ID.
- [ ] **Task 2.3: Fix Focal Loss Double-Weighting**
  - In `FocalLoss.forward()`, remove the scalar `self.alpha *` multiplier, allowing the standard PyTorch `weight` parameter to handle imbalance.
- [ ] **Task 2.4: Setup Classifier-Head-Only Retraining**
  - Freeze the encoder: `combined_model.requires_grad_(False)`.
  - Ensure the optimizer only receives parameters with `requires_grad=True` (the SNN layers).
- [ ] **Task 2.5: Add Parameter Counting**
  - Add a print statement: `sum(p.numel() for p in model.parameters() if p.requires_grad)` and for all params.
- [ ] **Task 2.6: Implement EEGNet Baseline**
  - Add a standard `EEGNet` PyTorch class.
  - Create a training block to train EEGNet on the *exact same* patient-level splits to provide a fair baseline for Table 2.

### Phase 3: XAI Script Fixes (`xai_explain_seizure_forecaster.py`)
- [ ] **Task 3.1: Rename SHAP to Perturbation Analysis**
  - Rename `compute_shap_values` to `compute_perturbation_importance`.
  - Update all plot labels, print statements, and DataFrame columns from "SHAP" to "Perturbation".
- [ ] **Task 3.2: Expand XAI Evaluation Scope**
  - Modify the script to run XAI (LIME, IG, Perturbation) on **20 preictal samples** instead of just 1.
  - Aggregate the results to show mean importance across the 20 samples.
- [ ] **Task 3.3: Update Brainmap Script (`enhance_xai_with_brainmap.py`)**
  - Ensure any references to SHAP are renamed to Perturbation Sensitivity.

### Phase 4: Training & Evaluation Execution (GPU Bound)
- [ ] **Task 4.1: Run `improved_seizure_forecaster.py`**
  - Train the fixed SNN classifier and the EEGNet baseline.
  - Save the new metrics (Accuracy, F1, AUC, Recall) and plots.
  - *Expected Time: 2-3 hours.*
- [ ] **Task 4.2: Run `xai_explain_seizure_forecaster.py`**
  - Generate the new aggregated XAI plots for the 20 samples.
  - *Expected Time: 3-5 hours (can run overnight).*

### Phase 5: Paper Revision Writing (`research-paper-draft.md`)
- [ ] **Task 5.1: Terminology & Tone Cleanup**
  - Replace "forecasting" with "preictal segment classification" or "detection" throughout.
  - Remove all hyperbolic words ("groundbreaking", "remarkable", "unprecedented", "prognostic prowess").
- [ ] **Task 5.2: Methodological Corrections**
  - Update Architecture description to match code (4 blocks, not 3).
  - Correct SNN parameters in text ($\beta=0.5$, dropout=$0.5$, slope=$30$).
  - Describe the new patient-level split protocol and freezing of the encoder.
- [ ] **Task 5.3: XAI Claim Corrections**
  - Change all "SHAP" references to "Perturbation Sensitivity Analysis" (or Occlusion).
  - **DELETE** the 85% expert concordance claim. Replace with qualitative alignment to temporal/central lobe literature.
- [ ] **Task 5.4: Results & Tables**
  - Update all performance metrics with the new patient-level results.
  - Update **Table 2** to include columns for Population (Adult/Neonatal), Task (Event/Segment), and add a disclaimer footnote. Add the EEGNet baseline row.
  - Add an Ablation Table.
- [ ] **Task 5.5: References & Formatting**
  - Fix Ref [1] DOI (Stevenson dataset) to `10.1038/sdata.2019.39`.
  - Fix reference numbering shift (SHAP/LIME).
  - Add the 3 reviewer-suggested DOIs to the Introduction.
  - Fix Table/Figure cross-references (e.g., Table III -> Table 6, Table 2 -> Table 5).
- [ ] **Task 5.6: Cover Letter**
  - Draft a point-by-point response letter to Reviewer #1 detailing all the above changes.

---

## 📊 Expected Outputs & Metrics to Track

When Phase 4 is complete, we must gather the following for the paper:

1. **Performance Metrics (SNN):** Test AUC, Test F1, Test Recall, Test Accuracy. *(Expect these to drop compared to the original 0.97 AUC due to the stricter patient-level split. This is normal and expected).*
2. **Performance Metrics (EEGNet Baseline):** Test AUC, Test F1, Test Recall, Test Accuracy.
3. **Visualizations Needed for Paper:**
   - ROC Curve & PR Curve (Updated)
   - Confusion Matrix (Updated)
   - Perturbation Channel Importance Bar Chart (Aggregated over 20 samples)
   - LIME Explanation Plot (Aggregated over 20 samples)
   - Integrated Gradients Saliency Map
   - Attention Weights Plot (Now valid because the module is connected)

---
*Note to Agent: Use this file to check off `[x]` tasks as they are completed and report progress to the user based on these phases.*
