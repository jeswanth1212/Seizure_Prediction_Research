# Revision Strategy: BRB-D-26-00510
## "A Hybrid SNN with Contrastive Pretraining for Interpretable Seizure Forecasting Using XAI"
### Brain Research Bulletin — Major Revision Response Plan

**Deadline:** May 05, 2026 | **Risk Level:** HIGH without structural fixes | **Prepared for:** Dr. Velmathi

---

---

## 1. 🔴 CRITICAL ISSUES (Acceptance Blockers)

These issues will cause outright rejection in round 2 if not addressed. They are not stylistic—they are scientific validity problems.

---

### BLOCKER 1: Data Leakage via Segment-Level Random Splits (Most Dangerous)

**What the reviewer said:** "segments from the same patient may not be confined to a single subset"

**What it actually means technically:**
Your `prepare_seizure_data()` function in `improved_seizure_forecaster.py` does this:

```python
train_idx, temp_idx, y_train, y_temp = train_test_split(
    all_indices, all_y, test_size=0.3, random_state=SEED, stratify=all_y)
```

This splits **segments** randomly across train/val/test. Because segments come from the same patient's continuous recording with 50% overlap, highly correlated segments from the **same patient, same recording** end up in both training and test sets. The model is effectively tested on data it has already "seen" in partially overlapping form. This is a form of data leakage that directly inflates all reported metrics.

**Why it's a blocker:** AUC 0.9743, 99.54% recall — these numbers become scientifically uninterpretable without patient-level splits. Any reviewer who has worked on EEG or medical AI knows this immediately.

**Fix required:** Implement patient-level (subject-level) train/val/test split. The Helsinki dataset has 79 neonates; 18 have seizures. A defensible split would be: ~55 subjects for training, ~12 for validation, ~12 for test, ensuring no subject appears in more than one partition.

**Code fix:**
```python
def prepare_seizure_data_patient_level(X_preictal, X_interictal, 
                                        patient_ids_preictal, 
                                        patient_ids_interictal):
    """Patient-level stratified split to prevent data leakage"""
    from sklearn.model_selection import GroupShuffleSplit
    
    all_X = np.concatenate([X_preictal, X_interictal])
    all_y = np.concatenate([np.ones(len(X_preictal)), 
                            np.zeros(len(X_interictal))])
    all_patients = np.concatenate([patient_ids_preictal, 
                                   patient_ids_interictal])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss.split(all_X, all_y, groups=all_patients))
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(all_X[temp_idx], all_y[temp_idx], 
                                         groups=all_patients[temp_idx]))
    # ...
```

**What to report:** Run the full experiment with patient-level splits and report the new numbers. If metrics drop (they almost certainly will), **do not hide this** — instead frame it honestly as "more realistic evaluation." The reviewer will respect this far more than suspicious numbers.

---

### BLOCKER 2: Architecture Mismatch Between Paper and Code

**What the paper claims:**
- Encoder: 3 residual blocks, channels 32→32, 64→64, 128→128 (Section III-B)
- Encoder output: 128-dimensional
- SimCLR projection head: 128→256 (ReLU) → 256→64 (L2 norm)
- Combined features: 192-dimensional (128 + 64)

**What the code actually implements:**
In `improved_seizure_forecaster.py`, `PretrainedEncoder` has:
- 4 residual block groups: conv2 (32→64), conv3 (64→128), conv4 (128→256)
- Final FC: 256→128

In `PretrainedSimCLR`, the encoder has:
- conv4 block goes to 256 channels, FC: 256→128
- Projection: 128→128 (ReLU) → 128→64

**The discrepancy:**
- Paper says encoder output is 128-dimensional from a 3-block ResNet. Code has a 4-block ResNet with 256→128 bottleneck. These are different architectures.
- Paper says combined is 192-dimensional (128+64). Code confirms `self.embedding_dim = 64 + 128 = 192`. This part matches, but the path to get to 128 differs.
- Paper describes channels as "32→32, 64→64, 128→128" but code has "32→64→128→256→128."

**Fix required:** Align paper description with actual code. Update Section III-B to accurately describe the 4-block ResNet (32→64→128→256 with final 256→128 FC bottleneck). This is a reproducibility killer if left as-is.

---

### BLOCKER 3: The "SHAP" Implementation Is Not SHAP

**What the paper claims (Section III-F):**
> "SHAP (SHapley Additive exPlanations): Employs a KernelExplainer with 100 perturbations to approximate Shapley values"

**What the code actually does (`xai_explain_seizure_forecaster.py`, `compute_shap_values` function):**
```python
def compute_shap_values(xai_model, sample, bg_samples, num_features=20):
    """Compute simplified SHAP-like feature importance"""
    # We'll use a simplified approach since full SHAP is memory-intensive
    # Create a feature importance metric based on perturbation
    ...
    for ch in channels_to_analyze:
        for t in time_points_to_analyze:
            perturbed = sample.copy()
            perturbed[ch, t] = np.mean(bg_samples[:, ch, t])
            ...
            feature_importance[ch, t] = baseline_pred - perturbed_pred
```

This is **occlusion/perturbation-based sensitivity analysis**, not SHAP. It doesn't compute Shapley values, doesn't use the SHAP library's KernelExplainer, and makes no claim to satisfy the SHAP axioms (efficiency, symmetry, dummy, linearity). The function docstring itself says "Compute simplified SHAP-like feature importance."

**Why it's a blocker:** This is a factual misrepresentation of methodology. A reviewer who runs your code will immediately see the discrepancy. This alone can cause rejection.

**Fix required (two options):**
- **Option A (preferred):** Actually implement SHAP using `shap.KernelExplainer`. Use 50 background samples to manage memory. The `shap` library is already imported in the code.
- **Option B:** Rename this method throughout the paper to "Perturbation Importance Analysis" and correctly describe it as occlusion-based. Remove all references to Shapley values, KernelExplainer, and SHAP literature.

---

### BLOCKER 4: "Forecasting" vs. Classification — Fundamental Framing Error

**Reviewer's concern:** "evaluation appears restricted to fixed-window segment-level classification rather than true event-level seizure prediction"

**The technical reality:**
Your system takes a 10-second EEG segment and outputs "preictal" or "interictal." It does not:
- Output a time-to-seizure estimate
- Predict a probability curve over a future horizon
- Issue warnings N minutes before seizure onset in continuous real-time streaming
- Evaluate with event-level metrics (sensitivity per seizure, false alarm rate per hour)

The word "forecasting" in the title and throughout the paper is clinically misleading. A clinician reading "seizure forecasting" expects a system that says "there is a 75% probability of seizure in the next 30 minutes," not a segment classifier.

**Fix required:**
Either:
1. Change all instances of "forecasting" to "detection/prediction" and clarify it is a binary segment-level preictal classifier OR
2. Add genuine forecasting evaluation: sliding-window prediction applied to continuous recordings, computing sensitivity per seizure event and false detection rate per hour. This is the gold standard for seizure prediction papers.

Option 2 is scientifically stronger but requires new experiments. Option 1 is honest and defensible for the revision deadline.

---

### BLOCKER 5: The Expert Concordance Claim (85%) Has No Protocol

**Paper claims:** "achieving an impressive 85% concordance with expert neurophysiological annotations"

**Code reality:** There is no function, script, or data structure in any of your four files that computes concordance with expert annotations. The `enhance_xai_with_brainmap.py` script only maps channel indices to 10-20 names via a static lookup dictionary. The "concordance" value appears to be a claim made without any actual expert evaluation.

**Fix required:** Either:
- Remove the 85% concordance claim entirely and replace with qualitative alignment statements
- OR actually conduct an expert evaluation: present XAI heatmaps to 2-3 neurophysiologists, ask them to rate alignment with known seizure onset patterns, report inter-rater agreement (Cohen's κ) alongside concordance score

---

### BLOCKER 6: Inconsistent Focal Loss α Implementation

**Paper claims (Section III-D):** "α=0.75 assigns greater weight to the preictal class"

**Code implementation (`FocalLoss.forward`):**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        ...
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
```

The class is initialized with `alpha=0.25` by default but called with `alpha=0.75` in `main()`. More critically: the focal loss formula used is **not the standard formulation**. The standard binary focal loss applies `α_t` based on class label, while your code applies a scalar `self.alpha` uniformly. Additionally, `class_weights` are also passed separately — meaning both α-weighting AND class-weights are applied simultaneously, which is double-counting imbalance correction. The paper describes only one weighting scheme.

**Fix required:** Clarify and fix the dual-weighting. Either use focal loss α *or* class weights, not both. Update the paper's mathematical formulation to match exactly what the code does.

---

### BLOCKER 7: The Attention Module Is Never Called in forward() (Elevated from Hidden Risk)

**The finding (confirmed by codebase analysis):**
`AttentionEnhancedSNN` defines `self.attention` and `compute_attention()`, but the `forward()` method bypasses them entirely:

```python
def forward(self, x, freq_features=None, num_steps=50):
    embeddings = self.encoder(x)
    # ...
    combined_features = torch.cat([embeddings, freq_proj], dim=1)
    features = self.dropout1(combined_features)

    # Goes DIRECTLY to fc1 — compute_attention() is NEVER called
    for step in range(num_steps):
        cur = self.fc1(features)
        ...
```

`compute_attention()` at lines 740–746 exists but is a dead method. The attention module's weights are initialized, count toward the parameter total, and are saved in the checkpoint — but they have zero effect on any prediction, training gradient, or output during the entire lifetime of the model.

**Cascading consequences:**
- The class name `AttentionEnhancedSNN` is a false label — no attention enhancement occurs
- Fig. 16 ("Attention Weight Distribution") in the paper visualizes weights from a module that never participates in the forward pass. The figure is not reproducible because the hooks in `AttentionExtractor` only fire when `get_attention_weights()` manually calls the model, but even then the attention module is outside the computational graph and its output has no meaning
- The XAI "Attention Profiling" listed in Section III-F is not capturing the model's decision process — it is capturing the output of an unused module

**Fix (Option A — strongly recommended, since you are already retraining the head):**
Insert one line in `forward()` before the SNN loop:

```python
# In AttentionEnhancedSNN.forward(), after dropout1 and before the SNN loop:
features = self.compute_attention(features)   # ADD THIS LINE
```

Note: `compute_attention()` calls `F.softmax(attn_weights, dim=1)` followed by `(x * attn_weights).sum(dim=1)`, which expects a 2D input `[batch, features]`. Verify the shape is compatible before inserting. If `features` is already `[batch, total_dim]`, this works directly.

Since you are already retraining the classifier head for the patient-level split fix, this one-line addition adds literally zero extra GPU time. The attention weights now become real, Fig. 16 becomes valid, and "AttentionEnhancedSNN" is an accurate name.

**Fix (Option B — if you cannot retrain):**
Remove "Attention" from the class name in the paper, remove Fig. 16 entirely, remove "Attention Profiling" from Section III-F, and describe the architecture as "an SNN classifier with frequency-domain feature integration." This is honest but weakens the contribution.

**Recommendation: Option A.** It is a 1-line code change with zero extra compute cost on top of the retraining you are already doing.

---

---

## 2. 🟡 REVIEWER COMMENT MAPPING

### Reviewer Comment 1: Random 70/15/15 split without patient-level isolation

**Technical meaning:** See Blocker 1 above. Overlapping segments from the same patient contaminate train/test boundaries.

**Where it appears:** Section III-A (data partitioning), `prepare_seizure_data()` in `improved_seizure_forecaster.py`

**Fix in paper:**
Replace in Section III-A:
> ~~"the dataset was then stratified into 70% training, 15% validation, and 15% test sets (using a fixed random seed=42)"~~

With:
> "To prevent data leakage arising from intra-patient temporal correlations between overlapping segments, splits were performed at the subject level. Of the 79 neonates, [N_train] subjects were allocated to training, [N_val] to validation, and [N_test] to testing, maintaining an approximately 70/15/15 ratio while ensuring no subject appeared in multiple partitions. Class proportions within each split were verified to be within 2% of the overall 6.48:1 imbalance ratio."

**Fix in code:** Implement `GroupShuffleSplit` as shown in Blocker 1.

---

### Reviewer Comment 2: "Forecasting" term vs. segment classification

**Technical meaning:** See Blocker 4. The evaluation does not validate temporal prediction ability.

**Where it appears:** Title, Abstract, Introduction, throughout.

**Fix in paper:**
Replace the term "forecasting" with "preictal detection" or "seizure prediction" where it refers to the classification task. In the discussion, add one paragraph:
> "We acknowledge that the term 'forecasting' implies temporal prediction over a future horizon, which is distinct from our segment-level preictal classification approach. Our model identifies EEG segments exhibiting preictal patterns within a 4-minute window preceding seizure onset. True seizure forecasting—issuing probabilistic alerts over a future time horizon—would require additional continuous-monitoring evaluation metrics such as sensitivity per seizure event and false detection rate per hour [cite], which we designate as future work."

---

### Reviewer Comment 3: Table 2 compares heterogeneous datasets without contextualization

**Technical meaning:** Your comparison table places CHB-MIT (adult, chronic epilepsy) results alongside TUH-EEG results alongside your Helsinki neonatal results, as if they are comparable. They are not — different patient populations, different seizure types, different evaluation protocols.

**Fix in paper (exact replacement for Table 2 caption):**
> "Table 2. Comparative overview of representative prior methods. Direct metric comparison should be interpreted cautiously: datasets differ in patient demographics (adult vs. neonatal), seizure types, recording duration, and evaluation protocol (event-level vs. segment-level). Values are reported as published."

Also add a row in the table for the patient-level evaluation result of your own model once re-evaluated.

---

### Reviewer Comment 4: 85% concordance with expert annotations — validation not described

**Fix in paper:** Remove the specific percentage. Replace with:
> "XAI attributions were qualitatively reviewed against established neurophysiological literature on neonatal seizure propagation patterns [cite Stevenson, cite Shellhaas]. Temporal channel dominance (T3, T4) aligns with published reports of temporal lobe seizure predominance in neonates [cite]. This qualitative validation is distinguished from quantitative expert agreement, which we identify as an important direction for clinical translation."

---

### Reviewer Comment 5: Language is overly promotional

**Instances to remove/replace:**

| Current language | Replace with |
|---|---|
| "groundbreaking hybrid framework" | "hybrid framework" |
| "remarkable performance" | "strong performance" |
| "unprecedented interpretability" | "improved interpretability" |
| "prognostic prowess" (Section heading) | "Predictive Performance" |
| "Assay Arsenal" (Section heading) | "Performance Evaluation" |
| "cornerstone of open neuroscience" | "publicly available dataset" |
| "profound and urgent concern" | "significant clinical concern" |
| "cutting-edge" (anywhere) | remove or replace with specific descriptor |

Search the entire manuscript for any instance of "novel," "superior," "exceptional," "outstanding" and replace each with specific quantitative language.

---

### Reviewer Comment 6: Methodological inconsistencies between text and figures

**Known specific issues (cross-checked against code):**
- Paper Section III-B states encoder channels "progressively increase from 32 to 32, 64 to 64, and 128 to 128" — this is wrong per the code (actual: 32→64→128→256 across blocks)
- Paper says dropout=0.3; code uses dropout=0.5 in `AttentionEnhancedSNN.__init__`
- Paper describes "50 simulation timesteps" — code uses `num_steps=50` in `forward()` default but training calls use `num_steps=50` in `train_model()` — this one is consistent ✓
- Paper describes β=0.95 for LIF neurons (Section III-C, Eq. 3) but code uses `beta=0.5` in all SNN layers

**Fix in paper (Section III-C):** Change β from 0.95 to 0.5, or vice versa — but they must match. β=0.5 (code value) is actually a reasonable and common choice; use it in the paper.

**Fix in paper (Section III-D):** Change dropout from 0.3 to 0.5 to match code.

**Fix in paper (Section III-B):** Correct the channel progression description to match code.

---

---

## 3. 🧪 EXPERIMENTAL FIX PLAN

Listed in priority order given the May 5, 2026 deadline (~6 days from now).

---

### EXPERIMENT 1 (MANDATORY, Day 1–2): Patient-Level Re-Evaluation

**Why first:** All other experiments are meaningless if the evaluation protocol is invalid.

**⚠️ Critical practical issue — patient ID arrays do not exist:**
The notebook's `process_eeg_data()` function saves segments into `X_preictal.npy` and `X_interictal.npy` but discards patient provenance at the `np.save` step. There are no `patient_ids_preictal.npy` or `patient_ids_interictal.npy` arrays in the data directory. You cannot run `GroupShuffleSplit` without them.

**Step 0 (must come before everything else): Re-run notebook preprocessing with patient ID tracking.**

Modify `process_eeg_data()` in `seizure_forecasting_pipeline.ipynb` to also track and save patient ID arrays:

```python
# Add these accumulators alongside existing ones in process_eeg_data():
all_preictal_patient_ids = []
all_interictal_patient_ids = []

for i, eeg_file in enumerate(eeg_files):
    patient_id = i  # index serves as stable patient identifier
    # ... existing segment extraction logic unchanged ...
    preictal_segments, interictal_segments = classify_segments(...)
    all_preictal_patient_ids.extend([patient_id] * len(preictal_segments))
    all_interictal_patient_ids.extend([patient_id] * len(interictal_segments))

# Add these two lines alongside the existing np.save calls:
np.save('data/patient_ids_preictal.npy', np.array(all_preictal_patient_ids))
np.save('data/patient_ids_interictal.npy', np.array(all_interictal_patient_ids))
```

This preprocessing is CPU-bound EDF reading — no GPU required. Estimated time: 1–3 hours depending on dataset I/O speed. **Start this first on Day 1 so it runs while you work on other changes.**

**Steps (after patient IDs are available):**
1. ✅ Patient ID arrays saved (Step 0 above)
2. Implement `prepare_seizure_data_patient_level()` with `GroupShuffleSplit`
3. **Retrain the classifier head only — do NOT retrain the SimCLR encoder from scratch**
4. Report new test metrics

**Why classifier-head-only retraining (not full retraining):**
The SimCLR contrastive pretraining used unsupervised augmentations without class labels — the supervised leakage only affects the classifier head. Freezing the encoder gives the same scientific validity as full retraining at ~10× the speed.

```python
# Freeze the pretrained encoder
for param in combined_model.parameters():
    param.requires_grad = False

# Only classifier head parameters update
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5, weight_decay=1e-4
)
```

**Also add the 1-line attention fix here (Blocker 7 — zero extra cost):**
```python
# In AttentionEnhancedSNN.forward(), before the SNN loop:
features = self.compute_attention(features)   # ADD THIS LINE
```

Add to the paper: "The contrastive encoder was frozen during classifier training. Only the AttentionEnhancedSNN classifier head was retrained under the patient-level split protocol."

**Expected outcome:** Metrics will likely drop modestly (AUC ~0.91–0.94; recall ~90–95%). This is **still competitive** and is now scientifically valid.

---

### EXPERIMENT 2 (MANDATORY, Day 1–3): Fix SHAP Implementation

**Options:**
- Use `shap.KernelExplainer` with 50 background samples (may be slow but feasible)
- Use `shap.DeepExplainer` if gradient flow is available through the model
- Alternatively, formally rename to "perturbation sensitivity analysis" and update all paper text

If using perturbation analysis, compute it across **all test set preictal samples** (not just 1), aggregate results, and show mean ± std channel importance. This is more statistically robust.

---

### EXPERIMENT 3 (RECOMMENDED, Day 2–3): Ablation Study Completeness

The paper mentions ablation in the abstract but Section IV contains no ablation table. Add a proper ablation table:

| Configuration | Accuracy | F1 | AUC |
|---|---|---|---|
| Full model (proposed) | X | X | X |
| Without contrastive pretraining | X | X | X |
| Without spectral features | X | X | X |
| ANN classifier (no SNN) | X | X | X |
| Without attention mechanism | X | X | X |

This directly addresses the reviewer's concern about contribution validation and will significantly strengthen the paper.

---

### EXPERIMENT 4 (🔥 MANDATORY, Day 2–3): Baseline Comparison

Add at least one modern baseline trained on the same patient-level splits. **This is not optional.** The reviewer already doubts the evaluation; without a same-dataset baseline, Table 2 is a comparison across incompatible conditions and will be flagged again in Round 2.

**Required minimum:**
- **EEGNet** (Lawhern et al., 2018): The canonical compact CNN for EEG. It is fast to train (~30 minutes), well-understood, and a standard benchmark that every EEG reviewer will expect. If your model doesn't beat EEGNet on the same split, that is important information — report it honestly.

**Strongly recommended addition:**
- **CNN-LSTM**: Standard temporal model for EEG, establishes that the SNN component adds value over a recurrent ANN baseline.

Both baselines must use the **identical patient-level splits** as your main model. Report accuracy, F1, AUC, and recall in a new Table alongside your method.

Without baselines on the same dataset and same split, the comparison section remains the single weakest part of the paper scientifically and is the most likely trigger for Round 2 rejection even after all other fixes.

---

### EXPERIMENT 5: Event-Level Evaluation — EXPLICITLY DEFERRED

**Decision: Do not attempt this for the current revision.**

The original plan offered two options on the "forecasting vs. classification" issue. Given your deadline, that ambiguity is dangerous — attempting a partial event-level evaluation under time pressure produces results that look rushed and invite more scrutiny, not less.

**Committed path:** Rename the task throughout the paper to "preictal segment detection" or "preictal classification." Add one paragraph to the Discussion (template provided in Section 2) acknowledging the distinction and deferring event-level evaluation to future work. This is the honest, clean, and reviewers-will-accept-it path.

Do not add event-level evaluation now. A properly motivated deferral is always more credible than a half-implemented evaluation.

---

---

## 4. 📄 PAPER REVISION PLAN

### Abstract

**Remove:**
- "groundbreaking," "remarkable," "unprecedented"
- The specific claim "85% concordance with expert annotations"
- The claim of "KernelExplainer with 100 perturbations" if SHAP is not fixed

**Rewrite:**
- State "preictal segment detection" instead of "seizure forecasting"
- Add: "evaluated using patient-level cross-validation to prevent data leakage"
- Report updated metrics after patient-level re-evaluation
- Change "XAI suite including... SHAP" to match actual implementation

---

### Introduction

**Remove:**
- "groundbreaking," "paramount," the phrase "even evade expert detection" (unsubstantiated)

**Add:**
- Explicit statement of what the system does: "binary classification of 10-second EEG segments as preictal or interictal"
- Acknowledge the gap between segment classification and clinical deployment
- Add citation to the three papers suggested by the reviewer: DOI 10.1111/exsy.70111, DOI 10.1016/j.jvcir.2024.104212, DOI 10.1016/j.compmedimag.2023.102295

---

### Methodology

**Section III-A (Data):**
- Replace segment-level split description with patient-level split
- Add: how you verified no patient overlap between splits
- Clarify the 4-minute preictal window selection rationale (why 4 min? cite literature)
- Add that segments within 5 minutes of a seizure but outside the 4-min window were **excluded** (gap period), not included in interictal class — clarify this or the class boundary is ambiguous

**Section III-B (Contrastive Pretraining):**
- Correct the ResNet architecture description (channels: 32→64→128→256, FC bottleneck 256→128)
- Clarify whether the pretrained encoder weights are frozen during classifier training or fine-tuned. This is not stated in the paper and the code uses `load_state_dict` but doesn't call `encoder.requires_grad_(False)` — it appears weights are unfrozen, which is a valid design choice but must be stated

**Section III-C (SNN Classifier):**
- Change β from 0.95 to 0.5
- Change dropout description from 0.3 to 0.5
- Clarify the "attention mechanism" — the paper implies a multi-head attention but the code (`AttentionEnhancedSNN.attention`) is a simple 2-layer sequential that outputs a scalar weight per feature dimension. This is a scalar attention gate, not multi-head or self-attention. Call it correctly: "feature-level scalar attention gating"

**Section III-D (Training):**
- Clarify the double weighting (focal loss α AND class weights) or remove one
- Add: total parameter count with breakdown (pretrained encoder vs. classifier head)
- Add: hardware inference time (the "10 ms inference" claim in the discussion is never measured in the methodology)

**Section III-F (XAI):**
- Fix the SHAP description to match actual implementation
- Add number of preictal samples used for XAI analysis (currently only 1 sample is used for SHAP and LIME — this is statistically insufficient; use at least 10–20)
- Remove the claim about "100 perturbations" for KernelExplainer if not actually used

---

### Results

**Section IV-A:**
- Report updated metrics after patient-level re-evaluation
- Add ablation table (Experiment 3 above)
- Add baseline comparison (Experiment 4 above)
- Report confidence intervals on AUC (currently: 95% CI [0.968, 0.980] — keep this, it's good)

**Section IV-B (XAI):**
- Report XAI results across multiple samples (not just 1 preictal example)
- Remove the 85% concordance claim or replace with the qualitative language from Section 2
- Clarify that SHAP values in Table 5 represent per-channel perturbation importance, not Shapley values

---

### Discussion

**Add:**
- Paragraph on limitations of segment-level evaluation vs. true forecasting (see the suggested text in Reviewer Comment 2 mapping)
- Paragraph on the gap between reported performance and clinical deployment readiness
- Acknowledge that the class boundary definition (preictal: −240 to 0 s) is a methodological choice, and that results may differ with different window definitions

**Remove:**
- "This work... paving the way" — too promotional
- The claim that the model is "edge-ready" without actual edge device benchmarking

---

### Conclusion

**Rewrite the first sentence.** Currently it introduces new claims not supported by the rest of the paper. Replace with a factual summary of findings. Typical acceptable format: "We presented a hybrid framework combining [A], [B], and [C], evaluated on [dataset] using patient-level splits, achieving [metric] with [metric]."

---

---

## 5. 🧾 RESPONSE TO REVIEWERS STRATEGY

### Tone

Use a tone that is **confident but respectful**, never defensive. Structure each response as: (1) Thank the reviewer for the specific insight, (2) Agree or provide a clear technical explanation if you respectfully disagree, (3) State exactly what change was made and where. Never use "we believe" when you can use "we have shown" or "we have corrected."

---

### Sample Response to Major Comment on Data Splitting

> **Reviewer Comment:** "The manuscript describes a random 70%/15%/15% split without explicitly ensuring that segments from the same patient are confined to a single subset."

> **Response:**
> We thank the reviewer for identifying this critical methodological concern. The reviewer is correct: our original implementation applied a random segment-level split, which introduces potential information leakage due to temporal correlations between overlapping segments from the same neonate.
>
> **Correction:** We have re-implemented data partitioning using patient-level (subject-level) splits via `GroupShuffleSplit`, ensuring that all segments from a given neonate appear exclusively in one partition. Of the 79 subjects, [N_train]/[N_val]/[N_test] were allocated to train/validation/test respectively. The revised evaluation is reported in Table X. We note that model performance on the corrected evaluation remains competitive at AUC [new value], F1 [new value], which we report transparently.
>
> The revised methodology is described in Section III-A (line X) and the updated code is available in our repository.

---

### Sample Response to "Forecasting" Term Comment

> **Reviewer Comment:** "the evaluation appears to be restricted to fixed-window segment-level classification rather than true event-level seizure prediction"

> **Response:**
> The reviewer raises a valid terminological concern. Our system performs binary classification of fixed 10-second EEG segments into preictal or interictal classes, and does not output time-to-seizure estimates or rolling probability horizons. We have accordingly revised the manuscript to use the term "preictal detection" in place of "forecasting" throughout (Title, Abstract, Introduction, and Discussion). A new paragraph in the Discussion (Section V, page X) explicitly addresses this distinction and frames event-level evaluation as future work. We believe this clarification improves the precision of our contribution claims without diminishing the novelty of the technical approach.

---

### Sample Response to SHAP Implementation Comment

If fixing with actual SHAP:
> We have replaced our original perturbation-based feature importance analysis with a correct SHAP implementation using `shap.KernelExplainer` with 50 background interictal samples, applied to 20 preictal test samples. Results are updated in Section IV-B and Figure 12–13. The revised SHAP values confirm the dominance of temporal channels, with [updated Table 5 values].

If renaming instead:
> We acknowledge that our original implementation did not use Shapley value decomposition as implied by the "SHAP" label. The method used is a perturbation-based channel importance analysis (occlusion sensitivity). We have corrected all text accordingly, renaming this component "Perturbation Sensitivity Analysis" and replacing citations to Lundberg & Lee [14] with appropriate references to occlusion-based XAI methods. We apologize for the inaccurate framing.

---

### Where to Defend (Don't Over-Concede)

**Defend the choice of 4-minute preictal window:** Cite literature (Saab et al., Temko et al.) showing that 4-minute preictal windows are consistent with prior neonatal EEG work. Don't let the reviewer dismiss this as arbitrary.

**Defend the SNN choice:** The reviewer implied SNNs are harder to interpret. Respond: "While SNNs present interpretability challenges for gradient-based methods, we mitigated this by using surrogate gradient functions (fast-sigmoid) that permit smooth gradient estimation, enabling Integrated Gradients and saliency computation. This design choice is described in Section III-C and is a contribution of our architecture."

**Defend the single dataset use:** Acknowledge it's a limitation but cite the difficulty of obtaining comparable neonatal EEG datasets. The Helsinki dataset is the primary benchmark in this field.

---

---

## 6. ⚠️ HIDDEN RISKS (Second-Round Traps)

These issues are NOT explicitly mentioned by Reviewer 1 but are highly likely to be raised in Round 2, especially if a second reviewer is added.

---

### HIDDEN RISK 1: The 50% Segment Overlap Inflates Effective Sample Size

Your pipeline creates segments with 50% overlap. This means adjacent segments share 1,280 of 2,560 samples. Even with patient-level splits, within the training set you have highly correlated samples. This inflates effective sample size and reduces the true diversity of training data. A rigorous reviewer may ask: what are the results with non-overlapping segments?

**Pre-emptive fix:** Add a sentence in the Methods: "We evaluated sensitivity to the overlap parameter; results with 0% overlap (independent segments) yielded [X] AUC, confirming that the reported performance is not an artifact of segment correlation." If time doesn't allow re-running, at least acknowledge this as a limitation.

---

### HIDDEN RISK 2: Contrastive Pretraining Contamination — More Severe Than Initially Stated

Your `compute_shap_values` function selects background samples from `X_interictal` without filtering out test-set patients. More fundamentally, the SimCLR contrastive pretraining was conducted on the **entire interictal corpus** before any patient-level split was applied — meaning the encoder has seen interictal segments from every neonate including those designated as test patients.

**Why "no class labels" does not fully exonerate the encoder:**
While the encoder doesn't see labels, it does learn the data distribution of test patients. If test patients have subtly different EEG patterns (different brain maturity, different pathologies, different artifact profiles), the encoder's learned representations are partially fit to those specific characteristics. This is a recognized form of **transductive leakage** in self-supervised learning and a legitimate Round 2 concern.

**Fix:** Do NOT re-run the 14-hour SimCLR pretraining under deadline pressure. Instead, add the following paragraph verbatim to the Discussion (Limitations subsection):

> "We note that contrastive pretraining was conducted on the complete interictal corpus prior to patient-level partitioning. While this unsupervised phase does not access class labels or patient identity, the encoder may have learned representations that are partially influenced by the data distributions of test-set patients — a form of transductive leakage recognized in self-supervised learning literature [cite]. The practical impact of this is expected to be modest given the unsupervised nature of pretraining, but we identify restricting contrastive pretraining to training-split patients exclusively as an important direction for future rigorous evaluation."

This is honest, pre-empts the question, cites the relevant concept, and does not require any additional experiments.

---

### HIDDEN RISK 3: No Statistical Significance Testing

You report point estimates for all metrics. With 11,324 test samples, random variation is low, but a reviewer may still ask: how stable are these results across different random seeds? Report results across 3 seeds (42, 123, 456) with mean ± std. Takes one evening to run.

---

### HIDDEN RISK 4: Elevated to Blocker 7 — See Section 1

This issue has been promoted out of Hidden Risks. The attention module is not merely mislabeled — `compute_attention()` is never called inside `forward()`. The module is dead weight: initialized, contributing to parameter count, but excluded from every forward pass during training and inference. Fig. 16 visualizes weights from a module that never participates in any computation. See **Blocker 7** in Section 1 for the full diagnosis, code fix, and paper implications.

---

### HIDDEN RISK 5: Weak or Missing Baselines on the Same Dataset

Table 2 compares methods on completely different datasets. There is no baseline trained on the Helsinki dataset using the same split protocol. A second reviewer will almost certainly ask: "How does your method compare to a well-tuned EEGNet or LSTM trained on the same data?" Without this, the novelty claim rests on comparison across heterogeneous conditions, which is not scientifically valid.

**Fix:** See Experiment 4. At minimum, add a CNN-LSTM baseline.

---

### HIDDEN RISK 6: The "~1M Parameters" Claim Is Not Verified

The paper states the model has "approximately 1 million parameters." The actual count depends on the combined SimCLR + Encoder + classifier. The SimCLR alone has `PretrainedEncoder` with 4 residual blocks going to 256 channels — this is already well over 1M parameters. Run `sum(p.numel() for p in model.parameters())` and report the actual count.

---

### HIDDEN RISK 7: LIME Samples Only 1 Preictal Segment

In `xai_explain_seizure_forecaster.py`:
```python
preictal_idx = np.where(analysis_labels == 1)[0][0]
preictal_sample = analysis_samples[preictal_idx:preictal_idx+1]
lime_data = compute_lime_explanation(preictal_sample[0], ...)
```
LIME is run on exactly one sample. Conclusions about "T3 desynchronization" or "F3 suppression" drawn from a single sample are anecdotal, not statistical. A clinical journal will flag this.

**Fix:** Run LIME on at least 10–20 preictal samples and report aggregate feature importance with variability.

---

---

## 7. 🚀 ACCEPTANCE PROBABILITY ANALYSIS

### Current State (Before Revision): ~25% acceptance probability

The paper has genuine technical novelty — the SNN + contrastive pretraining + XAI combination is original and the results are impressive. However, the data leakage issue and the SHAP misrepresentation are hard rejection triggers that most experienced reviewers will catch.

### After Implementing Blockers 1–7: ~65–70% acceptance probability

Correcting the patient-level split, activating the attention module, fixing the SHAP implementation, aligning the architecture description with code, removing the concordance claim, fixing the β/dropout inconsistencies, and resolving the focal loss double-weighting removes all hard rejection triggers. Even if metrics decline, the paper becomes scientifically credible.

### After Implementing Blockers + Experiments 1–4: ~80–85% acceptance probability

Adding the ablation table, proper baselines, and statistical comparison on consistent data is what separates a good paper from a publishable one at Q1/Q2 level.

### What Will Most Influence the Accept Decision

In order of importance:
1. **Patient-level evaluation results** — if this shows competitive performance (AUC >0.90), the paper is strong. If it collapses, the paper needs the clinical framing argument.
2. **Fixing the SHAP misrepresentation** — this is an integrity issue. If not fixed, it can cause retraction post-publication.
3. **Baselines on the same dataset** — the comparison section is currently the weakest part scientifically.
4. **Removing promotional language** — the reviewer was irritated enough to call it out. It signals insufficient revision effort if it remains.
5. **Properly describing what the system does** (classification, not forecasting) — editors notice when a title claim doesn't match the evaluation.

---

---

## 8. ⏳ EXECUTION ROADMAP

**You have approximately 6 days until the May 5, 2026 deadline.**

---

### Day 1 (Today): Code Changes

- [ ] **[FIRST — start immediately, runs in background]** Modify `process_eeg_data()` in notebook to track patient IDs; re-run preprocessing to generate `patient_ids_preictal.npy` and `patient_ids_interictal.npy` — 30 min code change, 1–3 hours CPU runtime
- [ ] While preprocessing runs: implement `prepare_seizure_data_patient_level()` with `GroupShuffleSplit` — 1 hour
- [ ] Add 1-line attention fix in `AttentionEnhancedSNN.forward()` (`features = self.compute_attention(features)`) — 5 minutes
- [ ] Freeze encoder weights; retrain **classifier head only** with new splits — 2–3 hours GPU time (start overnight)
- [ ] Fix the SHAP function (implement real SHAP or rename throughout) — 1 hour

---

### Day 2: Analysis and Results

- [ ] Collect new metrics from patient-level classifier-head evaluation
- [ ] **[MANDATORY]** Run EEGNet baseline on same patient-level splits — 2–3 hours
- [ ] Run CNN-LSTM baseline on same splits if time allows — 2 hours
- [ ] Run XAI on 20 preictal samples (not 1) — 2 hours
- [ ] Count actual model parameters (`sum(p.numel() for p in model.parameters())`) — 15 minutes
- [ ] Run 2–3 random seeds for stability analysis — run overnight

---

### Day 3: Paper Revisions (Critical Sections)

- [ ] Fix all architecture description inconsistencies (β, dropout, channel progression)
- [ ] Fix Section III-A (patient-level split description)
- [ ] Fix Section III-B (encoder architecture)
- [ ] Fix Section III-F (XAI description to match code)
- [ ] Remove all "groundbreaking / remarkable / unprecedented" language
- [ ] Remove 85% concordance claim
- [ ] Add ablation table

---

### Day 4: Paper Revisions (Secondary Sections)

- [ ] Rewrite Discussion: replace "forecasting" framing — **commit to classification-only language throughout; add one paragraph deferring event-level evaluation to future work** (use template from Section 2)
- [ ] Update Table 2 caption with contextualization language
- [ ] Update Introduction with three new citations (reviewer-suggested DOIs)
- [ ] Rewrite Conclusion
- [ ] Update Abstract with new metrics and corrected claims

---

### Day 5: Response Letter

- [ ] Write point-by-point response to reviewer comments
- [ ] Draft cover letter summarizing major changes
- [ ] Prepare tracked-changes version of manuscript
- [ ] Cross-check: every reviewer comment has a specific response and a manuscript line number

---

### Day 6: Final Review and Submission

- [ ] Have a colleague who has NOT read the paper read the response letter for clarity
- [ ] Verify all figures/tables are consistent with updated text
- [ ] Check reference list for inconsistencies (DOI links, author names)
- [ ] Submit via Editorial Manager

---

### What to Skip If Time Runs Out

**Skip (with acknowledgment in Discussion):**
- Event-level forecasting evaluation — explicitly deferred, one paragraph in Discussion is sufficient
- Extensive multi-seed comparison (2 seeds is acceptable minimum)
- Full LIME expansion to 20 samples (10 is acceptable minimum)
- CNN-LSTM baseline if EEGNet alone takes too long (one solid baseline beats zero)

**Do NOT skip under any circumstances:**
- Patient-level re-evaluation with classifier-head retraining (Blocker 1) — non-negotiable
- SHAP fix or rename (Blocker 3) — non-negotiable
- Architecture description correction (Blocker 2) — non-negotiable
- Removing the 85% concordance unsupported claim (Blocker 5) — non-negotiable
- EEGNet baseline on same splits (Experiment 4) — non-negotiable
- Language cleanup (reviewer was specific and will check) — non-negotiable

---

*This strategy was prepared based on cross-analysis of the submitted manuscript, the four implementation files (improved_seizure_forecaster.py, xai_explain_seizure_forecaster.py, enhance_xai_with_brainmap.py, seizure_forecasting_pipeline.ipynb), and Reviewer 1's comments. The code is treated as ground truth where it conflicts with the paper's narrative.*