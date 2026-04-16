# Comprehensive Architecture: Interpretable Neonatal Seizure Forecasting Framework

This document provides a deep, technical, and comprehensive architectural breakdown of the hybrid Spiking Neural Network (SNN) framework developed for neonatal seizure forecasting. This system is designed to tackle the notoriously difficult problem of predicting seizures in newborns using non-invasive EEG signals, characterized by severe class imbalances, subtle signal patterns, and the critical clinical requirement for transparent predictions.

---

## 1. System Architecture Diagram

Refer arch.png

---

## 2. Technical Component Breakdown & Rationale

### 2.1 Dataset Curation and Signal Conditioning
**What it is:**
The framework utilizes the Helsinki University Hospital Neonatal EEG Seizure Dataset. The signals, captured at 256 Hz across 21 channels using the International 10-20 referential montage, are chunked into 10-second segments (2,560 samples) with a 50% overlap. 

**Technical Deep Dive:**
*   **Windowing:** A 10-second window is long enough to capture low-frequency delta components (0.5-4 Hz) while short enough to maintain high temporal resolution for forecasting.
*   **Normalization:** $x'_{c,t} = \frac{x_{c,t} - \mu_c}{\sigma_c}$. Applied *per-channel* independently.
*   **Class Imbalance:** Segments within 4 minutes of a seizure are labeled "Preictal". Those 5+ minutes away are "Interictal". This strategy creates a massive `6.48:1` interictal-to-preictal ratio, which perfectly mimics the sparse nature of seizures in a real NICU setting.

**Why it's necessary:** 
Neonatal EEG signals are highly non-stationary and frequently contaminated by artifacts (movement, sweat). Normalization removes DC drifts and harmonizes amplitudes, ensuring models learn morphological shapes rather than absolute voltage discrepancies.

### 2.2 Contrastive Pretraining Regimen (SimCLR architecture)
**What it is:**
An unsupervised representational learning foundation based on SimCLR. It trains an encoder using only the abundant, unannotated interictal data.

**Technical Deep Dive:**
*   **Augmentation Engine:** To create "positive pairs" of the same segment, the pipeline applies domain-specific perturbations: Gaussian noise ($\sigma = 0.1 \times \text{std}$), uniform temporal shifts ($\pm 10\%$), probabilistic channel dropout ($p=0.3$), pointwise masking ($p=0.5$), and multiplicative amplitude scaling ($0.8 - 1.2 \times$).
*   **1D ResNet Encoder:** The encoder begins with an initial convolution (32 channels), cascades through hierarchical residual blocks expanding up to 256 channels, and concludes with Global Average Pooling (GAP) resolving into a 128-dimensional latent vector.
*   **Projection Head:** A non-linear MLP mapping the 128-dim vector to a 64-dim contrastive manifold.
*   **NT-Xent Loss:** Normalized Temperature-scaled Cross Entropy loss ($\tau=0.5$) maximizes cosine similarity between identically augmented pairs while pushing apart unrelated sample pairs in the batch.

**Why it's necessary:** 
Deep learning requires vast amounts of labeled data, which is scarce in forecasting. By forcing the encoder to recognize that a noisy, slightly time-shifted piece of EEG is essentially the "same" brain state, the model develops highly robust, invariant filters for normal brain rhythms *before* it even begins learning what a seizure precursor looks like.

### 2.3 Spectral Feature Exaction
**What it is:**
The extraction of explicit frequency-domain biomarkers alongside the deep learning embeddings. 

**Technical Deep Dive:**
*   **Welch's Method:** Applied windowed Fourier transforms (segment length 256) to extract Power Spectral Densities (PSD).
*   **Targeted Bands:** Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), Gamma (30–100 Hz).
*   **Dimensionality Reduction:** $21 \text{ channels} \times 5 \text{ bands} = 105 \text{ features}$. A linear FC layer compresses this into a cohesive 32-dimensional subspace.

**Why it's necessary:** 
Convolutional networks are excellent spatial-temporal pattern matchers, but explicit frequency bands are established neurophysiological gold standards. Injecting deterministically calculated PSDs provides the model with "ground truth" biological anchors, significantly improving precision and grounding the latent space.

### 2.4 Fused Feature Space
**What it is:**
The concatenation of different representational regimes into a single master vector.
*   `128-dim` Encoder Output + `64-dim` SimCLR Projection + `32-dim` Spectral Projection = **`224-dim` Total Vector**.

### 2.5 SNN-Infused Classifier Scaffold (`AttentionEnhancedSNN`)
**What it is:**
The primary predictor engine. The 224-dim vector is passed through a dense layer, gated by attention, and simulated through biologically-inspired Spiking Neural Networks (SNNs).

**Technical Deep Dive:**
*   **Attention Mechanism:** A lightweight affine transformation (`tanh` -> normalization) applies a relevance heatmap over the 224 features, silencing irrelvant noise points.
*   **LIF Neurons (Leaky Integrate-and-Fire):** Implemented via `snnTorch`. The membrane potential $u[t]$ leaks over time ($\beta=0.5$) and is integrated to a threshold ($V_{th}=1.0$). If it crosses the threshold, a binary spike is emitted, and the potential drops. 
*   **Temporal Iteration:** The network is simulated iteratively over $T=50$ timesteps, passing recurrent membrane context forward, yielding a dynamic time-series evaluation of the single snapshot.
*   **Surrogate Gradients:** Because spikes are non-differentiable (Heaviside step functions), Backpropagation Through Time (BPTT) relies on a smooth surrogate gradient (`fast_sigmoid`, slope=25).
*   **Focal Loss:** $FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$. With $\alpha=0.75$ and $\gamma=2$, the loss radically penalizes misses on the rare preictal class compared to the common interictal class.

**Why it's necessary:** 
SNNs capture complex temporal and sequential memory far better than static FFNNs and are roughly 10-100x more power-efficient on neuromorphic hardware, perfect for passive wearable devices in an incubator. Focal loss natively corrects our `6.48:1` dataset skew without throwing away rich background data via undersampling.

### 2.6 Comprehensive XAI & Translation Exegesis
**What it is:**
A multi-modal suite designed to "open the black box" using Captum, SHAP, and LIME frameworks.

**Technical Deep Dive:**
*   **Integrated Gradients:** Reintegrates the surrogate path gradients from a 0-baseline (flatline EEG) over 50 interpolation steps to map exact feature attributions.
*   **SHAP & LIME:** Kernel perturbation and local surrogate regressions quantify exactly how much a specific time-slice contributed to pushing the probability toward a seizure alert.
*   **10-20 Electrode Translation Hub:** The array indices (Channels 0-20) are piped into a Python `OrderedDict` and statically mapped back to anatomical syntax (e.g., `Ch7 -> T3: Mid-Temporal, Left, Auditory Cortex`).

**Why it's necessary:** 
In high-stakes pediatric neurology, an opaque "98% Seizure Probability" alert cannot be trusted blindly. If the network triggers an alarm, the XAI suite reverse-engineers the neural firing. If it reports that the prediction heavily relies on "T3 Temporal Left"—a known biological epileptogenic zone—the clinical staff gains profound trust in the algorithmic reasoning, facilitating preemptive administration of neuroprotective drugs.
