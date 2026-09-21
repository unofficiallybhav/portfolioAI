# Project Portfolio — Detailed Descriptions

This document contains detailed, resume-oriented descriptions of six projects based on the uploaded project reports. The descriptions preserve the terminology, methodology, implementation details, and reported metrics from the source material.

---

## 1. DepthForge — CUDA-Accelerated Stereo Depth Estimation

### Overview

**DepthForge** is a complete CUDA-accelerated implementation of the classical **Semi-Global Matching (SGM)** algorithm for stereo depth estimation. The system takes a rectified stereo image pair and produces a dense disparity/depth representation entirely through GPU computation.

The project was designed around the goal of moving the complete stereo-matching pipeline onto the GPU rather than using CPU computation in the critical path. It targets KITTI-resolution stereo images and focuses heavily on CUDA kernel optimization, memory hierarchy, shared-memory usage, and parallelization of the computationally challenging SGM recurrence.

### What was built

The implementation consists of a six-stage GPU pipeline:

1. **Census Transform**
   - Computes a local 9×7 binary descriptor for every pixel.
   - Stores the descriptor in a `uint64` census volume.
   - Uses one CUDA thread per pixel.

2. **Cost Volume Computation**
   - Computes the Hamming distance between corresponding left/right Census descriptors.
   - Uses CUDA's `__popcll` intrinsic for efficient population-count operations.
   - Evaluates disparity candidates across the configured disparity range.

3. **8-Path SGM Aggregation**
   - Implements eight directional paths:
     - Left → Right
     - Right → Left
     - Top → Bottom
     - Bottom → Top
     - Four diagonal directions
   - Uses CUDA shared memory to maintain previous scanline costs.
   - Uses parallel min-reduction to obtain the minimum previous-path cost.
   - Accumulates the aggregated cost volume using `atomicAdd`.

4. **Winner-Takes-All Disparity Selection**
   - Performs an argmin across the disparity dimension to select the lowest-cost disparity for every pixel.

5. **Median Filtering**
   - Applies a 3×3 sorting-network median filter to suppress impulse noise, particularly in textureless regions.

6. **Jet Colormap Visualization**
   - Converts the resulting disparity map into an RGB depth heatmap using a CUDA kernel.

All six stages operate on GPU memory, with each stage passing its output directly to the next stage without CPU intervention.

### CUDA/GPU engineering

A major focus of DepthForge is the optimization of the SGM aggregation stage. The implementation assigns CUDA blocks to scanlines and uses one thread per disparity level within a scanline. Shared memory stores the previous pixel's SGM costs, reducing repeated global-memory accesses.

For KITTI-resolution images of **1242×375 pixels with 256 disparities**, the cost volume and aggregated cost volume require approximately **592 MB of GPU memory** in total:

- Cost volume: approximately 118 MB
- Aggregated cost volume: approximately 474 MB

The implementation also uses empirically tuned SGM penalties of **P1=7** and **P2=86** for the KITTI benchmark.

### Benchmarking and results

DepthForge was evaluated on **all 200 KITTI Stereo 2015 training scenes** using LiDAR disparity ground truth.

Reported aggregate results:

| Metric | Result |
|---|---:|
| MAE | **1.784 px** |
| RMSE | **6.884 px** |
| Bad1.0 | **31.8%** |
| Bad2.0 | **12.1%** |
| Bad3.0 | **7.1%** |
| Bad5.0 | **4.1%** |
| Median error | **0.708 px** |

On a synthetic 640×480 benchmark with 128 disparities, the GPU implementation required **48.064 ms**, compared with **1440.6 ms** for the CPU 4-path baseline, corresponding to approximately **30× speedup**.

On KITTI-resolution images, the reported median processing time was approximately **168 ms** on an **NVIDIA RTX 3050 Laptop GPU**.

### Comparison with OpenCV

DepthForge achieved a **7.1% Bad3.0** rate, which is comparable to the reported OpenCV StereoSGBM range of approximately **6–8%**, while using GPU acceleration.

The report estimates DepthForge to be approximately **2–9× faster** than the cited CPU OpenCV stereo approaches, depending on the baseline.

### Technical focus

**Computer Vision · CUDA · C++ · GPU Parallelism · Shared Memory · Stereo Matching · Semi-Global Matching · KITTI · Performance Optimization**

### Resume-oriented summary

> Built a fully CUDA-accelerated six-stage stereo depth pipeline implementing Census matching, cost-volume computation, 8-path SGM aggregation, disparity selection, filtering, and visualization; achieved **7.1% Bad3.0 error**, **1.78 px MAE**, and approximately **30× GPU speedup** over the CPU baseline on KITTI Stereo 2015.

---

## 2. ForgetField — Adaptive Hopfield Memory with Dynamic Pattern Forgetting

### Overview

**ForgetField** introduces **Adaptive Hopfield Memory with Dynamic Pattern Forgetting (AHM-DPF)**, a neural-network architecture designed to address the limited capacity and catastrophic interference of conventional Hopfield networks during sequential pattern learning.

The central idea is to augment associative memory with **feedback-driven adaptation and selective forgetting**. Instead of treating stored patterns as equally persistent, the proposed architecture uses feedback from recent training performance to dynamically modify its pattern-retention behavior.

### Problem addressed

Classical Hopfield networks have limited storage capacity and can suffer from spurious attractors and interference when patterns are learned sequentially.

The project specifically investigates how a Hopfield-style associative memory can dynamically adapt its retention behavior based on performance feedback, allowing obsolete or less useful information to be forgotten while useful patterns are retained.

### What was built

The AHM-DPF architecture includes:

- A learned hidden representation for binary input patterns.
- A modern Hopfield-style associative memory mechanism.
- Internal state conditioned on previous training performance.
- A feedback signal derived from the previous epoch's training loss.
- Adaptive regularization behavior influenced by the feedback signal.
- A classification projection producing output logits.
- A combined classification and regularization objective.

The feedback signal is defined from the previous epoch's training loss:

`f_t = L_train(t−1)`

The total objective combines binary cross-entropy classification loss with the adaptive regularization loss.

### Experimental variants

Two main AHM-DPF configurations were evaluated:

- **AHM-DPF Fair:** 8-dimensional hidden representation, matching the capacity of the baseline configurations.
- **AHM-DPF Large:** 64-dimensional hidden representation, used to evaluate higher-capacity behavior.

The study also compared against multiple baseline variants:

- Hopfield Base
- Hopfield Adapted
- Hopfield Pooling Base/Adapted
- Hopfield Lookup Base/Adapted

### Dataset and evaluation

The experiments use the **BitPatternSet** benchmark.

The benchmark contains:

- **2,048 bags**
- **16 instances per bag**
- **8-bit binary patterns**
- **1 classification signal per bag**
- 1,536 training bags
- 512 evaluation bags

The experiments were implemented in PyTorch and trained for **250 epochs**, using AdamW with a learning rate of `10^-3`, batch size 32, and gradient clipping with maximum norm 1.0.

### Results

The high-capacity AHM-DPF Large model achieved:

- **99.8% final evaluation accuracy**
- **0.043 final evaluation loss**
- **100% best evaluation accuracy**

The fair-comparison AHM-DPF model achieved:

- **87.3% final evaluation accuracy**
- **0.280 final evaluation loss**
- **88.3% best evaluation accuracy**

For comparison, the standard Hopfield Base model achieved only **56.4% final accuracy**.

The study also found that adaptive variants consistently outperformed their non-adaptive counterparts by approximately **25–33 percentage points**, supporting the conclusion that feedback-driven adaptation—not merely architectural changes—was a major source of the performance improvement.

### Learning dynamics

The fair AHM-DPF model improved from approximately **54% to 87.3% evaluation accuracy** and converged around epoch 100.

The large-capacity model displayed an interesting transient event around epoch 80: evaluation accuracy temporarily dropped from nearly 100% to approximately **75%**, after which the feedback mechanism enabled recovery to **99.8%**.

The report interprets this behavior as evidence of dynamic pattern reorganization/forgetting followed by reconsolidation.

### Technical focus

**PyTorch · Neural Networks · Hopfield Networks · Associative Memory · Adaptive Learning · Feedback Learning · Pattern Recognition · Continual Learning**

### Resume-oriented summary

> Proposed **AHM-DPF**, a feedback-driven Hopfield architecture with selective pattern forgetting to mitigate catastrophic interference; achieved **99.8% accuracy** in the 64-D setting and **87.3%** under fair 8-D comparison, outperforming non-adaptive variants by **25–33 percentage points**.

---

## 3. Network Intrusion Anomaly Detection Using Machine Learning

### Overview

This project develops a **dual-paradigm network intrusion detection framework** designed to detect both known and previously unseen network threats using the **UNSW-NB15** benchmark.

The system combines:

1. **Supervised learning** for known, previously catalogued attack patterns.
2. **Unsupervised anomaly detection** for zero-day threats without requiring labeled attack examples during training.

This creates a defense-in-depth approach in which known attacks can be classified with high confidence while anomalous traffic that does not match known patterns can be separately flagged.

### Dataset and preprocessing

The UNSW-NB15 dataset used in the study contains approximately **2.54 million records** with **49 features** across four CSV files.

The project performed extensive exploratory data analysis and preprocessing, including:

- Missing-value analysis.
- Feature correlation analysis.
- Data leakage prevention.
- Stratified train/test splitting.
- Class imbalance analysis.
- Feature selection.
- SMOTE-based oversampling.
- PCA dimensionality reduction.

The original training distribution contained approximately:

- **95.16% normal traffic**
- **4.84% attack traffic**
- **19.67:1 class imbalance ratio**

To address this imbalance, SMOTE was applied with a 0.3 sampling strategy, reducing the imbalance ratio to approximately **3.33:1**.

For unsupervised anomaly detection, PCA reduced the feature representation to **25 components while retaining 92.87% of the variance**.

### Supervised learning

Four supervised models were implemented and compared:

- Random Forest
- LightGBM
- XGBoost
- Logistic Regression

The models were trained to distinguish normal traffic from known attack traffic and to capture the nonlinear patterns associated with network attacks.

### Supervised results

| Model | Accuracy | F1 |
|---|---:|---:|
| Random Forest | **99.47%** | **99.47%** |
| LightGBM | 99.41% | 99.41% |
| XGBoost | 99.34% | 99.35% |
| Logistic Regression | 98.55% | 98.64% |

Random Forest produced the strongest overall result.

Its class-specific performance included:

- Normal precision: **99.79%**
- Normal recall: **99.65%**
- Normal F1: **99.72%**
- Attack precision: **93.30%**
- Attack recall: **95.96%**
- Attack F1: **94.61%**

### Zero-day anomaly detection

Two unsupervised methods were evaluated:

- Isolation Forest
- One-Class SVM

The purpose was to identify anomalous network behavior without using labeled attack examples.

One-Class SVM achieved:

- **0.858 ROC-AUC**
- **100% recall of test attacks**
- **28.4% false-positive rate**

Isolation Forest achieved an AUC of **0.585**, making One-Class SVM substantially stronger for this particular zero-day detection setup.

### Overall system

The resulting architecture can be viewed as a layered security system:

**Network traffic → supervised known-threat classifier → anomaly detector for traffic requiring additional scrutiny**

The supervised layer provides high-confidence detection for known attacks, while the unsupervised layer provides additional coverage for potentially novel threats.

### Technical focus

**Python · Scikit-learn · Random Forest · LightGBM · XGBoost · Logistic Regression · Isolation Forest · One-Class SVM · SMOTE · PCA · Network Security · Anomaly Detection**

### Resume-oriented summary

> Built a dual-stage ML-based NIDS for known and zero-day threats on UNSW-NB15; evaluated **6 classification/anomaly models**, addressed a **19.67:1 class imbalance** using SMOTE, and achieved **99.47% accuracy** with Random Forest and **0.858 ROC-AUC** with One-Class SVM for zero-day detection.

---

## 4. Adversarial Robustness of Deep Learning-Based Network Intrusion Detection Systems

### Overview

This project studies the adversarial robustness of a **deep neural network-based Network Intrusion Detection System (NIDS)** trained on UNSW-NB15.

Instead of evaluating the model only on clean network traffic, the project investigates whether an attacker can manipulate network-flow features by small perturbations to cause the NIDS to misclassify malicious traffic.

The work includes four major stages:

1. Baseline DNN training.
2. Systematic adversarial attack evaluation.
3. Adversarial training as a defense.
4. Explainability and representation analysis.

### Baseline DNN

The project implements a fully connected neural network called **IDS-DNN**.

Architecture:

- Input: 42 features
- Fully connected layer: 42 → 256, ReLU, dropout 0.30
- Fully connected layer: 256 → 128, ReLU, dropout 0.30
- Fully connected layer: 128 → 64, ReLU, dropout 0.20
- Fully connected layer: 64 → 32, ReLU
- Output: 32 → 10 classes

The preprocessing pipeline includes:

- Categorical feature encoding.
- Data leakage prevention.
- Min-max normalization to `[0,1]`.
- Targeted SMOTE for highly underrepresented classes.
- Class-weighted loss.

Targeted SMOTE specifically increased the training examples for:

- Analysis: **1,700 → 3,000**
- Backdoor: **1,484 → 3,000**
- Worms: **111 → 3,000**

### Adversarial attacks

The model was evaluated using two gradient-based attacks:

- **FGSM — Fast Gradient Sign Method**
- **PGD — Projected Gradient Descent**

The attacks were evaluated at six perturbation budgets:

`ε ∈ {0.00, 0.01, 0.05, 0.10, 0.20, 0.30}`

PGD used 40 attack iterations during evaluation.

### Baseline robustness results

On clean data, the standard model achieved:

- **Macro F1: 0.406**
- **Weighted F1: 0.709**

However, under PGD with **ε=0.10**, macro F1 dropped to:

- **0.074**

This represented an **81.7% reduction in macro F1**.

The project also identified an important evaluation artifact: at large perturbation budgets, the apparent robustness of the standard model was partly caused by the model predicting traffic as benign almost universally. This demonstrated why accuracy or weighted metrics alone can produce misleading conclusions for heavily imbalanced intrusion-detection datasets.

### Adversarial training defense

A robust model was trained using **ε-randomized PGD adversarial training**.

Instead of training at a single fixed perturbation budget, ε was randomly sampled from:

`[0.01, 0.15]`

The defense also used a weighted combination of clean and adversarial losses.

This was intended to avoid the narrow robustness band produced by fixed-ε adversarial training and provide broader robustness across different attack strengths.

### Robustness improvements

At **ε=0.10 PGD**:

- Standard macro F1: **0.074**
- Robust macro F1: **0.240**
- Macro-F1 loss recovery: **223.8%**

Weighted F1 improved from:

- **0.257 → 0.553**

The ε-randomized strategy also eliminated the previously observed robustness reversal at ε=0.20 associated with fixed-budget adversarial training.

### Explainability and representation analysis

The project used **SHAP DeepExplainer** to compare feature attribution between standard and adversarially trained models.

The analysis found that:

- The standard model concentrated more heavily on rate and TTL-related features that were vulnerable to gradient-based attacks.
- The robust model distributed attribution more broadly across volume and timing features.

The project also used **t-SNE** on 32-dimensional penultimate-layer representations. The robust model produced tighter and better-separated clusters for several traffic classes, providing representation-level evidence supporting the observed robustness improvements.

### Important limitations identified

The study also explicitly analyzed the robustness/accuracy trade-off.

For example, the Reconnaissance class experienced a clean F1 reduction from **0.731 → 0.347**, while its adversarial F1 at ε=0.10 improved from **0.008 → 0.224**.

The report also notes that large perturbation budgets can be physically unrealistic for constrained network features, making the operationally meaningful evaluation regime approximately **ε ≤ 0.05**.

### Technical focus

**PyTorch · Deep Learning · Network Security · Adversarial ML · FGSM · PGD · Adversarial Training · SHAP · t-SNE · SMOTE · UNSW-NB15 · IBM ART**

### Resume-oriented summary

> Developed and stress-tested a DNN-based NIDS against **FGSM/PGD attacks**, revealing an **81.7% macro-F1 degradation** at ε=0.10; designed ε-randomized PGD adversarial training that improved macro F1 from **0.074 to 0.240** and weighted F1 from **0.257 to 0.553**, with SHAP/t-SNE analysis of robustness mechanisms.

---

## 5. ARC-Nodule — Acquisition-Robust Consistency for Lung Nodule Detection *(active / in progress)*

### Overview

**ARC-Nodule (Acquisition-Robust Consistency)** is an ongoing research pipeline building a lung-nodule detection and malignancy-classification system engineered for robustness to CT acquisition protocol — specifically slice thickness, dose, and reconstruction-kernel variation — without requiring any labeled data from the target protocol. It is built as a fully standalone package, independent of a companion nodule-detection pipeline developed in the same project. The work is mid-build: two of its three core mechanisms are validated with multi-seed replication, and the third has been tested twice without yet clearing its own pre-registered bar, with the properly-powered retest identified as the next step rather than already completed.

### Problem addressed

Nodule detectors in the published literature are trained and benchmarked almost exclusively on thin-slice CT (≤2.5mm reconstruction); the field's dominant benchmark (LUNA16) structurally excludes anything coarser, so the standard evaluation most published work relies on cannot even measure what happens on thicker slices. In practice, a large share of real-world and legacy-scanner acquisitions are reconstructed at 3–10mm, and a cited 2026 stress-test of a standard detector found sensitivity collapsing **45.2% → 26.2%** (a 42% relative drop) at 5mm — a bigger hit than heavy dose reduction causes. ARC-Nodule targets this specific failure mode (acquisition-protocol shift), distinct from cross-dataset/cross-hospital domain shift, on the premise that the fix must act on the network's internal representation rather than just its label mapping.

### What was built

The pipeline combines three mechanisms:

1. **Physics-Simulated Acquisition Augmentation (PSAA)** — a deterministic transform generating a labeled "hard domain" for free: z-axis point-spread-function blur followed by resampling to a randomized target slice thickness, plus calibrated dose-noise injection, requiring no new annotations. Validated with 37 unit tests and an independent image-domain realism check (no detector involved) confirming the simulated degradation matches real acquired thickness to within ~3% across the z-frequency spectrum.
2. **Anisotropy-aware Swin Transformer encoder** — a 3D Swin-style backbone conditioned on the actual voxel spacing of its input via FiLM-style injection per attention block, with independently sized attention windows along the z-axis versus in-plane, so the network is told how anisotropic a given volume is rather than assuming isotropy.
3. **Cross-Acquisition Consistency Distillation (CACD)** — the intended key differentiator: detection heatmaps/embeddings and malignancy logits computed from a degraded "twin" of a scan are pulled toward the outputs from the native-resolution pass of the *same* scan via a cosine/KL consistency loss, while both passes remain supervised on the true label. Implemented with a per-term magnitude-balancing mechanism (`TermBalancer`) after an early version was found to have its detection-side loss term unintentionally diluted to under 5% of the total signal.

Every architectural claim is tested with a matched ablation-control arm trained under identical architecture, seed, and schedule, and a from-scratch, multi-seed (3–5 seed) training protocol is used throughout rather than relying on single-run results.

### Dataset and evaluation

Training and the primary synthetic evaluation use **LUNA16** subsets 0–2 (~221 scans). Because LUNA16 itself excludes scans coarser than 2.5mm, thick-slice conditions are simulated via PSAA across a 1.25–10mm sweep, with sensitivity/CPM (competition performance metric) reported against thickness. A held-out **LIDC-IDRI** split restricted to real scans ≥3mm slice thickness (~121 of 1018 scans, with an existing extraction pipeline) has been scoped as a genuine real-acquisition validation arm but has not yet been evaluated. Detection is scored with a whole-scan sliding-window harness — built after an earlier version was found to silently score cropped, nodule-centered patches instead of full scans, invalidating false-positive-rate-based metrics. A formal statistical power analysis of the evaluation regime was also conducted, reconstructing the exact sample sizes and per-metric detectable-effect thresholds from prior result files, to determine in advance how many seeds a given claim requires before being treated as evidence.

### Results

- **Validated, replicated finding:** combining PSAA augmentation with the spacing-conditioned encoder measurably flattens the thickness-sensitivity curve, replicated cleanly across 3 independent seeds. Mean CPM drop at 5mm shrinks from **33.1% (untreated baseline) to 21.6%** (treated), and at 10mm from **70.2% to 59.5%**, improving in **9 of 9** seed-by-thickness cells at ≥3mm, at **no measurable cost to native (thin-slice) detection**, with the gain concentrated on small nodules — the population the grounding literature identifies as most vulnerable.
- **Unresolved finding:** the consistency-distillation module (CACD), the pipeline's designated key differentiator, has not yet beaten a matched no-consistency control on its pre-registered criterion in either of two test rounds. A first round found its detection-loss term was unintentionally near-zero-weighted and coincided with a small performance cost at severe thicknesses; a rebalanced second round reversed that cost into a consistent (but not yet properly powered or pre-registered) gain in the same region. The identified next step — a properly powered, pre-registered retest — is scoped but not yet executed.

### Technical focus

**PyTorch · 3D Swin Transformer · Medical Imaging · CT Image Processing · Physics-Based Data Augmentation · Domain Generalization · Consistency Distillation · LUNA16 · LIDC-IDRI · GPU Training (Kaggle) · Experimental Design · Statistical Power Analysis · Ablation Studies**

### Resume-oriented summary

> Building **ARC-Nodule**, an acquisition-robust lung-nodule detection/classification pipeline combining physics-simulated CT augmentation, an anisotropy-conditioned 3D Swin Transformer encoder, and a cross-acquisition consistency-distillation loss; validated a **9/9-seed-replicated** reduction in thickness-induced detection collapse (**5mm CPM drop 33.1%→21.6%**) at no cost to native detection, and applied a custom statistical-power framework to rigorously separate real effects from evaluation noise across a multi-seed, ablation-controlled experimental protocol.

---

## 6. Lepton — Rain-to-Traffic Impact Prediction and Analysis *(production system)*

### Overview

**Lepton** is a production rain → road-traffic analytics platform (part of the TraffiCure system) that quantifies and predicts how rainfall degrades road traffic at **individual-road granularity** across ten Indian cities. It spans two complementary halves:

- a **retrospective half** that measures, with causal-inference controls, which roads actually slowed during rain, flood alerts, and rain-triggered congestion spillover; and
- a **predictive half** that serves a Pune week-ahead **forecast** and an hours-ahead, forecast-free **nowcast** of per-road rain-induced delay.

Unlike a single-model project, Lepton is an end-to-end data system: weather ingest from four independent reanalysis/forecast sources, a cross-database join layer over a PostgreSQL/PostGIS traffic warehouse split across three physical boxes, versioned model families (forecast v1–v2, nowcast v1–v4), persisted prediction tables, and three stdlib-only operational dashboards. Every product is documented in a methodology document kept 1:1 with its script.

### Problem addressed

Naive "rain slows traffic" analytics fail on real municipal data for three reasons the project had to solve explicitly:

1. **Baseline contamination** — raw slowdown during a rainy evening is mostly rush hour, not rain. The system works exclusively in *baseline-relative* space: a road's delay is always measured against **its own** median dry-hour delay for the same weekday × hour, never against a city-wide average.
2. **Spatio-temporal misalignment** — the hourly traffic spine and the IFS/Google weather tables are stamped at `HH:30` UTC (a clean IST clock hour) while ERA5-Land and CDS-ERA5 are stamped at `HH:00` UTC. A naive timestamp-equality join across the two grids returns **zero rows**; all joins are bucketed to the IST clock hour.
3. **Rain-source disagreement** — the available precipitation products disagree materially. Validation against IMD gauge data found Open-Meteo/ECMWF-IFS under-catching real rainfall by roughly **2.5–3×** (~0.46× catch ratio), while ERA5/ERA5-Land catch ~**0.81×**; ERA5-Land's finer ~9 km grid also resolves 10 distinct cell values over Pune versus 3 for coarse ERA5. This finding drove the switch of the default historical rain source to ERA5-Land.

### What was built

**Weather ingest layer.** Ingest jobs for ERA5 and ERA5-Land (Copernicus CDS), NASA IMERG (Earthdata), Google Weather live QPF, and Open-Meteo, each normalising to a common hourly per-H3-cell precipitation schema, with publication-latency handling (CDS reanalysis lags by days to months, so each run reports the actual `max(ts_utc)` retrieved rather than assuming the requested window was filled). Roads are mapped to rain cells through a best-overlap H3 resolution-7 mapping.

**Retrospective analytics (multi-city).**

- **Rain vulnerability builder** — the central product. For every road-hour it computes trailing rain totals, labels wet (`r3h ≥ 1.0 mm`) versus dry (`r3h < 0.2 mm`) with the ambiguous 0.2–1.0 mm band excluded, subtracts the matched (road, weekday, hour) dry baseline, and ranks roads by **total excess delay seconds**. It groups wet days into rain events (wet days ≤ 2 days apart are one event), flags roads whose excess is dominated by a single storm, and gates evidence with a **leave-one-event-out jackknife** — requiring ≥ 10 wet and ≥ 20 dry observations, ≥ 5 wet days, ≥ 3 events, and a worst-case-event median residual ≥ 5 s — emitting tiers `proven` / `storm_dependent` / `not_robust` / `data_gate_excluded` / `below_choke`. A `percent` gate mode (Mann-Whitney + matched delay-%) handles cities of short, fast roads where absolute seconds under-detect.
- **Rain network propagation** — builds a road adjacency graph (endpoints within 20 m, PostGIS, unioned with shared-junction adjacency), then measures whether a congested wet "source" road's 1- and 2-hop neighbours show delay uplift at a lag, producing source-export/receiver-import scores, hop decay, spreader junctions, and "surprise receivers".
- **Flood impact** — isolates delay inside confirmed flood-alert windows, scoring only flood-active hours against the same road/weekday/hour non-flood baseline, and classifies each flood as `rain_driven` / `mixed` / `non_rain` / `unknown` (missing rain coverage is explicitly *unknown*, never silently dry).
- **PDF reporting** — a read-only "Top-N Rain Bottlenecks" report generator that reverse-geocodes placeholder road identifiers via Google.

**Predictive models (Pune).**

- **Forecast v2 (production)** — a **weighted log-linear regression** on the multiplicative response `R = actual_speed / typical_speed`, weighted by traffic sample count: `log(R) ~ road class + peak + rain band + rain decay + lag + onset + rain-front edge + flood flag + interactions`. Rain features include hourly intensity bands, trailing rain, onset, a spatial rain-front indicator, and exponentially decayed rain memory as a standing-water proxy (a dry-but-recently-wet hour stays in a four-hour lag window). Inversion from log space uses a **Duan smearing** correction plus a 3 km/h speed floor. Severity is deliberately **not** the regression's confidence interval: separate **conformalized quantile regressions (CQR)** fit on wet and lag rows produce p80/p90 severity bounds, with one-sided conformal offsets calibrated per rain band on a held-out fold of the rainiest days. It replaced a v1 LightGBM quantile model in production after beating it on a rainiest-day holdout.
- **Nowcast v2–v4 (forecast-free)** — predict from *observed* live traffic plus known-future rain only, never calling the week-ahead model. v3 adds per-road ~2 km **spateGAN-ERA5** downscaled rain fields; v4 operates on a 15-minute traffic panel with 10-minute rain re-binned to 15 minutes and an hourly typical profile broadcast to sub-hour slots. The served product is a **v4 + v3 blend** (v4 owns +15/30/45/60 min, v3 owns +2 h/+3 h), written as long-format quantile rows to a shared prediction table.
- **Adaptive serve gating** — rather than a fixed wet threshold, the live nowcast drops a cell unless forecast rain ≥ **0.5 × that city-day's maximum hourly forecast precipitation**, falling back to a fixed 1.0 mm when the day has no forecast rain — so light-rain days are still served proportionally instead of being blanked.

**Serving layer.** Three dependency-free `http.server` dashboards (Pune forecast/nowcast, multi-city historical, event impact) serving HTML plus JSON APIs, with the forecast dashboard auto-reissuing the nowcast every 15 minutes by orchestrating the v4 and v3 serve jobs as subprocesses, normalising their horizon scales, and de-duplicating the overlapping +1 h row.

### Data engineering and system design

The traffic warehouse is split across **three PostgreSQL/PostGIS instances** (staging, dev, production) with an awkward but real topology: rain and all computed output tables live only on staging, while a given city's road geometry and hourly traffic may live on dev or production — Delhi's full pre-monsoon traffic exists **only** on the dev box. Several builders are therefore **cross-database**: they open two or three independent engines, pull each side separately, and perform the join **in pandas** rather than in SQL. Every analysis connection is pinned `default_transaction_read_only=on`, and dev is treated as strictly read-only.

The traffic spine is an hourly per-road metrics table whose delay column is *already* baseline-relative, a non-obvious property that invalidates naive re-baselining; new derived tables are namespaced into a dedicated `raw` schema. Training panels reach multi-million-row scale — the spateGAN-sourced forecast retrain used **2,013,497 road-hour rows**, with severity quantile regressions fit on 92,858 wet/lag rows.

### Validation and results

**Retrospective builder — reproduction test.** The rain-vulnerability builder was reconstructed from its outputs and validated **road-for-road** against the live production bottleneck report over Pune, 2026-03-03 → 2026-06-02:

| Metric | Rebuilt builder | Live report |
|---|---|---|
| Qualifying roads | 151 | **151** (exact) |
| Data-gate excluded | 4 | **4** (exact) |
| Storm-dependent tier | 33 | **33** — same 33 roads |
| #1 proven road | `397c7792…` | **same road** |
| #1 road total excess | 3958.5 s | 3971.0 s (**0.3% delta**) |

Total-excess differences across the proven set had a median of **0.5 s** (max 12.5 s), traceable to a ~15-hour difference in the dry-hour set. The `proven` boundary remains approximate for roads within a few seconds of the 5 s jackknife floor, because the reference implementation's robustness step is not observable — a limitation the project documents rather than papers over.

**Causal-inference validation of the observational products.** Rather than accepting correlational output, the propagation and flood products were stress-tested with:

- a **spatial permutation placebo** — 300 draws of random *non-adjacent* control roads over the *same hours*, so the null already absorbs the city-wide "everyone is wet and jammed" effect;
- **onset event studies** checking that the delay signal is time-locked to rain/flood onset;
- **independent corroboration** against a congestion-alert system that never touches the delay metric, yielding a **3.7× alert lift** during propagation windows and **4.4×** during flood windows — confirming the measured delay is real congestion rather than an aggregation artifact.

All 3/3 verification tests passed, establishing the signal as real, time-locked, and adjacency-specific. The project also reports its **negative** result honestly: the neighbour-dry subset returned **zero samples** (adjacent roads share the same ~9 km rain cell, so source-wet always implies neighbour-wet), and peak propagation lag was mostly 0 h — meaning causal direction (A→B propagation versus common rain trigger) is **empirically unprovable in the available pre-monsoon window**. Structural findings that *are* trustworthy include a network amplification of **5.92** and a spillover share of **0.83**: the 73 robust source roads account for only ~17% of total wet-hour excess delay, while **329 "surprise receivers"** absorb spillover despite never appearing on the vulnerability ranking.

**Forecast model performance.** Evaluation is deliberately restricted to the rows the model exists for, because a test window that is ~96% dry makes overall metrics meaningless (the naive `R = 1` predictor is nearly unbeatable on dry hours):

| Configuration | Wet-row skill vs naive | Notes |
|---|---:|---|
| v2 WLS, ERA5 rain (production) | **+14.2%** | MAPE 7.67% vs naive 8.94%, n = 17k–29k |
| v2 WLS, spateGAN per-road 2 km rain | **+19.2%** | n = 17,048; lag rows +3.6%, overall +1.6% |
| v1 LightGBM (retired) | ≈ 0% | q50 ≈ naive; intervals miscalibrated both directions |

Raw WLS observation intervals were found to over-cover (~100% against an 80% target) because they assume homoscedasticity; this drove the replacement of interval-based severity with conformalized quantile regression, which tightened flood-road p90 severity breach from **0.137 to 0.121** before conformal calibration. A rain-front-edge flag lifted skill on previously unserved boundary hours from +1.5% to **+2.8%**. Several candidate features were **rejected** on holdout — a weekend term cut wet-row skill from +14.2% to +11.8% and was dropped rather than retained for apparent complexity.

**Nowcast performance.** Per-horizon backtesting against realised traffic showed the model's shrunk autoregressive term slightly *underperforming pure persistence* at the shortest leads. Instead of hiding this, v4 implements a per-horizon **persistence blend** whose weights are grid-searched to minimise holdout MAE and then re-calibrated against served-versus-actual rows — self-tuning to heavy persistence at short lead. Calibrated weights of `[0.90, 0.70, 0.25, 0.15]` cut **+15 min MAE by ~26%** and **+30 min by ~15%**. Conformal widening lifted quantile coverage from a raw ~0.62–0.80 to **~0.90**, with the documented caveat that this calibration is in-sample and therefore optimistic.

**Data-source validation.** Independent comparison of four precipitation products against IMD gauge observations established the catch ratios cited above and directly determined the production rain source, rather than defaulting to whichever feed was easiest to obtain.

### Engineering judgment and honest limitations

The project's documentation is unusually explicit about what its data cannot support — the binding constraint throughout is that the pre-monsoon window contains only about **three independent storms** under the validated wet-event definition. That single fact killed several plausible extensions: a temporal-convolutional-network nowcast (v1) whose training entry point now *refuses to run*, and neighbour, lag-forward, and flood-prone feature experiments that all returned ~0 gain and were dropped. The spateGAN integration is written as an explicit **seam** with an honest null: when the real pre-generated GAN field is absent, the surrogate path is mass-conserving identity, so the spatial nowcast cleanly degrades to its non-spatial predecessor rather than fabricating apparent skill.

### Technical focus

**Python · PostgreSQL / PostGIS · pandas · NumPy · SQLAlchemy · psycopg · statsmodels (WLS, quantile regression) · Conformal Prediction (CQR) · LightGBM · PyTorch · H3 Geospatial Indexing · ERA5 / ERA5-Land / IMERG / Copernicus CDS · netCDF4 · Google Weather API · Cross-Database ETL · Causal Inference (matched baselines, difference-in-differences, permutation placebos, event studies) · Time-Series Forecasting & Nowcasting · Matplotlib / PDF Reporting · Production Dashboards**

### Resume-oriented summary

> Built **Lepton**, a production rain → road-traffic impact platform serving per-road delay analytics and forecasts across **10 cities** from a cross-database PostGIS traffic warehouse; reproduced a live production bottleneck ranking **road-for-road** (151/151 qualifying roads, 33/33 storm-dependent tier, 0.3% delta on top-road excess delay), validated observational findings with permutation placebos and independent alert corroboration (**3.7–4.4× alert lift**, 3/3 verification tests), and shipped a WLS + conformalized-quantile-regression rain forecast reaching **+19.2% wet-hour skill** over naive plus a self-tuning 15-minute nowcast blend that cut short-horizon MAE **~26%**.

---

# Portfolio-Level Summary

These projects collectively demonstrate experience across several complementary areas:

| Area | Demonstrated Work |
|---|---|
| **GPU / Systems** | CUDA kernels, shared memory, parallel reduction, GPU memory optimization |
| **Computer Vision** | Stereo matching, disparity estimation, KITTI benchmarking |
| **Deep Learning** | DNN classifiers, Hopfield networks, representation learning |
| **Machine Learning** | Random Forest, LightGBM, XGBoost, Logistic Regression |
| **Unsupervised ML** | One-Class SVM, Isolation Forest, PCA |
| **Adversarial ML** | FGSM, PGD, adversarial training |
| **Explainable AI** | SHAP DeepExplainer, t-SNE |
| **Cybersecurity** | Network intrusion detection, zero-day anomaly detection |
| **Research** | Novel AHM-DPF architecture, ablation studies, robustness analysis |
| **Performance Engineering** | CUDA optimization and 30× CPU/GPU benchmark improvement |
| **Medical Imaging** | 3D CT processing, domain generalization, physics-based augmentation |
| **Data Engineering** | PostgreSQL/PostGIS, cross-database ETL, geospatial (H3) joins, multi-source weather ingest |
| **Statistical Modelling** | Weighted least squares, quantile regression, conformal prediction, time-series nowcasting |
| **Causal Inference** | Matched baselines, difference-in-differences, permutation placebos, event studies |
| **Experimental Methodology** | Multi-seed replication, ablation-controlled A/B design, statistical power analysis |

## Strongest metrics across the portfolio

- **30×** GPU speedup — DepthForge
- **7.1%** Bad3.0 stereo error — DepthForge
- **99.8%** classification accuracy — AHM-DPF Large
- **25–33 pp** improvement over non-adaptive Hopfield variants
- **99.47%** known-threat detection accuracy — Random Forest NIDS
- **0.858 ROC-AUC** zero-day detection — One-Class SVM
- **81.7%** adversarial macro-F1 degradation exposed — DNN NIDS
- **223.8%** macro-F1 loss recovery through adversarial training
- **0.074 → 0.240** macro F1 under ε=0.10 PGD
- **0.257 → 0.553** weighted F1 under ε=0.10 PGD
- **33.1% → 21.6%** thickness-induced CPM drop at 5mm, replicated across 3 seeds — ARC-Nodule
- **151/151** roads reproduced against a live production ranking (0.3% top-road delta) — Lepton
- **+19.2%** wet-hour forecast skill over naive; **~26%** short-horizon nowcast MAE reduction — Lepton
- **3.7–4.4×** independent alert-lift corroboration of measured rain/flood congestion — Lepton

## 7. PortfolioAI — Resume-Based Portfolio Chatbot and Job-Match Web App

  ### Overview

  PortfolioAI is a full-stack LLM-powered portfolio assistant that recruiters and
  hiring managers can chat with to learn about the candidate’s technical background.
  Every answer is grounded in two source documents: the resume and a detailed project
  write-up file. A recruiter can also paste or attach a job description, and the
  assistant scores the candidate’s fit out of 100 and writes a cover-letter style
  verdict.

  The project went through two versions:

  - **Version 1:** a hand-written ReAct-style agent loop with tools for job
    description extraction and skill matching, served through FastAPI and deployed
    publicly.
  - **Version 2 (current):** a rebuild as a LangGraph agent graph with native tool
    calling, typed document schemas, per-tool retry limits, file uploads, and
    token-by-token streaming.

  The project combines:

  - agent orchestration with LangGraph
  - LLM tool calling with conditional routing
  - document parsing into schema-constrained JSON
  - grounded question answering over a structured candidate profile
  - job-description parsing, candidate-role scoring and cover-letter generation
  - a FastAPI backend with Server-Sent Events streaming
  - a custom HTML/CSS/JavaScript frontend

  ### What was built

  #### LangGraph agent graph

  The agent is a LangGraph state graph with four main nodes: one LLM node and three
  tool nodes (build_profile, parse_documents and match_profile).

  - The entry point is conditional: build_profile runs first only when the resume or
    the project write-up file has changed, detected by a SHA-256 content hash of each
    file. Otherwise the graph goes straight to the LLM node.
  - The LLM node decides whether to answer directly or call a tool, and conditional
    edges route each tool call to the matching tool node.
  - Tools are called one at a time, and each tool node returns its result to the LLM
    node as a tool message.
  - Conversation state is persisted per chat session with LangGraph’s in-memory
    checkpointer, keyed by session ID.

  The graph state holds:

  - the chat messages
  - the grounded candidate profile and its version
  - the latest uploaded document
  - the last successfully parsed job description
  - the last match result
  - a retry counter for each tool
  - the last tool error

  #### Tools

  - **build_profile** parses the resume and the project write-up file and merges them
    into a single structured profile, stored as profile.json. The project file is the
    authoritative source for project detail, and the resume fills in anything it
    lacks. Long project files are parsed one section at a time, and an LLM step links
    resume project titles to their detailed write-ups when the names differ.
  - **parse_documents** accepts .pdf, .docx and .md files, or pasted text, and parses
    them into JSON. Each document type has its own Pydantic schema: job descriptions,
    resumes and project write-ups. Job descriptions are validated, so a posting with no
    role title or no skills, requirements or qualifications is reported as unclear.
  - **match_profile** compares the candidate profile with the last parsed job
    description. It returns a score from 0 to 100, a hiring recommendation, the matched
    and missing skills, and a first-person cover-letter style verdict. The verdict
    talks about technologies used, systems designed and how they fit the role and
    company. Code checks reject any verdict that quotes metrics from the resume or
    project files, or that contains placeholders.

  #### Failure handling and retries

  - Each tool allows at most two retries after its first failure.
  - Transient failures, such as model output that does not match the schema or a
    verdict that breaks the rules, loop straight back into the same tool node with the
    failure reason as feedback.
  - Unclear inputs, such as a vague job description, go back to the LLM, which tells
    the user exactly what is missing and how many attempts remain.
  - Once a tool’s retries are exhausted, it is removed from the LLM’s tool list for
    that chat, and the assistant explains that it cannot process the input and points
    the user to a new chat or to email.
  - A successful call resets the tool’s retry counter.

  #### Grounding and safety

  - The LLM answers only from the structured profile, and says so when information is
    not available instead of guessing.
  - In-progress projects are described as ongoing work.
  - Off-topic requests are declined.
  - Uploaded documents and pasted text are treated as untrusted data, and instructions
    inside them are ignored.

  #### FastAPI backend

  - Multipart chat endpoint that accepts a message and an optional file upload.
  - Streams the LLM’s final answer token by token over Server-Sent Events, along with
    status events such as “Parsing the job description”.
  - Session management with per-session locks, idle-session expiry and a cap on
    active sessions.
  - Upload validation for file type and size.
  - The profile is built or loaded at startup, so the server refuses to start without
    a knowledge base.

  #### HTML frontend

  - A single-page chat interface that renders streamed answers as they arrive.
  - File attachment by button or drag and drop for .pdf, .docx and .md job
    descriptions.
  - Live status while tools run.
  - A verdict card with the fit score, recommendation, and matched and missing skill
    chips.

  #### Terminal client

  A command-line chat client streams answers in the terminal and supports attaching
  files and inspecting the agent state.

  ### Technical design

  A notable engineering choice is how the agent is kept honest and cheap to run:

  - The profile is rebuilt only when a source document’s content hash changes, so
    restarts and ordinary chats never re-parse documents.
  - Tool failures are classified as transient, unclear or precondition errors, so the
    graph can retry automatically, ask the user for better input, or correct the tool
    order without wasting attempts.
  - The Groq SDK is called directly inside the LangGraph nodes, which gives full
    control over streaming and keeps the model’s reasoning out of the user-facing
    answer.

  ### Technical focus

  Python · LangGraph · Groq API · OpenAI GPT-OSS · FastAPI · Server-Sent Events ·
  Pydantic · PDF Parsing · DOCX Parsing · Markdown Parsing · HTML/CSS/JavaScript ·
  LLM Tool Calling · Agent State Management · Structured JSON Output · Retry and
  Failure Handling · Grounded Question Answering · Job Matching

  ### Status

  Version 2 is built and runs locally. Deploying it publicly to replace version 1 is
  the next step.

  ### Resume-oriented summary

  > Built PortfolioAI, a full-stack AI portfolio chatbot that answers recruiter
  > questions grounded in a resume and project write-ups; rebuilt it as a LangGraph
  > agent graph with an LLM node and three tool nodes (profile building, typed
  > document parsing for PDF/DOCX/Markdown, and candidate-role matching with a
  > cover-letter style verdict), conditional routing, per-tool retry limits, hash-based
  > profile caching and token-by-token streaming over a FastAPI backend.

## 8. Video Transcript RAG — Lecture and Media Question Answering System

  ### Overview

  Video Transcript RAG is a retrieval-augmented question-answering pipeline for video
  or lecture content. It converts media into searchable transcript chunks, stores
  them in a vector database, and answers user questions using only retrieved
  transcript context.

  The system is designed for:

  - ingesting long-form video content
  - transcribing the audio
  - indexing transcript chunks semantically
  - retrieving relevant passages by query
  - answering with timestamps and grounded context

  ### What was built

  The pipeline has several stages:

  1. Media-to-audio conversion
      - Converts uploaded video files into .flac audio using FFmpeg.
      - Normalizes audio to 16 kHz mono for transcription.

  2. Large-file handling
      - Splits oversized audio files into smaller segments so they can be processed
        within upload limits.

      - Preserves time offsets so transcript timestamps remain accurate after
        splitting.

  3. Speech-to-text transcription
      - Uses Groq Whisper (whisper-large-v3-turbo) to generate verbose transcript
        segments.

      - Caches transcript JSON locally so repeated runs do not retranscribe the same
        file.

  4. Chunking for retrieval
      - Groups transcript segments into overlapping text chunks.
      - Stores:
          - chunk text
          - start time
          - end time
          - source file name
          - chunk index

  5. Embedding generation
      - Uses sentence-transformers (all-MiniLM-L6-v2) to create dense vector
        embeddings for each chunk.

  6. Vector database indexing
      - Uploads embeddings and metadata into Qdrant.
      - Creates a payload index on the source field for source-based filtering.
      - Uses deterministic UUIDs for stable point IDs.

  7. Semantic search and Q&A
      - Converts the user query into an embedding.
      - Retrieves top-k transcript chunks from Qdrant.
      - Passes the retrieved context to the Groq LLM for grounded answering.
      - Instructs the model to answer only from the retrieved transcript context and
        to include timestamps.

  ### Technical design
  This project is a practical RAG system with a clear separation of concerns:

  - ingestion
  - transcription
  - vector search
  - grounded generation

  It also includes engineering details that matter in production-style pipelines:

  ### Resume-oriented summary

  > Built a video transcript RAG system that converts media into timestamped
  > transcript chunks, embeds them with sentence transformers, indexes them in
  > Qdrant, and answers user questions with Groq LLM responses grounded only in
  > retrieved transcript context.
---

## Suggested resume positioning

If these projects are being used for a software/ML systems resume, the strongest ordering is:

1. **DepthForge** — strongest systems/GPU engineering signal
2. **ForgetField / AHM-DPF** — strongest novel ML/research signal
3. **Adversarial NIDS** — strongest security/deep-learning signal
4. **Network Intrusion Detection** — strongest breadth of classical ML techniques
5. **Lepton** — strongest production-systems, data-engineering, and applied-statistics signal; the only project with a live deployment, real operational data, and a reproduced-against-production validation
6. **ARC-Nodule** — strongest domain-specific (medical imaging) and experimental-rigor signal; list as an active/in-progress project rather than a completed one, since its headline mechanism is still under test
7. **Portfolio AI**
8. **Video Transcript RAG**

The six projects together give a much broader technical profile than presenting them all simply as “machine learning projects” — spanning systems/GPU engineering, novel architecture research, classical and adversarial ML, cybersecurity, production data engineering and applied statistics, and applied medical-imaging research with a strong emphasis on statistically rigorous experimental design.
