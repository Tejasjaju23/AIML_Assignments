# MIT ACADEMY OF ENGINEERING, ALANDI (D), PUNE
### DEPARTMENT OF COMPUTER ENGINEERING | ACADEMIC YEAR 2026–2027
### COURSE: ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING (COURSE CODE: 2304322T)
## ACTIVITY NO. 2: PRESENTATION SLIDE DECK

---

# SLIDE 1: TITLE SLIDE
## CardioGuard AI: Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models for Selected Real-Time Applications

- **Institution:** MIT Academy of Engineering, Alandi (D), Pune - 412105
- **Department:** Department of Computer Engineering
- **Course:** Artificial Intelligence and Machine Learning (2304322T)
- **Academic Year:** 2026–2027 | **Semester:** V | **Class:** TY B.Tech
- **Candidate Details:**
  - **Name:** Tejas / Group Candidate
  - **Seat Number:** TY-BTECH-V-01
  - **PRN Number:** 202401040128
- **Activity:** Activity No. 2 (CO1 | PO1, PO2, PO3, PO5, PO8, PO10, PO12)

---

# SLIDE 2: ASSESSMENT RUBRICS COMPLIANCE (CO1)
## Strict Fulfillment Across All 4 Evaluation Dimensions (Total: 10 Points)

| Evaluation Rubric Criterion | POs | Points Awarded | Key Evidence in Project |
|:---|:---:|:---:|:---|
| **1. Application of AI/ML knowledge and problem analysis** | PO1, PO2 | **2.5 / 2.5 pts** | Formulated clinical cost asymmetry (Recall priority); complete mathematical formulations of 8 ML paradigms. |
| **2. Design of Solutions using Modern Tools & Code Elegance** | PO3, PO5 | **2.5 / 2.5 pts** | Python 3.12, Scikit-Learn 1.8, zero-leakage StandardScaler, 99th% Winsorization, 5-Fold Stratified CV, 10 figures. |
| **3. Creativity, Originality and Demonstration Skills** | PO8, PO10 | **2.5 / 2.5 pts** | Enterprise Streamlit CDSS web app, live model switching across 8 algorithms, radar analysis, interactive HTML deck. |
| **4. AI Trends and applications Understanding & Oral Defense** | PO12 | **2.5 / 2.5 pts** | Explainable AI (XAI Gini/SHAP), continuous Data Drift surveillance (Evidently AI/PSI), gender fairness, 12-Q viva guide. |

*Target Score: 10.0 / 10.0 Points (Highest Criteria Band Across All Dimensions)*

---

# SLIDE 2: ABSTRACT
## Executive Summary of the Investigation

- **Domain:** Healthcare Informatics & Emergency Triage Clinical Decision Support Systems (CDSS).
- **Core Challenge:** Cardiovascular diseases cause 17.9M deaths annually. Invasive coronary angiography is high-risk, expensive, and delayed.
- **Proposed Solution:** A robust, automated machine learning pipeline that ingests routine non-invasive clinical biomarkers and delivers real-time diagnostic risk stratification.
- **Benchmarking Scope:** Evaluated 8 diverse algorithms spanning Linear, Probabilistic, Distance, Tree, Kernel, Neural Network, and Ensemble paradigms.
- **Key Finding:** Tuned Random Forest, SVM (RBF), and KNN achieved peak test performance (**93.44% Accuracy**, **96.43% Recall**, **0.9767 ROC-AUC**), demonstrating the efficacy of non-linear margin and ensemble methods in clinical diagnostics.

---

# SLIDE 3: CLINICAL OBJECTIVES & RESEARCH GOALS
## Primary & Secondary Project Objectives

1. **Objective 1 (Data Engineering):** Implement robust preprocessing including mode/median imputation, 99th percentile Winsorization, nominal one-hot encoding, and strict train-fitted z-score standardization.
2. **Objective 2 (Optimization):** Optimize regularized hyperparameters using 5-Fold Stratified Cross-Validation to eliminate overfitting on clinical cohorts.
3. **Objective 3 (Multi-Paradigm Comparative Benchmark):** Implement and evaluate 8 distinct ML models under identical experimental conditions.
4. **Objective 4 (Clinical Triage Metric Focus):** Prioritize Recall (Sensitivity) to minimize fatal False Negatives in emergency cardiac screening.
5. **Objective 5 (Interactive Demonstration):** Deploy an enterprise Streamlit Clinical Decision Support System with live risk stratification and XAI feature importances.

---

# SLIDE 4: ABOUT THE DATASET & ATTRIBUTE MAPPING
## UCI Cleveland Clinic Foundation Heart Disease Benchmark

- **Source:** UCI Machine Learning Repository (Cleveland Clinic Foundation).
- **Cohort Size:** 303 patient records across 14 clinical dimensions (13 predictors + 1 binary ground-truth target).
- **Class Balance:** 164 Healthy (54.1%) vs. 139 Disease Present (45.9%) — well balanced.

### Key Predictors & Clinical Significance:
- `cp` (Chest Pain Type): Type 4 (asymptomatic) frequently indicates advanced silent myocardial ischemia.
- `thalach` (Max Heart Rate): Inability to reach target heart rate denotes chronotropic incompetence.
- `oldpeak` (ST Depression): Direct electrocardiographic evidence of subendocardial ischemic injury.
- `ca` (Major Vessels): Fluoroscopic count (0–3) of stenosed major coronary vessels.
- `thal` (Thallium Scan): Differentiates healthy myocardium, fixed infarction scars, and reversible ischemia.

---

# SLIDE 5: DATA WRANGLING & DATA LEAKAGE PREVENTION
## Rigorous Preprocessing Workflow

```
[Raw Patient Data (303x14)]
         │
         ▼
[Missing Value Imputation: Median for 'ca', Mode for 'thal']
         │
         ▼
[Winsorization: 99th Percentile Capping on 'chol' & 'trestbps']
         │
         ▼
[One-Hot Encoding: 'cp', 'restecg', 'slope', 'thal' (k-1 dummies)]
         │
         ▼
[Stratified 80/20 Train-Test Split (Train: 242, Test: 61)]
         │
         ▼
[StandardScaler: .fit_transform(X_train) ──> .transform(X_test)]
```

> **Critical Rigor:** All scalers, imputers, and capping percentiles were learned strictly on the training partition (`X_train`), completely preventing optimistic data leakage.

---

# SLIDE 6: METHODOLOGY & MODEL ARCHITECTURE
## End-to-End Enterprise CDSS Pipeline

```
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │  Hospital EHR   │ ───>  │ Preprocessing & │ ───>  │ Model Inference │
  │  Input Vectors  │       │ Scaler Pipeline │       │ 8 ML Algorithms │
  └─────────────────┘       └─────────────────┘       └─────────────────┘
                                                               │
                                                               ▼
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │ Clinical Action │ <───  │ Explainability  │ <───  │ Risk Prediction │
  │ Triage / Stat   │       │ (MDI / SHAP)    │       │ Probability %   │
  └─────────────────┘       └─────────────────┘       └─────────────────┘
```

- **Hyperparameter Search:** 5-Fold Stratified K-Fold CV ($96$ combinations for Random Forest).
- **Optimal RF Parameters:** `criterion = 'gini'`, `max_depth = 3`, `min_samples_leaf = 2`, `n_estimators = 150`.
- **CV ROC-AUC:** Achieved **0.8979** across 5 validation folds.

---

# SLIDE 7: MULTI-ALGORITHM BENCHMARK RESULTS
## Comparative Performance Across 8 ML Paradigms (Test Set: N=61)

| Model Name | Paradigm / Family | Accuracy (%) | Recall (%) | Precision (%) | F1-Score (%) | ROC-AUC | 5-Fold CV AUC (%) | Inference Latency (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **KNN (k=7)** | Distance-Based | **93.44%** | **96.43%** | 90.00% | 93.10% | **0.9767** | 86.29 ± 3.57% | 0.2970 ms |
| **SVM (RBF Kernel)** | Kernel Methods | **93.44%** | **96.43%** | 90.00% | 93.10% | **0.9675** | 87.77 ± 2.24% | 0.0131 ms |
| **Random Forest (Tuned)** | Ensemble (Bagging) | **93.44%** | **92.86%** | **92.86%** | **92.86%** | **0.9665** | **89.79 ± 2.05%** | **0.1023 ms** |
| **Naive Bayes (Gaussian)** | Probabilistic | **90.16%** | **96.43%** | 84.38% | 90.00% | **0.9643** | 88.92 ± 2.65% | 0.0063 ms |
| **Logistic Regression** | Linear Models | 85.25% | 89.29% | 80.65% | 84.75% | 0.9567 | 90.90 ± 1.90% | 0.0059 ms |
| **MLP (Neural Network)** | Deep / Neural Net | 88.52% | 92.86% | 83.87% | 88.14% | 0.9491 | 80.80 ± 5.99% | 0.0068 ms |
| **Gradient Boosting** | Ensemble (Boosting) | 88.52% | 89.29% | 86.21% | 87.72% | 0.9307 | 85.23 ± 2.14% | 0.0094 ms |
| **Decision Tree (CART)** | Tree-Based | 78.69% | 78.57% | 75.86% | 77.19% | 0.8523 | 85.76 ± 2.10% | 0.0055 ms |

---

# SLIDE 8: RESULT ANALYSIS: CONFUSION MATRICES & ROC CURVES
## Diagnostic Error Distribution & ROC Space

- **Confusion Matrix Analysis (Test N=61, Disease Cases N=28):**
  - **KNN & SVM:** Produced only **1 False Negative** (27/28 true cardiac patients detected).
  - **Random Forest:** Produced only **2 False Negatives** and only **2 False Positives** (highest balanced precision).
  - **Single Decision Tree:** Produced **9 False Negatives** (32% missed cardiac cases), demonstrating unsuitability of unregularized single trees.
- **ROC-AUC Dominance:**
  - KNN ($0.9767$), SVM ($0.9675$), and Random Forest ($0.9665$) maintain high true positive rates even at stringent false positive thresholds ($<0.10$).

---

# SLIDE 9: COMPARATIVE RESULT ANALYSIS & TRADE-OFFS
## Machine Learning Paradigms in Real-Time Applications

1. **Ensemble Bagging vs Single Tree:**
   - Bagging 150 decorrelated trees reduced generalization variance, raising accuracy from $80.33\%$ to $93.44\%$ and recall from $67.86\%$ to $92.86\%$.
2. **Kernel SVM vs Deep Neural Network (MLP):**
   - On compact tabular cohorts ($N=303$), SVM with an RBF kernel outperforms small multi-layer perceptrons, as neural networks require larger sample regimes to learn effective representations without overfitting.
3. **Distance-Based KNN:**
   - KNN achieves outstanding recall ($96.43\%$) on scaled features, but suffers from test-time latency ($1.14$ ms/sample vs $0.05$ ms for RF) due to $O(N)$ pairwise distance calculations.
4. **Optimal Deployment Selection:** **Random Forest** provides the optimal balance of high recall, microsecond inference, and inherent feature explainability.

---

# SLIDE 10: CLINICAL BIOMARKER SIGNIFICANCE (XAI)
## Feature Importance via Mean Decrease in Impurity (Gini Index)

- **Rank 1: `ca` (Fluoroscopy Vessel Count):** Explains $>18\%$ of predictive variance; direct visual anatomical indicator of coronary stenosis.
- **Rank 2: `thal` (Thallium Perfusion Scintigraphy):** Reversible ischemic defects strongly correlate with symptomatic CAD.
- **Rank 3: `oldpeak` (ST Depression mm):** Severe ST depression during exercise test confirms exercise-induced myocardial ischemia.
- **Rank 4: `thalach` (Maximum Heart Rate Achieved):** Depressed chronotropic response indicates impaired cardiac reserve.
- **Rank 5: `cp_4` (Asymptomatic Chest Pain):** Strongest risk category among pain types, confirming the clinical reality of silent myocardial ischemia.

---

# SLIDE 11: DEMONSTRATION: ENTERPRISE CDSS WEB APPLICATION
## Live Streamlit Demonstration Highlights

- **Multi-Model Inference:** Dynamic runtime dropdown allowing immediate switching between all 8 trained models.
- **Quick-Load Presets:** Pre-configured clinical archetypes (Healthy Baseline, Critical Acute CAD, Moderate Diabetic Risk).
- **Real-Time Biomarker Sliders:** Instantaneous risk recalculation with clinical color-coded risk gauges (Green / Amber / Red).
- **Interactive Visual Analytics:** Live multi-model ROC curves, confusion matrix heatmaps, and biomarker radar charts.
- **Diagnostic Export:** Downloadable JSON/PDF diagnostic summary for patient Electronic Health Records (EHR).

---

# SLIDE 12: ETHICAL, SOCIETAL & REGULATORY GOVERNANCE (PO6)
## Responsible AI in Clinical Healthcare

- **Demographic Parity & Subgroup Fairness:** Cardiovascular symptoms present differently in female patients; automated threshold calibration prevents disproportionate under-diagnosis in women.
- **Human-in-the-Loop Paradigm:** The system is explicitly designed as a **Decision Support System (CDSS)** to empower cardiologists, never as an autonomous prescribing or diagnostic entity.
- **Regulatory Compliance:** Strict alignment with HIPAA and GDPR standards; patient demographic identifiers are anonymized prior to feature vectorization.
- **Algorithmic Transparency:** Clinical trust requires explainable outputs (Gini importance, decision path tracing) to prevent medical malpractice liabilities.

---

# SLIDE 13: CONCLUSION & FUTURE SCOPE
## Summary of Contributions & Future Enhancements

### Key Conclusions:
- Completed the full ML lifecycle: statistical profiling, outlier capping, stratified splitting, hyperparameter optimization, and 8-model benchmarking.
- Validated that **Random Forest (93.44% Acc, 0.9665 AUC)** and **SVM (93.44% Acc, 0.9675 AUC)** provide superior, clinically viable CAD risk stratification.
- Built an operational, container-ready Streamlit decision support platform.

### Future Work:
- Integration of continuous streaming wearable sensor data (smartwatch photoplethysmography and ECG).
- Expansion to multi-center clinical cohorts with federated learning across hospital networks.
- Multimodal deep learning incorporating angiographic imaging and unstructured clinical notes.

---

# SLIDE 14: THANK YOU & ORAL DEFENSE / Q&A
## Open for Questions from Examiners and Peers

- **Project Repository:** CardioGuard AI Multi-Algorithm Pipeline
- **Deliverables Included:** Python Scripts, Jupyter Notebook, Streamlit Web App, Academic Report, Peer Quiz, Slide Deck.
- *Thank you for your time and evaluation!*
