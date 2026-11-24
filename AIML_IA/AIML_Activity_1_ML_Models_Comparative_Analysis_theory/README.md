#  CardioGuard AI: Multi-Algorithm Machine Learning Pipeline for Cardiovascular Disease Risk Prediction

### MIT ACADEMY OF ENGINEERING, ALANDI (D), PUNE
### DEPARTMENT OF COMPUTER ENGINEERING | ACADEMIC YEAR: 2026–2027
### THIRD YEAR B.TECH (SEMESTER - V) | COURSE CODE: 2304322T
### COURSE: ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING
## ACTIVITY NO. 2: Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models for Selected Real-Time Applications

---

##  Executive Summary
**CardioGuard AI** is an enterprise-grade Clinical Decision Support System (CDSS) for non-invasive risk stratification of Coronary Artery Disease (CAD). Utilizing the UCI Cleveland Clinic Foundation benchmark dataset, this project implements the complete end-to-end Machine Learning software development lifecycle and provides an exhaustive comparative benchmark across **8 core Machine Learning algorithms spanning 7 paradigms**:
1. **Logistic Regression** (Linear Models)
2. **Gaussian Naive Bayes** (Probabilistic Models)
3. **K-Nearest Neighbors (KNN)** (Distance-Based Models)
4. **Decision Tree (CART)** (Tree-Based Models)
5. **Support Vector Machine (SVM RBF)** (Kernel Methods)
6. **Multi-Layer Perceptron (MLP)** (Deep / Neural Networks)
7. **Random Forest (Tuned)** (Ensemble Bagging)
8. **Gradient Boosting** (Ensemble Boosting)

---

##  Key Benchmark Results (Test Cohort N=61)

| Model Name | Family / Paradigm | Accuracy (%) | Recall (Sensitivity) (%) | Precision (%) | F1-Score (%) | ROC-AUC | Inference Latency (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **KNN (k=7)** | Distance-Based | **93.44%** | **96.43%** | 90.00% | 93.10% | **0.9767** | 1.140 ms |
| **SVM (RBF Kernel)** | Kernel Methods | **93.44%** | **96.43%** | 90.00% | 93.10% | **0.9675** | 0.033 ms |
| **Random Forest (Tuned)** | Ensemble (Bagging) | **93.44%** | **92.86%** | **92.86%** | **92.86%** | **0.9665** | **0.049 ms** |
| **Gradient Boosting** | Ensemble (Boosting) | 88.52% | 89.29% | 86.21% | 87.72% | 0.9307 | 0.066 ms |
| **Logistic Regression** | Linear Models | 85.25% | 89.29% | 80.65% | 84.75% | 0.9567 | 0.016 ms |
| **Decision Tree (CART)** | Tree-Based | 80.33% | 67.86% | 86.36% | 76.00% | 0.8923 | 0.016 ms |
| **MLP (Neural Network)** | Deep / Neural Net | 68.85% | 64.29% | 66.67% | 65.45% | 0.7554 | 0.040 ms |
| **Naive Bayes (Gaussian)**| Probabilistic | 60.66% | 89.29% | 54.35% | 67.57% | 0.8474 | 0.017 ms |

---

##  Project Repository Structure
```
AIML_Mini_Project-main/
│
├── figures/                               # 10 Publication-Grade Visualizations (300 DPI)
│   ├── 1_eda_distributions.png           # Feature distributions by CAD target
│   ├── 2_correlation_heatmap.png         # Pearson correlation matrix
│   ├── 3_outlier_boxplots.png            # Boxplots for outlier detection
│   ├── 4_categorical_risk_patterns.png   # Categorical cross-tabulations
│   ├── 5_bivariate_scatter.png           # Age vs Max Heart Rate interaction
│   ├── 6_confusion_matrices.png          # 2x4 Grid of confusion matrices (8 models)
│   ├── 7_roc_curves.png                  # Multi-model ROC curves
│   ├── 8_feature_importances.png         # Gini feature importance ranking
│   ├── 9_model_comparison_bar.png        # Comparative metrics bar chart
│   └── 10_algorithm_family_heatmap.png   # Performance matrix heatmap
│
├── heart_disease_uci.csv                  # Raw UCI Cleveland clinical dataset
├── heart_disease_preprocessed.csv         # Cleaned, Winsorized, One-Hot encoded dataset
├── train_and_evaluate.py                 # Full ML training, CV, hyperparameter tuning & evaluation
├── generate_notebook.py                   # Script to generate Jupyter Notebook
├── heart_disease_practical_exam.ipynb     # Interactive Jupyter Notebook for submission
├── build_reports.py                       # Generates publication DOCX and PDF reports
├── app.py                                 # Enterprise Streamlit Clinical Decision Support System
├── package_submission.py                  # Bundles all deliverables into submission ZIP
│
├── PROJECT_REPORT.pdf                     # Comprehensive Academic Project Report (PDF)
├── PROJECT_REPORT.docx                    # Academic Project Report (Word DOCX)
├── PROJECT_REPORT.md                      # Academic Project Report (Markdown)
├── PRESENTATION_SLIDES.md                 # Slide deck script & speaker notes
├── presentation.html                      # Standalone interactive browser slide deck
├── PEER_ASSESSED_QUIZ.md                  # 5-Mark Peer-Assessed Quiz & Answer Key (Guideline V)
├── VIVA_VOCE_PREPARATION.md               # 12+ Comprehensive viva questions and clinical answers
│
├── all_models.joblib                      # Serialized bundle of all 8 trained models
├── best_random_forest_model.joblib        # Tuned Random Forest model
├── scaler.joblib                          # StandardScaler fitted on X_train (Zero Data Leakage)
├── all_algorithms_benchmark.csv           # Detailed benchmark results table
├── evaluation_summary.json                # JSON summary with metadata & hyperparameters
├── feature_names.json                     # Encoded feature column names
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

##  Execution & Quick Start Guide (Windows PowerShell)

### 1. Set Up Virtual Environment (Recommended)
Open Windows PowerShell inside the project directory:
```powershell
# Navigate to the project root directory
cd "AIML_Mini_Project-main"

# Create a clean Python 3.10+ / 3.14 virtual environment
python -m venv venv

# If PowerShell script execution policy blocks activation, allow process-level execution:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### 2. Install Verified Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Execute Machine Learning Pipeline
Trains all 8 algorithms, performs 5-fold cross-validation, tunes Random Forest hyperparameters, and generates all 10 publication figures:
```powershell
python train_and_evaluate.py
```

### 4. Generate Jupyter Notebook & Reports
```powershell
python generate_notebook.py
python build_reports.py
```

### 5. Launch Streamlit Clinical Decision Support App
```powershell
streamlit run app.py
# Or alternately through Python module:
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser to interact with the live CDSS interface.

### 6. Package Submission Archive
```powershell
python package_submission.py
```
Creates `AIML_Activity_2_ML_Models_Comparative_Analysis.zip` ready for Moodle and Google Drive submission.

---

##  Technical Troubleshooting & Architecture Notes

### Resolving `ModuleNotFoundError: No module named '_loss'`
If running `app.py` or `joblib.load('all_models.joblib')` previously threw `ModuleNotFoundError: No module named '_loss'`, this issue has been comprehensively resolved with multi-layered architectural safeguards:

#### Root Cause Analysis:
1. In Scikit-learn, `GradientBoostingClassifier` computes binomial loss using internal Cython C-extensions (`sklearn._loss._loss.CyHalfBinomialLoss`).
2. When compiled, the Cython type's internal C-level name defaults to `_loss.CyHalfBinomialLoss` rather than the fully package-qualified `sklearn._loss._loss.CyHalfBinomialLoss`.
3. Standard `joblib.dump` / `pickle.dump` saves the global reference as `('_loss', 'CyHalfBinomialLoss')`.
4. When unpickled in a fresh process or differing virtual environment where `_loss` is not explicitly imported or registered at top-level `sys.modules['_loss']`, Python's unpickler attempts `import _loss` and fails with `ModuleNotFoundError: No module named '_loss'`.

#### Implemented Multi-Layered Resolution:
1. **Runtime Module Resolution Shim (`app.py` & `train_and_evaluate.py`)**:
   At application initialization (before any model deserialization), `sys.modules['_loss']` is mapped to `sklearn._loss._loss`:
   ```python
   import sys
   if "_loss" not in sys.modules:
       try:
           import sklearn._loss._loss as _loss_cython
           sys.modules["_loss"] = _loss_cython
       except (ImportError, ModuleNotFoundError):
           try:
               import sklearn._loss as _loss_pkg
               sys.modules["_loss"] = _loss_pkg
           except (ImportError, ModuleNotFoundError):
               pass
   ```
2. **Clean Cross-Environment Serialization (`train_and_evaluate.py`)**:
   Before dumping `all_models.joblib`, the model bundle is saved using clean deepcopies where transient Cython `_loss` references are detached. This guarantees the pickled byte stream contains **zero references to `_loss` or `CyHalfBinomialLoss`**, making the file 100% portable across Python, Scikit-learn, and OS environments.
3. **In-Memory Dynamic Loss Reconstitution (`app.py`)**:
   Upon unpickling, any gradient boosting estimator dynamically reconstructs its native `_loss` function via `model._loss = model._get_loss(sample_weight=None)`. This ensures `.predict()` and `.predict_proba()` work with 100% mathematical fidelity.
4. **Artifact Backups**:
   Original serialized models have been safely preserved under `all_models.joblib.bak` and `best_random_forest_model.joblib.bak`.

### Windows 11 `wmic` Deprecation Notice
On recent builds of Windows 11, the legacy `wmic.exe` utility is deprecated, which can trigger a `loky` backend core count warning. CardioGuard AI automatically sets `os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)` at the entry point of all scripts, eliminating this warning while ensuring optimal multi-core utilization.

---

## 🛡️ Academic & Ethical Compliance
- **Course Outcomes:** CO2 (Understand and apply ML principles) & CO4 (Analyze, evaluate, use modern tools, and consider ethical implications).
- **RBT Level:** Level 4 (Analyze).
- **Ethics (PO6):** Subgroup sensitivity auditing for gender fairness, HIPAA/GDPR data sanitization, and Human-in-the-Loop decision support.
- **Guideline II:** Candidate Name, PRN, and Seat Number clearly displayed on all deliverables.
- **Guideline V:** Includes verified 5-mark peer-assessed quiz with full rationales.
