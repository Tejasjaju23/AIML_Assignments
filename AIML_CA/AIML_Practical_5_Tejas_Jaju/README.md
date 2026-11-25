# AIML Practical Assignment 5
**Course:** Artificial Intelligence and Machine Learning (AIML)  
**Academic Practical:** Practical Assignment 5  
  

---

## Student Information

| Field | Details |
|---|---|
| **Student Name** | Tejas Gopal Jaju |
| **PRN** | 202401040124 |
| **Roll Number** | 29 |
| **Batch** | A2 |

---

## 1. Project Title
**AIML Practical Assignment 5: Ensemble Learning and Performance Evaluation**  
*A Comparative Empirical Study of Stacking, AdaBoost, Hard Voting, Soft Voting Ensembles, and Five Base Machine Learning Classifiers.*

---

## 2. Assignment Objectives

The primary academic and technical objectives of this practical assignment are:
1. **Implement and evaluate heterogeneous Stacking Ensembles** using diverse Level-0 base estimators (Decision Tree, K-Nearest Neighbors, Support Vector Machine, and Logistic Regression) coupled with a Logistic Regression meta-classifier using 5-fold cross-validation.
2. **Implement and evaluate the AdaBoost Ensemble** utilizing 50 sequential Decision Stumps (`max_depth=1`) with exponential loss minimization and adaptive sample weight updating.
3. **Implement and evaluate Voting Classifiers** across both **Hard Voting** (discrete majority consensus) and **Soft Voting** (posterior class probability averaging).
4. **Train and benchmark five standard individual base classifiers**: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest.
5. **Construct a rigorous evaluation framework** with zero data leakage using Scikit-Learn `Pipeline` architectures, computing Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, Cross-Entropy Log Loss, and full Confusion Matrices.
6. **Deploy an interactive, modern Streamlit Web Application Dashboard** presenting live model inference, data exploration, performance leaderboards, and comparative architectural insights.

---

## 3. Introduction to Ensemble Learning

In statistical machine learning, single hypothesis estimators often suffer from either high **bias** (underfitting) or high **variance** (overfitting). Ensemble learning is a meta-algorithmic framework that combines predictions from multiple learning algorithms to produce a unified model with superior generalization capability, reduced variance, and lower error rates.

Ensemble methods can broadly be categorized into three paradigms:
1. **Bagging (Bootstrap Aggregation):** Reduces variance by training multiple independent, high-variance base learners (e.g., deep Decision Trees) on bootstrap subsamples in parallel and aggregating outputs via uniform voting or averaging (e.g., Random Forest).
2. **Boosting:** Reduces bias by training weak learners sequentially. Each successive learner focuses on the specific training instances misclassified by preceding estimators through dynamic sample weight redistribution (e.g., AdaBoost, Gradient Boosting).
3. **Stacking (Stacked Generalization):** Introduces a meta-learner that learns how to optimally combine the predictions of multiple diverse, heterogeneous base models via internal out-of-fold cross-validation.

---

## 4. Dataset Description

The project employs the **Breast Cancer Wisconsin (Diagnostic)** benchmark dataset available directly in `scikit-learn`:

- **Origin:** Computed from digitized images of Fine Needle Aspirates (FNA) of breast masses by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin).
- **Total Samples ($N$):** 569 instances
- **Total Features ($p$):** 30 continuous, real-valued input predictors describing cell nucleus characteristics:
  - *Ten fundamental properties:* Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, and Fractal Dimension.
  - *Three statistical aggregates per property:* Mean, Standard Error (SE), and "Worst" (mean of the three largest values).
- **Target Variable:** Binary diagnostic label:
  - `0`: **Malignant** (212 samples, 37.26%)
  - `1`: **Benign** (357 samples, 62.74%)
- **Missing / Null Values:** Exactly 0 missing values across all 30 features.
- **Data Splitting:** 80% Training ($N=455$) and 20% Testing ($N=114$).
  - **Stratified Partitioning (`stratify=y`):** Guarantees exact class proportions are maintained in both training and test splits.
  - **Random Seed:** `random_state=42` for deterministic reproducibility.
- **Data Leakage Prevention:** Feature standardization (`StandardScaler`) is encapsulated strictly inside `sklearn.pipeline.Pipeline` objects. Scaler statistics ($\mu, \sigma$) are fit *solely* on the training split, completely isolating test split distributions.

---

## 5. Algorithms Implemented

A total of **nine distinct machine learning models** are trained, evaluated, and compared:

### Base Classifiers (5 Models):
1. **Logistic Regression** (Pipeline: `StandardScaler` $\rightarrow$ `LogisticRegression(C=1.0, max_iter=1000)`)
2. **Decision Tree Classifier** (`DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)`)
3. **K-Nearest Neighbors (KNN)** (Pipeline: `StandardScaler` $\rightarrow$ `KNeighborsClassifier(n_neighbors=5, metric='minkowski')`)
4. **Support Vector Machine (SVM)** (Pipeline: `StandardScaler` $\rightarrow$ `SVC(kernel='rbf', C=1.0, probability=True)`)
5. **Random Forest Classifier** (`RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)`)

### Ensemble Classifiers (4 Models):
6. **Stacking Classifier** (Level-0: DT, KNN, SVM, LR; Level-1 Meta-Classifier: LR with 5-fold CV)
7. **AdaBoost Classifier** (Weak Learner: Decision Stump `max_depth=1`, `n_estimators=50`, `learning_rate=1.0`)
8. **Hard Voting Classifier** (Majority consensus across LR, DT, KNN, RF)
9. **Soft Voting Classifier** (Argmax of average posterior probability vectors across LR, DT, KNN, RF)

---

## 6. Algorithmic Principles & Mechanics

### 6.1 Stacking Ensemble
* **Architecture:** Level-0 consists of heterogeneous estimators (Decision Tree, KNN, SVM, Logistic Regression). Level-1 meta-estimator is Logistic Regression.
* **Meta-Feature Generation:** To prevent overfitting and target leakage, Stacking performs **5-fold internal cross-validation** on the training set. In each fold, base estimators fit on 4 folds and predict the holdout fold. These out-of-fold predictions concatenate into a meta-matrix $Z \in \mathbb{R}^{N_{train} \times 4}$.
* **How the Meta-Classifier Combines Predictions:** The meta-classifier fits on matrix $Z$. Instead of naive majority voting, the Logistic Regression meta-model computes regression coefficients $\beta_m$ that discover which base classifier provides the most reliable signal, dynamically blending their predictions.

### 6.2 AdaBoost Ensemble
* **Weak Learner:** Decision Stump (`DecisionTreeClassifier(max_depth=1)`).
* **Sample Weight Redistribution Mechanism:**
  1. Initialize all sample weights equally: $w_i^{(1)} = \frac{1}{N}$.
  2. In round $m \in \{1 \dots 50\}$:
     - Fit stump $h_m(x)$ minimizing weighted error $\epsilon_m = \sum_{i=1}^N w_i^{(m)} \mathbb{I}(y_i \neq h_m(x_i))$.
     - Compute estimator coefficient: $\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$.
     - Update weights: $w_i^{(m+1)} = w_i^{(m)} \exp\left(-\alpha_m y_i h_m(x_i)\right)$.
     - Normalize so $\sum w_i^{(m+1)} = 1$.
  3. **Intuition:** Correctly classified instances have their weights attenuated, while misclassified instances have weights multiplied. Subsequent stumps are forced to focus heavily on borderline or difficult samples.

### 6.3 Hard vs. Soft Voting Ensemble
* **Base Models:** Logistic Regression, Decision Tree, KNN, Random Forest.
* **Hard Voting:** Outputs the discrete mode: $\hat{y} = \text{mode}(\hat{y}_1, \dots, \hat{y}_M)$. Ignores individual model certainty. Cross-Entropy Log Loss cannot be computed because no class probability vectors are generated.
* **Soft Voting:** Averages posterior probability vectors:
  $$P_{ens}(c \mid x) = \frac{1}{M} \sum_{m=1}^M P_m(y = c \mid x)$$
  Selects $\hat{y} = \text{argmax}_c P_{ens}(c \mid x)$. Confident predictions have larger numerical influence, smoothing boundaries and yielding superior calibration (Log Loss = 0.1003).

---

## 7. Project Architecture & Directory Structure

```text
AIML_Practical_5_Tejas_Jaju/
│
├── app.py                     # Interactive multi-page Streamlit web dashboard
├── main.py                    # Complete end-to-end CLI execution pipeline
├── dataset.py                 # Dataset loading, profiling, and leakage-free splitting
├── models.py                  # Model definitions, Pipelines, and ensemble builders
├── evaluation.py              # Metric calculation, reports, and comparison logic
├── visualization.py           # Matplotlib & Seaborn publication-quality plotting
├── requirements.txt           # Minimal, standard Python library dependencies
├── README.md                  # Comprehensive academic report and project documentation
│
├── outputs/                   # Exported evaluation artifacts
│   ├── model_comparison.csv   # Complete tabular comparison data
│   ├── accuracy_comparison.png
│   ├── metrics_comparison.png
│   ├── log_loss_comparison.png
│   ├── class_distribution.png
│   ├── overall_model_comparison.png
│   └── confusion_matrices/    # High-resolution confusion matrices for all 9 models
│       ├── logistic_regression_cm.png
│       ├── decision_tree_cm.png
│       ├── knn_cm.png
│       ├── svm_cm.png
│       ├── random_forest_cm.png
│       ├── stacking_classifier_cm.png
│       ├── adaboost_classifier_cm.png
│       ├── hard_voting_classifier_cm.png
│       └── soft_voting_classifier_cm.png
│
└── screenshots/               # Visual previews of charts and dashboard components
```

---

## 8. Installation Instructions

1. Clone or extract the project folder onto your local Windows/Linux environment:
   ```bash
   cd "AIML_Practical_5_Tejas_Jaju"
   ```

2. Create a clean virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. Install the verified dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 9. Execution Commands

### A. Run CLI Pipeline (Full Model Training & Metric Generation):
```bash
python main.py
```
*Executes all 9 models, computes all metrics, outputs terminal reports, and saves all confusion matrices and comparison charts to `outputs/`.*

### B. Launch Streamlit Web Application:
```bash
streamlit run app.py
```
*Opens the responsive dashboard in your web browser at `http://localhost:8501`.*

---

## 10. Performance Evaluation Methodology

- **Holdout Test Evaluation:** All models are tested strictly on the identical 114 holdout instances (20% stratified test set).
- **Core Metrics Calculated:**
  - **Accuracy:** Proportion of overall correct classifications: $\frac{TP + TN}{TP + TN + FP + FN}$.
  - **Weighted Precision:** Weighted average of class precision: $\frac{TP}{TP + FP}$.
  - **Weighted Recall (Sensitivity):** Weighted average of class recall: $\frac{TP}{TP + FN}$.
  - **Weighted F1-Score:** Harmonic mean of precision and recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
  - **Cross-Entropy Log Loss:** $-\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^1 y_{ic} \ln(P_{ic})$ (lower is better; reflects probabilistic confidence calibration).
  - **Confusion Matrices:** Explicit $2 \times 2$ matrices with True Negative (TN), False Positive (FP), False Negative (FN), and True Positive (TP) counts.

---

## 11. Results and Observations

The following empirical results were obtained from actual test set predictions ($N=114$):

### Model Performance Comparison Table:

| Rank | Model | Category | Accuracy | Precision | Recall | F1-Score | Log Loss | Train Time (s) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Logistic Regression** | Base Classifier | **0.9825** | **0.9825** | **0.9825** | **0.9825** | **0.0777** | 0.0054 |
| 🥇 | **Support Vector Machine (SVM)** | Base Classifier | **0.9825** | **0.9825** | **0.9825** | **0.9825** | **0.0860** | 0.0116 |
| 🥈 | **Soft Voting Classifier** | Ensemble (Voting) | **0.9737** | **0.9747** | **0.9737** | **0.9735** | **0.1003** | 0.1191 |
| 🥈 | **Hard Voting Classifier** | Ensemble (Voting) | **0.9737** | **0.9740** | **0.9737** | **0.9737** | *N/A* | 0.1192 |
| 🥈 | **Stacking Classifier** | Ensemble (Stacking) | **0.9737** | **0.9737** | **0.9737** | **0.9736** | **0.0859** | 0.1551 |
| 🥉 | **K-Nearest Neighbors (KNN)** | Base Classifier | **0.9561** | **0.9561** | **0.9561** | **0.9560** | **0.4073** | 0.0019 |
| 🥉 | **Random Forest Classifier** | Base Classifier | **0.9561** | **0.9561** | **0.9561** | **0.9560** | **0.1099** | 0.1116 |
| 🥉 | **AdaBoost Classifier** | Ensemble (Boosting) | **0.9561** | **0.9569** | **0.9561** | **0.9558** | **0.3982** | 0.0880 |
| 4 | **Decision Tree Classifier** | Base Classifier | **0.9211** | **0.9234** | **0.9211** | **0.9216** | **1.9708** | 0.0042 |

### Key Observations:
1. **Linear and Kernel Classifiers Lead on Continuous Features:** Standalone Logistic Regression and RBF-kernel SVM reached the top classification accuracy of **98.25%** with only 2 misclassifications out of 114 test samples. In standardized 30-dimensional space, the malignant and benign classes exhibit near linear separability.
2. **Ensembles Ensure High Boundary Stability:** Stacking, Soft Voting, and Hard Voting all achieved **97.37%** accuracy, demonstrating how ensembling successfully insulates against the high variance and fragility of standalone Decision Trees (92.11%).
3. **Soft Voting Outperforms Hard Voting in Probability Calibration:** While both voting methods achieved identical 97.37% accuracy, Soft Voting attained a very low Log Loss of **0.1003** because it weights decisions by model confidence rather than discarding confidence.
4. **Decision Tree Underperformance:** Standalone Decision Tree had the highest Log Loss (1.9708) and lowest accuracy (92.11%), showing that orthogonal axis-aligned splits struggle on smooth continuous decision boundaries.

---

## 12. Strategic Guide: When to Use Each Ensemble Technique

| Ensemble Method | Best Use Cases | Key Advantages | Limitations |
|---|---|---|---|
| **Stacking** | Heterogeneous model collections; distinct algorithmic families (tree + linear + kernel); ample training data. | Learns non-linear combination weights; highest representational capacity. | High computation (cross-validation); risk of Level-1 overfitting if base models are too correlated. |
| **AdaBoost** | High-bias weak learners (shallow stumps); clean datasets with low label noise. | Rapidly drives training error to zero; highly effective on structured tabular data. | Extremely vulnerable to noisy outliers, which receive runaway weight inflation. |
| **Hard Voting** | Fast deployment; multiple equally tuned classifiers; probability outputs unavailable. | Simple; robust to uncalibrated probability estimates. | Discards classifier confidence; prone to arbitrary tie-breaking. |
| **Soft Voting** | Well-calibrated probabilistic models; multi-class classification where margin matters. | Confidence-weighted aggregation; smoother decision boundaries; superior log loss. | Sensitive to poorly calibrated or overconfident models. |

---

## 13. Limitations and Future Scope

### Limitations:
- The base models were evaluated with standard baseline hyperparameters rather than post-tuning optimality.
- The 30 features have high inter-feature correlations (e.g., radius, perimeter, and area are algebraically collinear).

### Future Research Directions:
1. **Automated Hyperparameter Optimization:** Implement Bayesian Optimization via **Optuna** or `GridSearchCV` to optimize $C$, $\gamma$, tree depths, and boosting learning rates.
2. **Modern Gradient Boosting Implementations:** Integrate **XGBoost**, **LightGBM**, and **CatBoost** to benchmark second-order gradient tree boosting against classical AdaBoost.
3. **Dimensionality Reduction & Feature Selection:** Apply Recursive Feature Elimination (RFE) or Principal Component Analysis (PCA) to alleviate collinearity prior to ensembling.
4. **Probability Calibration:** Apply Platt Scaling or Isotonic Regression to calibrate base model probabilities prior to soft voting aggregation.

---

## 14. Verification and Compliance Checklist

- [x] Programmatically loaded Breast Cancer Wisconsin dataset.
- [x] Verified 0 missing values and displayed class distribution.
- [x] Implemented 80:20 stratified split with `random_state=42`.
- [x] Standardized features using Scikit-Learn `Pipeline` to prevent data leakage.
- [x] Trained and evaluated all 5 base models (LR, DT, KNN, SVM, RF).
- [x] Implemented Stacking Classifier with Logistic Regression meta-classifier and 5-fold CV.
- [x] Implemented AdaBoost Classifier with Decision Stump (`max_depth=1`, $n=50$, $\eta=1.0$).
- [x] Implemented both Hard Voting and Soft Voting Classifiers.
- [x] Generated confusion matrices and classification reports for all 9 models.
- [x] Generated comparison table with Accuracy, Precision, Recall, F1, and Log Loss.
- [x] Exported `model_comparison.csv` and all PNG confusion matrices to `outputs/`.
- [x] Developed and verified the full multi-page Streamlit web dashboard (`app.py`).
- [x] Created clean `requirements.txt` with zero incompatible packages.
- [x] Verified full Windows compatibility without warnings.

---

*Submitted by Tejas Gopal Jaju (PRN: 202401040124, Roll No: 29, Batch A2) for AIML Practical Assignment 5.*
