# AIML Practical Assignment 4 – Classification Models

## 1. Student Details

| Field | Information |
| :--- | :--- |
| **Student Name** | Tejas Gopal Jaju |
| **PRN** | 202401040124 |
| **Roll Number** | 29 |
| **Batch** | A2 |
| **Subject** | Artificial Intelligence and Machine Learning (AIML) |
| **Practical Assignment** | 4 |
| **Assignment Title** | Classification Models |

---

## 2. Assignment Aim

To develop, evaluate, and compare four fundamental machine learning classification algorithms:
1. **Support Vector Machine (SVM)**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Gaussian Naïve Bayes**

Using the **Breast Cancer Wisconsin (Diagnostic) Dataset**, and evaluate each model using five core evaluation metrics:
- **Accuracy**
- **Precision (Weighted Average)**
- **Recall (Weighted Average)**
- **F1-Score (Weighted Average)**
- **Log Loss (Cross-Entropy Loss)**

Summarize the findings through comparative tables, visual diagnostic plots, and an interactive **Streamlit** user interface.

---

## 3. Technology Stack

- **Programming Language:** Python 3.10+ (Tested on Python 3.14)
- **Data Manipulation:** NumPy, Pandas
- **Machine Learning Library:** Scikit-learn (`sklearn`)
- **Data Visualization:** Matplotlib, Seaborn
- **Interactive Web UI:** Streamlit

---

## 4. Dataset Description

The project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset** provided natively by `sklearn.datasets.load_breast_cancer`:

- **Total Samples:** 569 instances
- **Total Features:** 30 continuous numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses
- **Target Classes:** Binary Classification
  - `0`: **Malignant** (212 samples, 37.26%) — Cancerous
  - `1`: **Benign** (357 samples, 62.74%) — Non-cancerous
- **Missing Values:** 0 (Clean dataset with no missing entries)

### Why this dataset is suitable for classification:
1. **Clinical Relevance:** Distinguishing benign from malignant tumors is a high-stakes clinical challenge where false negatives (diagnosing malignant as benign) can be fatal.
2. **Clean & High Quality:** 30 real-valued morphological descriptors (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) with zero missing data.
3. **Balanced Class Distribution:** 62.7% benign to 37.3% malignant provides an objective benchmark without requiring extreme resampling.
4. **Algorithmic Stress-Testing:** Contains collinear attributes (radius, perimeter, area) and non-linear boundaries, making it ideal for comparing linear vs non-linear and parametric vs non-parametric algorithms.

---

## 5. Data Preprocessing & Leakage Prevention

1. **Features & Target Separation:** Features ($X$) and target labels ($y$) are separated.
2. **Stratified Train-Test Split:**
   - **Ratio:** 80% Training (455 samples) and 20% Testing (114 samples).
   - **Stratification:** Maintains equal 62.7% / 37.3% class proportions across both splits.
   - **Reproducibility:** Seeded with `random_state=42`.
3. **Data Leakage Prevention:**
   - `StandardScaler` is fitted **strictly on `X_train`** (`scaler.fit(X_train)`).
   - Test data is transformed using the training parameters (`scaler.transform(X_test)`).
   - This ensures test distribution parameters do not bias the training phase.
4. **Algorithm-Specific Scaling Strategy:**
   - **SVM & KNN:** Use **StandardScaler (Scaled features)** because Euclidean distance and maximum-margin calculations are distorted by varying feature scales.
   - **Decision Tree & Naïve Bayes:** Use **Original (Unscaled features)** because Decision Trees make univariate orthogonal splits (scale-invariant), and Gaussian Naïve Bayes computes variance per feature independently.

---

## 6. Classification Models Implementation

### Model 1: Support Vector Machine (SVM)
- **Estimator:** `sklearn.svm.SVC`
- **Kernel:** Radial Basis Function (`kernel='rbf'`)
- **Probability Calibration:** `probability=True` (uses Platt scaling for valid `predict_proba()` output)
- **Feature Space:** Scaled features

### Model 2: Decision Tree
- **Estimator:** `sklearn.tree.DecisionTreeClassifier`
- **Criterion:** Gini Impurity (`criterion='gini'`)
- **Random State:** `42`
- **Feature Space:** Original features
- **Visualization:** Exportable tree diagram with Gini impurity, sample counts, and class majorities

### Model 3: K-Nearest Neighbors (KNN)
- **Estimator:** `sklearn.neighbors.KNeighborsClassifier`
- **Neighbors:** `n_neighbors=5`
- **Distance Metric:** Euclidean distance ($L_2$ Minkowski norm)
- **Feature Space:** Scaled features
- **Importance of K:** Balances variance (small $K$ overfits to noise) and bias (large $K$ over-smooths class boundaries).

### Model 4: Gaussian Naïve Bayes
- **Estimator:** `sklearn.naive_bayes.GaussianNB`
- **Feature Space:** Original features
- **Conditional Independence Assumption:** Assumes features are conditionally independent given the class label ($P(X_1, X_2 \dots | Y) = \prod P(X_i | Y)$). Despite correlated medical features, it computes fast $O(N \cdot D)$ closed-form estimates.

---

## 7. Performance Evaluation Metrics

Every model is evaluated using 5 distinct metrics:
1. **Accuracy:** Fraction of correct predictions: $\frac{TP + TN}{TP + TN + FP + FN}$
2. **Weighted Precision:** Ratio of true positive instances over all predicted positives, weighted by class support: $\sum w_i \frac{TP_i}{TP_i + FP_i}$
3. **Weighted Recall:** Ratio of true positive instances over all actual positive instances: $\sum w_i \frac{TP_i}{TP_i + FN_i}$
4. **Weighted F1-Score:** Harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
5. **Log Loss (Cross-Entropy Loss):**
   $$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i,k} \ln(p_{i,k})$$
   - Evaluates the probability confidence from `predict_proba()`.
   - Lower values are superior ($0.0$ indicates perfect prediction with $100\%$ confidence).
   - **Note:** Log Loss is **never** calculated as $1 - \text{Accuracy}$, because accuracy ignores prediction confidence.

---

## 8. Comparative Analysis & Actual Model Results

The models were trained and tested on the 80:20 split (`random_state=42`). Below are the actual numerical results:

### Model Performance Comparison Table

| Algorithm | Accuracy | Precision | Recall | F1-Score | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **0.9825** | **0.9825** | **0.9825** | **0.9825** | **0.0860** |
| **Decision Tree** | 0.9123 | 0.9161 | 0.9123 | 0.9130 | 3.1617 |
| **K-Nearest Neighbors (KNN)** | 0.9561 | 0.9561 | 0.9561 | 0.9560 | 0.4073 |
| **Gaussian Naïve Bayes** | 0.9386 | 0.9384 | 0.9386 | 0.9384 | 0.3713 |

### Training Accuracy vs Testing Accuracy (Overfitting Analysis)

| Algorithm | Training Accuracy | Testing Accuracy | Overfitting Observation |
| :--- | :---: | :---: | :--- |
| **Support Vector Machine** | 98.24% | 98.25% | **Perfect Generalization** (no overfitting) |
| **Decision Tree** | 100.00% | 91.23% | **Slight Overfitting** (unconstrained tree memorizes training set) |
| **K-Nearest Neighbors** | 97.36% | 95.61% | **Good Generalization** (slight variance gap) |
| **Gaussian Naïve Bayes** | 94.07% | 93.86% | **Excellent Consistency** (parametric stability) |

---

## 9. Comparative Findings and Algorithm Suitability

1. **Highest Accuracy:** **Support Vector Machine (98.25%)**
2. **Highest Precision:** **Support Vector Machine (0.9825)**
3. **Highest Recall:** **Support Vector Machine (0.9825)**
4. **Highest F1-Score:** **Support Vector Machine (0.9825)**
5. **Lowest Log Loss:** **Support Vector Machine (0.0860)**
6. **Best Overall Performer:** **Support Vector Machine (SVM)**

### Suitability Breakdown for Different Scenarios:
- **Best for High-Dimensional Data:** **Support Vector Machine (SVM)**. With 30 continuous clinical features, the RBF kernel maps inputs into higher-dimensional feature space, finding the maximum-margin hyperplane without suffering from high dimensionality.
- **Best for Computational Simplicity:** **Gaussian Naïve Bayes**. Closed-form parameter estimation makes it the fastest algorithm for training and inference with $O(N \cdot D)$ complexity.
- **Best for Small Datasets:** **Naïve Bayes & SVM**. Naïve Bayes has low variance due to simple parameter estimation, and SVM relies solely on support vectors.
- **Best for Interpretability & Auditability:** **Decision Tree**. Can be translated directly into intuitive medical if-then rules for clinical review.

---

## 10. Project Folder Structure

```
AIML_Practical_4_Tejas_Jaju/
│
├── app.py                             # Interactive Streamlit Web Application
├── main.py                            # Standalone Command-Line Execution Script
├── requirements.txt                   # Python Project Dependencies
├── README.md                          # Comprehensive Academic Documentation
│
├── models/
│   └── classification_models.py       # Model estimators, configs & training logic
│
├── utils/
│   ├── preprocessing.py               # Data loading, train-test split, scaling
│   └── evaluation.py                  # Metric calculations, log loss, reports export
│
├── visualizations/
│   └── plots.py                       # Bar charts, heatmaps, decision tree visualizer
│
├── outputs/
│   ├── comparison_results.csv         # Saved metrics comparison table
│   ├── classification_reports.txt     # Detailed classification reports
│   ├── metrics_comparison_dashboard.png # 5-metric comparison chart
│   ├── all_confusion_matrices.png     # 2x2 grid of confusion matrices
│   ├── decision_tree_visualization.png# Decision tree structure diagram
│   ├── class_distribution.png         # Target class distribution bar chart
│   ├── train_vs_test_accuracy.png     # Overfitting analysis chart
│   ├── metric_accuracy.png            # Accuracy comparison bar chart
│   ├── metric_precision.png           # Precision comparison bar chart
│   ├── metric_recall.png              # Recall comparison bar chart
│   ├── metric_f1_score.png            # F1-Score comparison bar chart
│   └── metric_log_loss.png            # Log Loss comparison bar chart
│
└── Report/
    ├── AIML_Practical_4_Report.pdf    # Academic Report (PDF Format)
    └── AIML_Practical_4_Report.docx   # Academic Report (Microsoft Word Format)
```

---

## 11. Installation & Execution Guide

### Prerequisites
- Python 3.10+ installed on Windows.
- Windows PowerShell.

### Step 1: Open Windows PowerShell in Project Directory
```powershell
cd c:\Users\tejas\Downloads\aiml_lab_ass4
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Complete Machine Learning Pipeline (Command Line)
```powershell
python main.py
```
This executes dataset inspection, data preprocessing, model training, metric evaluation, exports `outputs/comparison_results.csv` and `outputs/classification_reports.txt`, and generates all visualization figures.

### Step 4: Launch the Interactive Streamlit Web Application
```powershell
python -m streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 12. Final Practical Journal Conclusion

> In this practical assignment, four foundational machine learning classification algorithms (**Support Vector Machine**, **Decision Tree**, **K-Nearest Neighbors**, and **Gaussian Naïve Bayes**) were implemented, trained, and evaluated on the **Breast Cancer Wisconsin Diagnostic Dataset** using an 80:20 stratified train-test split.
>
> **Support Vector Machine (SVM)** with an RBF kernel demonstrated the best overall performance, achieving a test **Accuracy of 98.25%**, **F1-Score of 0.9825**, and a **Log Loss of 0.0860**, with only 1 misclassification out of 114 test samples. **StandardScaler** was critical in normalizing features for SVM and KNN to prevent large-magnitude features from dominating distance calculations, while fit parameters were calculated strictly on training data to prevent data leakage. **Gaussian Naïve Bayes** proved to be the fastest computationally, while **Decision Tree** offered the highest structural interpretability despite slight training-set overfitting.
>
> The project successfully achieved all learning objectives of AIML Practical Assignment 4.
