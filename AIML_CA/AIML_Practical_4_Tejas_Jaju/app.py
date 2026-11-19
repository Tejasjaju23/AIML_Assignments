"""
AIML Practical Assignment 4 – Classification Models
Interactive Streamlit Web Application

Student Information:
- Name: Tejas Gopal Jaju
- PRN: 202401040124
- Roll Number: 29
- Batch: A2
- Subject: Artificial Intelligence and Machine Learning
- Practical Assignment: 4
- Title: Classification Models
"""

import os
import sys

# Ensure current directory is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ensure local writable cache directories and clean execution on Windows
os.environ.setdefault('MPLCONFIGDIR', os.path.join(current_dir, '.matplotlib'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocessing import (
    load_dataset,
    prepare_data,
    get_dataset_summary,
    get_dataset_rationale
)
from models.classification_models import (
    get_models_dict,
    train_models,
    train_single_model
)
from utils.evaluation import (
    evaluate_all_models,
    generate_comparative_analysis
)
from visualizations.plots import (
    plot_metric_comparison,
    plot_all_metrics_comparison,
    plot_confusion_matrix,
    plot_all_confusion_matrices,
    plot_decision_tree_vis,
    plot_class_distribution,
    plot_train_vs_test_accuracy
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AIML Practical 4 - Classification Models",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished college journal / project styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .student-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .info-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .highlight-card {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_data():
    df, X, y, feature_names, target_names, raw_data = load_dataset()
    data_dict = prepare_data(X, y, test_size=0.20, random_state=42)
    summary = get_dataset_summary(df)
    return df, X, y, feature_names, target_names, data_dict, summary

@st.cache_resource
def get_cached_models_and_evaluation(_data_dict, _target_names):
    models_dict = get_models_dict()
    trained_models = train_models(_data_dict, models_dict)
    results_df, detailed_results = evaluate_all_models(
        trained_models, _data_dict, models_dict, class_names=_target_names
    )
    findings, conclusion_text = generate_comparative_analysis(results_df, detailed_results)
    return models_dict, trained_models, results_df, detailed_results, findings, conclusion_text

# Load data and models
df, X, y, feature_names, target_names, data_dict, summary = get_cached_data()
models_dict, trained_models, results_df, detailed_results, findings, conclusion_text = get_cached_models_and_evaluation(
    data_dict, target_names
)

# -----------------------------------------------------------------------------
# Sidebar Configuration: Student Details & Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Student Profile")
    st.markdown("""
    **Name:** Tejas Gopal Jaju  
    **PRN:** `202401040124`  
    **Roll No.:** `29`  
    **Batch:** `A2`  
    **Subject:** AIML  
    **Practical:** 4  
    """)
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    menu = st.radio(
        "Select Section:",
        [
            "1. Home / Overview",
            "2. Dataset Information",
            "3. Data Preprocessing",
            "4. Model Training",
            "5. Performance Evaluation",
            "6. Confusion Matrices",
            "7. Comparative Analysis",
            "8. Conclusion & Viva Guide"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Quick Info")
    st.info("**Dataset:** Breast Cancer Wisconsin\n\n**Instances:** 569 | **Features:** 30\n\n**Split:** 80% Train, 20% Test")

# -----------------------------------------------------------------------------
# PAGE 1: HOME / OVERVIEW
# -----------------------------------------------------------------------------
if menu == "1. Home / Overview":
    st.markdown('<div class="main-title">AIML Practical Assignment 4 – Classification Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparative Analysis of SVM, Decision Tree, K-Nearest Neighbors, and Naïve Bayes</div>', unsafe_allow_html=True)
    
    # Student Details Banner
    st.markdown("""
    <div class="student-card">
        <h4 style="margin: 0; color: #1e3a8a;">📋 Academic Practical Submission</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 8px;">
            <div><strong>Student Name:</strong> Tejas Gopal Jaju</div>
            <div><strong>PRN:</strong> 202401040124</div>
            <div><strong>Roll Number:</strong> 29</div>
            <div><strong>Batch:</strong> A2</div>
            <div><strong>Subject:</strong> Artificial Intelligence & Machine Learning</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Assignment Aim & Objective")
    st.markdown("""
    The primary aim of this practical is to develop and evaluate four foundational machine learning classification models:
    1. **Support Vector Machine (SVM):** Utilizing the Radial Basis Function (RBF) kernel to maximize hyperplanes in high dimensions.
    2. **Decision Tree Classifier:** Recursive binary splitting using the Gini Impurity criterion to build an interpretable rule hierarchy.
    3. **K-Nearest Neighbors (KNN):** Distance-based instance learning using majority voting among the $K=5$ nearest neighbors.
    4. **Gaussian Naïve Bayes:** Probabilistic classification utilizing Bayes' Theorem with continuous feature likelihoods.
    
    **Evaluation Strategy:**
    Evaluate each algorithm across **Accuracy**, **Weighted Precision**, **Weighted Recall**, **Weighted F1-Score**, and **Log Loss** (via predicted probabilities), highlighting strengths, limitations, and real-world applicability in medical diagnosis.
    """)
    
    st.markdown("### 🛠️ Core Technology Stack")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **Core Computation**
        - Python 3.14
        - NumPy
        - Pandas
        """)
    with col2:
        st.markdown("""
        **Machine Learning**
        - Scikit-learn
        - StandardScaler
        - Model Evaluation API
        """)
    with col3:
        st.markdown("""
        **Visualization**
        - Matplotlib
        - Seaborn
        - Graph Tree Plot
        """)
    with col4:
        st.markdown("""
        **User Interface**
        - Streamlit Dashboard
        - Responsive Layouts
        - Interactive Plots
        """)

    st.markdown("---")
    st.markdown("### 🔄 End-to-End Workflow Pipeline")
    st.markdown("""
    ```
    ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
    │  Wisconsin Dataset   │ ──> │   80:20 Split &      │ ──> │ Feature Representation│
    │  (569 Rows, 30 Cols) │     │   Leakage Prevention │     │ (Scaled vs Original) │
    └──────────────────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                                         │
    ┌──────────────────────┐     ┌──────────────────────┐                ▼
    │ Comparative Findings │ <── │ 5 Evaluation Metrics │ <── ┌──────────────────────┐
    │  & Viva Exam Guide   │     │ (Acc, Prec, Rec, F1, │     │ 4 Trained Classifiers│
    │                      │     │  True Log Loss)      │     │ (SVM, DT, KNN, GNB)  │
    └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
    ```
    """)

# -----------------------------------------------------------------------------
# PAGE 2: DATASET INFORMATION
# -----------------------------------------------------------------------------
elif menu == "2. Dataset Information":
    st.markdown("## 📊 Dataset Information & Exploration")
    st.markdown("Explore the **Breast Cancer Wisconsin (Diagnostic) Dataset** provided natively by `sklearn.datasets`.")
    
    # Summary Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records (Rows)", summary['total_rows'])
    with c2:
        st.metric("Total Features", summary['total_features'])
    with c3:
        st.metric("Missing Values", summary['missing_values_count'])
    with c4:
        st.metric("Target Classes", f"2 ({target_names[0]} / {target_names[1]})")
        
    st.markdown("---")
    
    # Dataset Preview & Class Distribution
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📋 Dataset Preview (First 5 Rows)")
        preview_mode = st.radio("Display columns:", ["Key Diagnostic Features (10)", "All 30 Features"], horizontal=True)
        if preview_mode == "Key Diagnostic Features (10)":
            st.dataframe(df[feature_names[:10] + ['class_name']].head(5), use_container_width=True)
        else:
            st.dataframe(df.head(5), use_container_width=True)
            
    with col_right:
        st.subheader("🎯 Target Class Distribution")
        fig_dist = plot_class_distribution(y, target_names)
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        
    # Statistical Summary & Rationale
    st.markdown("---")
    st.subheader("🔬 Clinical Significance & Suitability Rationale")
    st.markdown("""
    <div class="info-box">
        <strong>Why this dataset is ideal for classification benchmark:</strong>
        <ol>
            <li><strong>High Clinical Relevance:</strong> Accurate classification between <em>benign</em> (non-cancerous) and <em>malignant</em> (cancerous) is a mission-critical diagnostic task. Missing a malignant case (False Negative) has grave clinical consequences.</li>
            <li><strong>Continuous Multi-Dimensional Feature Space:</strong> 30 real-valued features computed from fine-needle aspirates (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).</li>
            <li><strong>Data Quality:</strong> Zero missing entries eliminates arbitrary imputation bias, allowing direct focus on algorithmic performance.</li>
            <li><strong>Well-Formed Benchmark:</strong> Natural 62.7% benign to 37.3% malignant distribution provides sufficient instances for both classes without extreme class imbalance.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 View Complete List of 30 Features & Data Types"):
        feat_df = pd.DataFrame({
            "Feature Index": range(1, len(feature_names) + 1),
            "Feature Name": feature_names,
            "Type": ["float64"] * len(feature_names),
            "Sample Mean": [X[col].mean().round(4) for col in feature_names],
            "Sample Std": [X[col].std().round(4) for col in feature_names]
        })
        st.dataframe(feat_df, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 3: DATA PREPROCESSING
# -----------------------------------------------------------------------------
elif menu == "3. Data Preprocessing":
    st.markdown("## ⚙️ Data Preprocessing & Leakage Prevention")
    st.markdown("Understanding data partitioning, feature standardization, and prevention of train-test information leakage.")
    
    # 3-step breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 1. Stratified Train-Test Split (80:20)
        - **Total Dataset:** 569 instances
        - **Training Set (80%):** 455 instances
        - **Testing Set (20%):** 114 instances
        - **Stratification:** Preserves the 62.7% benign / 37.3% malignant class ratio across both splits.
        - **Random State:** Fixed at `42` for guaranteed academic reproducibility.
        """)
        
    with col2:
        st.markdown("""
        ### 2. StandardScaler & Leakage Prevention
        - **Formula:** $z = \\frac{x - \\mu}{\\sigma}$
        - **Data Leakage Rule:** The scaler was **fitted strictly on the training set (`X_train`)**.
        - `X_test` was transformed using the training mean and standard deviation:
          - `scaler.fit_transform(X_train)`
          - `scaler.transform(X_test)`
        - Prevents test statistics from artificially biasing model learning.
        """)
        
    st.markdown("---")
    st.subheader("🎯 Algorithmic Feature Representation Requirements")
    
    st.markdown("""
    | Algorithm | Feature Representation | Reason for Choice |
    | :--- | :---: | :--- |
    | **Support Vector Machine (SVM)** | **Standardized (Scaled)** | Calculates Euclidean distances between support vectors; unscaled large values would dominate the margin. |
    | **Decision Tree** | **Original (Unscaled)** | Scale-invariant. Splits are evaluated per feature independently via threshold cutoffs, unaffected by monotonic scaling. |
    | **K-Nearest Neighbors (KNN)** | **Standardized (Scaled)** | Calculates Euclidean/Minkowski distance between neighbors. Unscaled features with large ranges distort neighborhood radius. |
    | **Gaussian Naïve Bayes** | **Original (Unscaled)** | Computes mean $\\mu$ and variance $\\sigma^2$ per feature independently. Linear standardization does not alter likelihood ratios. |
    """)
    
    st.markdown("---")
    st.subheader("📈 Visualizing the Effect of Standardization")
    st.markdown("Notice how original features like `mean area` (range 140 to 2500) dominate features like `mean smoothness` (range 0.05 to 0.16) before scaling:")
    
    sample_feats = ['mean radius', 'mean texture', 'mean area', 'mean smoothness']
    fig_scale, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    
    data_dict['X_train'][sample_feats].boxplot(ax=ax1)
    ax1.set_title("Before Scaling (Original Features)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Original Units")
    ax1.tick_params(axis='x', rotation=20)
    
    data_dict['X_train_scaled'][sample_feats].boxplot(ax=ax2)
    ax2.set_title("After StandardScaler (Zero Mean, Unit Variance)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Standard Deviations (Z-score)")
    ax2.tick_params(axis='x', rotation=20)
    
    plt.tight_layout()
    st.pyplot(fig_scale)
    plt.close(fig_scale)

# -----------------------------------------------------------------------------
# PAGE 4: MODEL TRAINING
# -----------------------------------------------------------------------------
elif menu == "4. Model Training":
    st.markdown("## 🧠 Model Training & Overfitting Analysis")
    st.markdown("Execution details for all four machine learning models.")
    
    # Model detail tabs
    tab_svm, tab_dt, tab_knn, tab_nb = st.tabs([
        "1. Support Vector Machine",
        "2. Decision Tree",
        "3. K-Nearest Neighbors",
        "4. Naïve Bayes"
    ])
    
    with tab_svm:
        st.markdown("### 🔹 Support Vector Machine (SVM)")
        st.markdown(r"""
        - **Estimator:** `sklearn.svm.SVC(kernel='rbf', probability=True, random_state=42)`
        - **Kernel:** Radial Basis Function (RBF) measures similarity: $K(x, x') = \exp(-\gamma ||x - x'||^2)$
        - **Probability Calibration:** `probability=True` enables Platt scaling so `predict_proba()` produces calibrated probabilities for log loss.
        """)
        svm_train = detailed_results['Support Vector Machine']['training_accuracy'] * 100
        svm_test = detailed_results['Support Vector Machine']['accuracy'] * 100
        st.markdown(f"- **Training Accuracy:** `{svm_train:.2f}%` | **Testing Accuracy:** `{svm_test:.2f}%`")
        st.info("💡 **Key Strength:** Maximizes the geometric margin between classes, excelling in high-dimensional feature spaces.")
        
    with tab_dt:
        st.markdown("### 🔹 Decision Tree Classifier")
        st.markdown(r"""
        - **Estimator:** `sklearn.tree.DecisionTreeClassifier(criterion='gini', random_state=42)`
        - **Splitting Criterion:** Gini Impurity: $I_G(p) = 1 - \sum p_i^2$
        """)
        dt_train = detailed_results['Decision Tree']['training_accuracy'] * 100
        dt_test = detailed_results['Decision Tree']['accuracy'] * 100
        st.markdown(f"- **Training Accuracy:** `{dt_train:.2f}%` | **Testing Accuracy:** `{dt_test:.2f}%`")
        st.warning("⚠️ **Overfitting Note:** The tree achieves 100% training accuracy because leaves continue splitting until nodes are pure, resulting in a slight drop to 91.23% on unseen test data.")
        
    with tab_knn:
        st.markdown("### 🔹 K-Nearest Neighbors (KNN)")
        st.markdown(r"""
        - **Estimator:** `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)`
        - **Distance Metric:** Euclidean distance ($L_2$ norm)
        - **Importance of K:**
          - **Small K ($K=1$):** High variance, low bias (sensitive to noisy outliers and overfitting).
          - **Large K ($K=50$):** High bias, low variance (over-smooths decision boundary, underfitting).
          - **$K=5$:** Balances local detail with neighborhood stability.
        """)
        knn_train = detailed_results['K-Nearest Neighbors']['training_accuracy'] * 100
        knn_test = detailed_results['K-Nearest Neighbors']['accuracy'] * 100
        st.markdown(f"- **Training Accuracy:** `{knn_train:.2f}%` | **Testing Accuracy:** `{knn_test:.2f}%`")
        
    with tab_nb:
        st.markdown("### 🔹 Gaussian Naïve Bayes")
        st.markdown(r"""
        - **Estimator:** `sklearn.naive_bayes.GaussianNB()`
        - **Probability Density:** $P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$
        - **Independence Assumption:** Assumes features are conditionally independent given class $y$: $P(X \mid y) = \prod_{i=1}^{30} P(x_i \mid y)$.
        """)
        nb_train = detailed_results['Naïve Bayes']['training_accuracy'] * 100
        nb_test = detailed_results['Naïve Bayes']['accuracy'] * 100
        st.markdown(f"- **Training Accuracy:** `{nb_train:.2f}%` | **Testing Accuracy:** `{nb_test:.2f}%`")
        st.info(r"💡 **Key Strength:** Extremely fast with $O(N \cdot D)$ training complexity, ideal for real-time applications.")

    st.markdown("---")
    st.subheader("📊 Training vs Testing Accuracy (Overfitting vs Generalization)")
    fig_tt = plot_train_vs_test_accuracy(detailed_results)
    st.pyplot(fig_tt)
    plt.close(fig_tt)

# -----------------------------------------------------------------------------
# PAGE 5: PERFORMANCE EVALUATION
# -----------------------------------------------------------------------------
elif menu == "5. Performance Evaluation":
    st.markdown("## 📈 Performance Evaluation (5 Standard Metrics)")
    st.markdown("Detailed breakdown of model evaluations using standard scikit-learn metrics.")
    
    # Model Selector
    selected_model = st.selectbox(
        "Select Model to Inspect Detailed Evaluation:",
        list(models_dict.keys()),
        index=0
    )
    
    res = detailed_results[selected_model]
    
    # Five Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{res["accuracy"]:.4f}</div><div class="metric-label">Accuracy</div><small>{res["accuracy"]*100:.2f}%</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{res["precision"]:.4f}</div><div class="metric-label">Precision</div><small>Weighted Avg</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{res["recall"]:.4f}</div><div class="metric-label">Recall</div><small>Weighted Avg</small></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{res["f1_score"]:.4f}</div><div class="metric-label">F1-Score</div><small>Weighted Avg</small></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: #dc2626;">{res["log_loss"]:.4f}</div><div class="metric-label">Log Loss</div><small>Lower is Better</small></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_rep, col_cm = st.columns([3, 2])
    with col_rep:
        st.subheader("📄 Scikit-Learn Classification Report")
        st.code(res['classification_report_text'], language='text')
        
    with col_cm:
        st.subheader("🎯 Model Confusion Matrix")
        fig_single_cm = plot_confusion_matrix(res['confusion_matrix'], target_names, title=f"{selected_model} Confusion Matrix")
        st.pyplot(fig_single_cm)
        plt.close(fig_single_cm)
        
    # Log Loss Theoretical Explanation
    st.markdown("---")
    st.subheader("💡 Why Log Loss is NOT '1 - Accuracy'")
    st.markdown("""
    <div class="info-box">
        <strong>Academic Distinction between Accuracy and Log Loss:</strong>
        <ul>
            <li><strong>Accuracy</strong> only measures discrete binary correctness (whether $\\hat{y} == y$), ignoring how confident the model was.</li>
            <li><strong>Log Loss (Cross-Entropy Loss)</strong> evaluates the <em>probabilities</em> output by <code>predict_proba()</code>:
            $$\\text{Log Loss} = -\\frac{1}{N} \\sum_{i=1}^N \\left[ y_i \\ln(p_i) + (1 - y_i) \\ln(1 - p_i) \\right]$$
            </li>
            <li>If a model predicts <strong>0.51</strong> probability for a positive label and is correct, accuracy counts it as a full success. But Log Loss penalizes the uncertainty!</li>
            <li>If a model predicts <strong>0.99</strong> probability for a positive label and is wrong, Log Loss imposes an enormous penalty ($-\\ln(0.01) \\approx 4.6$).</li>
            <li>Therefore, <strong>Log Loss measures probabilistic confidence calibration</strong>, where lower values are strictly superior.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 6: CONFUSION MATRICES
# -----------------------------------------------------------------------------
elif menu == "6. Confusion Matrices":
    st.markdown("## 🔍 Confusion Matrices Comparison")
    st.markdown("Heatmaps showing exact True Positives, True Negatives, False Positives, and False Negatives across all four models.")
    
    # Display 2x2 grid
    fig_all_cm = plot_all_confusion_matrices(detailed_results, target_names)
    st.pyplot(fig_all_cm)
    plt.close(fig_all_cm)
    
    st.markdown("---")
    st.subheader("🏥 Clinical Diagnostic Interpretation (Cancer Screening Context)")
    st.markdown("""
    In breast cancer diagnostics:
    - **True Negative (TN - Top Left):** Patient has **malignant** tumor and model correctly detects malignancy. (Note: in dataset, Class 0 = Malignant).
    - **True Positive (TP - Bottom Right):** Patient has **benign** tumor and model correctly diagnoses benign.
    - **False Positive (FP - Top Right / Type I Error):** Model diagnoses Benign when the patient actually has Malignant cancer!
    - **False Negative (FN - Bottom Left / Type II Error):** Model diagnoses Malignant when the patient was Benign (leads to unnecessary follow-up biopsy).
    
    **Critical Diagnostic Observation:**
    - **Support Vector Machine (SVM)** achieved only **1 error** across the entire 114-patient test cohort (98.25% test accuracy)!
    - **Decision Tree** made 10 misclassifications (3 false benigns, 7 false malignancies).
    """)

# -----------------------------------------------------------------------------
# PAGE 7: COMPARATIVE ANALYSIS
# -----------------------------------------------------------------------------
elif menu == "7. Comparative Analysis":
    st.markdown("## 📊 Comprehensive Comparative Analysis")
    st.markdown("Direct side-by-side comparison across all four classification algorithms.")
    
    # Official Comparison Table
    st.subheader("📋 Official Model Comparison Table")
    
    # Formatting table to highlight best performers
    def highlight_best(s):
        if s.name in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            is_best = s == s.max()
            return ['background-color: #dcfce7; font-weight: bold;' if v else '' for v in is_best]
        elif s.name == 'Log Loss':
            is_best = s == s.min()
            return ['background-color: #dcfce7; font-weight: bold;' if v else '' for v in is_best]
        return ['' for _ in s]
        
    styled_df = results_df.style.apply(highlight_best, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Log Loss'])
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Visual Comparison Across All 5 Metrics")
    fig_dash = plot_all_metrics_comparison(results_df)
    st.pyplot(fig_dash)
    plt.close(fig_dash)
    
    # Individual Metric Selectable View
    st.markdown("---")
    st.subheader("🔍 Individual Metric Focus")
    sel_metric = st.selectbox(
        "Choose metric to visualize:",
        ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Log Loss'],
        index=0
    )
    fig_ind = plot_metric_comparison(results_df, metric=sel_metric)
    st.pyplot(fig_ind)
    plt.close(fig_ind)
    
    # Decision Tree Structural Visualization
    st.markdown("---")
    st.subheader("🌳 Decision Tree Structure Visualization")
    st.markdown("Visualizing the recursive Gini splits generated by the Decision Tree classifier:")
    depth_choice = st.slider("Select Tree Visualization Depth:", min_value=1, max_value=4, value=3)
    fig_dt = plot_decision_tree_vis(
        trained_models["Decision Tree"],
        feature_names,
        target_names,
        max_depth=depth_choice
    )
    st.pyplot(fig_dt)
    plt.close(fig_dt)

# -----------------------------------------------------------------------------
# PAGE 8: CONCLUSION & VIVA GUIDE
# -----------------------------------------------------------------------------
elif menu == "8. Conclusion & Viva Guide":
    st.markdown("## 🎓 Automated Findings, Conclusion & Viva Preparation")
    
    # Dynamic automated findings
    st.subheader("🏆 Automated Analytical Findings (Based on Actual Results)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        **Highest Accuracy:**  
        `{findings['highest_accuracy']['algorithm']}` ({findings['highest_accuracy']['score']:.4f})  
        
        **Highest Precision:**  
        `{findings['highest_precision']['algorithm']}` ({findings['highest_precision']['score']:.4f})
        """)
    with c2:
        st.markdown(f"""
        **Highest Recall:**  
        `{findings['highest_recall']['algorithm']}` ({findings['highest_recall']['score']:.4f})  
        
        **Highest F1-Score:**  
        `{findings['highest_f1']['algorithm']}` ({findings['highest_f1']['score']:.4f})
        """)
    with c3:
        st.markdown(f"""
        **Lowest Log Loss:**  
        `{findings['lowest_log_loss']['algorithm']}` ({findings['lowest_log_loss']['score']:.4f})  
        
        **Best Overall Model:**  
        `{findings['best_overall']}`
        """)
        
    st.markdown("---")
    st.subheader("⚖️ Algorithm Suitability Matrix")
    
    st.markdown("""
    | Evaluation Dimension | Best Suited Algorithm | Technical Justification |
    | :--- | :--- | :--- |
    | **Highest Overall Accuracy** | **Support Vector Machine (SVM)** | The RBF kernel projects the 30 continuous features into an infinite-dimensional space to find a maximum-margin hyperplane with minimum error. |
    | **Computational Simplicity** | **Gaussian Naïve Bayes** | Requires only a single pass $O(N \\cdot D)$ to calculate sample mean and variance per feature with zero iterative gradient descent. |
    | **High-Dimensional Data** | **Support Vector Machine (SVM)** | Maximizes margin boundaries using support vectors, making it naturally immune to the curse of dimensionality. |
    | **Small Dataset Training** | **Naïve Bayes & SVM** | Naïve Bayes possesses low variance (few parameters); SVM generalizes well even with limited support vectors. |
    | **Interpretability / Explainability** | **Decision Tree** | Produces explicit, human-readable if-then decision rules that can be audited by clinical medical experts. |
    """)
    
    st.markdown("---")
    st.subheader("📝 Final Practical Journal Conclusion")
    st.markdown(f"""
    <div class="highlight-card">
        <h4 style="color: #065f46; margin-top: 0;">Practical Assignment 4 – Final Conclusion</h4>
        <p style="white-space: pre-line; line-height: 1.6; color: #1f2937;">{conclusion_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📚 Viva / Oral Exam Quick Reference Q&A")
    with st.expander("Q1: Why must StandardScaler be fitted ONLY on training data?"):
        st.write("""
        Fitting the scaler on the entire dataset would leak the global mean and standard deviation of the test set into the training process. 
        This is called **Data Leakage**, causing over-optimistic performance estimates that fail to generalize in production.
        """)
    with st.expander("Q2: Why does Decision Tree not require feature scaling?"):
        st.write("""
        Decision Trees perform univariate orthogonal splits. A split condition such as `mean area <= 695.5` depends only on the order of values within that single feature, 
        not on its scale relative to other features. Monotonic transformations do not change the split ranking.
        """)
    with st.expander("Q3: What is the conditional independence assumption in Naïve Bayes?"):
        st.write("""
        Naïve Bayes assumes that all features are mutually independent given the class label ($P(X_1, X_2 | Y) = P(X_1 | Y) \\cdot P(X_2 | Y)$). 
        While clinical features like radius, perimeter, and area are strongly correlated, Gaussian Naïve Bayes still produces surprisingly competitive rankings.
        """)
    with st.expander("Q4: How does Log Loss differ from (1 - Accuracy)?"):
        st.write("""
        Accuracy assigns a 1 or 0 binary score regardless of confidence. Log Loss measures cross-entropy between true labels and probability distributions 
        from `predict_proba()`. An incorrect prediction with 99% confidence incurs an extreme penalty under Log Loss, whereas accuracy only counts a single mistake.
        """)
