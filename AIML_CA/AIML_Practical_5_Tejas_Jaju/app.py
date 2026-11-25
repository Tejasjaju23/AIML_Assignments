"""
=============================================================================
AIML Practical Assignment 5: Ensemble Learning and Performance Evaluation
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: app.py - Interactive Streamlit Dashboard
=============================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from dataset import (
    load_raw_dataset,
    get_dataset_dataframe,
    get_dataset_summary,
    prepare_data
)
from models import get_all_models
from evaluation import (
    evaluate_all_models,
    build_comparison_dataframe,
    get_formatted_comparison_table,
    identify_best_model
)
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_accuracy_comparison,
    plot_metrics_comparison,
    plot_log_loss_comparison,
    plot_overall_model_comparison
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ensemble Learning & Performance Evaluation | Tejas Jaju",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styling */
    .main {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-box h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .header-box p {
        color: #e0e6ed;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    /* Student Details Pill */
    .student-pill {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.18);
        padding: 5px 14px;
        border-radius: 20px;
        margin-right: 8px;
        margin-top: 8px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    }
    .kpi-title {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #1e293b;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .kpi-sub {
        color: #10b981;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    /* Best Model Banner */
    .best-badge {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 10px;
        font-weight: 600;
        margin: 15px 0 25px 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    
    /* Info Callouts */
    .info-callout {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 14px 18px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 20px;
    }
    .warning-callout {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 20px;
    }
    
    /* Sidebar Details Card */
    .sidebar-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-top: 15px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching Data & Model Executions
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_dataset():
    df, feat_names, target_names = get_dataset_dataframe()
    summary = get_dataset_summary()
    X_train, X_test, y_train, y_test, feats, targets = prepare_data(test_size=0.2, random_state=42)
    return {
        'df': df,
        'feature_names': feat_names,
        'target_names': target_names,
        'summary': summary,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@st.cache_resource(show_spinner="Training and evaluating all 9 models...")
def get_cached_results():
    data = get_cached_dataset()
    models_dict = get_all_models(random_state=42)
    results = evaluate_all_models(
        models_dict=models_dict,
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        target_names=data['target_names']
    )
    df_comparison = build_comparison_dataframe(results)
    best_model = identify_best_model(df_comparison)
    return results, df_comparison, best_model


# Load cached assets
data_pkg = get_cached_dataset()
results, df_comparison, best_model = get_cached_results()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Student Credentials
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page / Section:",
        [
            "📊 Dashboard",
            "🔍 Dataset Exploration",
            "🤖 Individual Models",
            "⚡ Ensemble Models",
            "📈 Performance Comparison",
            "💡 Conclusion & Insights",
            "🧪 Interactive Model Playground"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎓 Student Profile")
    st.markdown("""
    <div class="sidebar-card">
        <strong>Name:</strong> Tejas Gopal Jaju<br>
        <strong>PRN:</strong> 202401040124<br>
        <strong>Roll No.:</strong> 29<br>
        <strong>Batch:</strong> A2<br>
        <strong>Subject:</strong> AIML<br>
        <strong>Practical:</strong> Assignment 5
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Quick Settings")
    st.info("Stratified 80:20 Split\nRandom State: 42\nTotal Models: 9")


# -----------------------------------------------------------------------------
# Universal Top Header Banner
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="header-box">
    <h1>Ensemble Learning & Performance Evaluation</h1>
    <p>AIML Practical Assignment 5 &bull; Comparative Analysis of Stacking, AdaBoost, Voting Ensembles, and Base Classifiers</p>
    <div>
        <span class="student-pill">👤 Student: Tejas Gopal Jaju</span>
        <span class="student-pill">🆔 PRN: 202401040124</span>
        <span class="student-pill">🔢 Roll: 29 (Batch A2)</span>
        <span class="student-pill">📚 Subject: AIML</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == "📊 Dashboard":
    st.subheader("Executive Dashboard")
    st.markdown(
        "This interactive dashboard presents the complete experimental evaluation of "
        "**Ensemble Learning Techniques** (Stacking, AdaBoost, Hard Voting, Soft Voting) "
        "versus **Individual Base Classifiers** (Logistic Regression, Decision Tree, KNN, SVM, Random Forest) "
        "on the **Breast Cancer Wisconsin (Diagnostic)** benchmark."
    )
    
    # KPI Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Models Evaluated</div>
            <div class="kpi-value">{len(results)}</div>
            <div class="kpi-sub">5 Base + 4 Ensembles</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Top Benchmark Accuracy</div>
            <div class="kpi-value">{best_model['accuracy']*100:.2f}%</div>
            <div class="kpi-sub">Test Split (N=114)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Best Performing Model</div>
            <div class="kpi-value" style="font-size:1.35rem; margin-top:6px;">{best_model['model_name']}</div>
            <div class="kpi-sub">F1: {best_model['f1_score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Dataset Scale</div>
            <div class="kpi-value">{data_pkg['summary']['n_samples']}</div>
            <div class="kpi-sub">30 Real-Valued Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Best Model Callout
    st.markdown(f"""
    <div class="best-badge">
        🏆 <strong>Benchmark Champion:</strong> {best_model['model_name']} ({best_model['model_type']}) 
        achieved the top classification accuracy of <strong>{best_model['accuracy']*100:.2f}%</strong> 
        with an F1-score of <strong>{best_model['f1_score']:.4f}</strong> and cross-entropy Log Loss of <strong>{best_model['log_loss']:.4f}</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    # Two Column Layout: Quick Comparison Chart & Top Models Table
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.markdown("##### 📊 Accuracy Leaderboard")
        fig_acc = plot_accuracy_comparison(df_comparison)
        st.pyplot(fig_acc, use_container_width=True)
        plt.close(fig_acc)
        
    with col_right:
        st.markdown("##### 🥇 Model Ranking Summary")
        df_fmt = get_formatted_comparison_table(df_comparison)
        st.dataframe(
            df_fmt[['Model', 'Accuracy', 'F1-Score', 'Log Loss', 'Type']],
            use_container_width=True,
            height=380
        )
        
    st.markdown("---")
    st.markdown("### 📌 Assignment Objectives & Architecture Highlights")
    ob1, ob2, ob3 = st.columns(3)
    with ob1:
        st.markdown("""
        **1. Stacking Ensemble (Meta-Learning)**
        - **Level 0 Base Learners:** Decision Tree, KNN, SVM, Logistic Regression.
        - **Level 1 Meta-Classifier:** Logistic Regression.
        - **CV Strategy:** 5-fold internal cross-validation generating out-of-fold meta-features.
        """)
    with ob2:
        st.markdown("""
        **2. AdaBoost Ensemble (Boosting)**
        - **Weak Learner:** Decision Tree Stump (`max_depth=1`).
        - **Estimators:** 50 sequential boosting iterations.
        - **Mechanism:** Adaptive sample re-weighting based on exponential loss minimization.
        """)
    with ob3:
        st.markdown("""
        **3. Voting Ensemble (Aggregation)**
        - **Base Classifiers:** Logistic Regression, Decision Tree, KNN, Random Forest.
        - **Hard Voting:** Plurality majority vote across predicted class labels.
        - **Soft Voting:** Probability-weighted averaging across class vectors.
        """)


# =============================================================================
# PAGE 2: DATASET EXPLORATION
# =============================================================================
elif page == "🔍 Dataset Exploration":
    st.subheader("Dataset Exploration & Preprocessing Pipeline")
    
    summary = data_pkg['summary']
    df = data_pkg['df']
    
    st.markdown("""
    The **Breast Cancer Wisconsin (Diagnostic)** dataset is a classic binary classification benchmark 
    computed from digitized images of fine needle aspirate (FNA) of breast masses. Features describe 
    geometric and statistical characteristics of cell nuclei.
    """)
    
    # Metadata Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Instances (N)", summary['n_samples'])
    m2.metric("Total Features (p)", summary['n_features'])
    m3.metric("Missing Values", f"{summary['total_missing_values']} (0%)")
    m4.metric("Train / Test Split", "455 (80%) / 114 (20%)")
    
    st.markdown("---")
    
    # Class Distribution Visualization
    st.markdown("#### 🎯 Target Class Distribution")
    fig_dist = plot_class_distribution(df['target'], data_pkg['target_names'])
    st.pyplot(fig_dist, use_container_width=True)
    plt.close(fig_dist)
    
    st.markdown("---")
    
    # Dataset Preview & Statistics Tabs
    tab_preview, tab_stats, tab_missing = st.tabs(["📋 Dataset Preview", "📊 Descriptive Statistics", "🔎 Missing Value Audit"])
    
    with tab_preview:
        st.markdown("##### First 10 Rows of Dataset:")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption("Target 0 indicates Malignant; Target 1 indicates Benign.")
        
    with tab_stats:
        st.markdown("##### Feature Summary Statistics (Transposed):")
        st.dataframe(summary['statistical_summary'], use_container_width=True)
        
    with tab_missing:
        st.markdown("##### Missing Value Analysis per Column:")
        missing_df = pd.DataFrame({
            'Feature': list(summary['missing_per_feature'].keys()),
            'Missing Count': list(summary['missing_per_feature'].values()),
            'Missing Percentage': [0.0] * len(summary['missing_per_feature'])
        })
        st.dataframe(missing_df, use_container_width=True, height=280)
        st.success("✅ Clean dataset: 0 missing, null, or NaN values detected across all 30 features.")

    st.markdown("---")
    st.markdown("#### 🛡️ Data Leakage Prevention Architecture")
    st.markdown(r"""
    <div class="info-callout">
        <strong>StandardScaler Pipeline Integration:</strong><br>
        To prevent subtle <em>data leakage</em>, feature normalization (StandardScaler) is strictly contained 
        within Scikit-Learn <code>Pipeline</code> objects. The mean \(\mu\) and standard deviation \(\sigma\) are computed 
        <strong>strictly on the training split</strong> (<code>X_train</code>) during <code>fit()</code>, and identically applied 
        to testing instances (<code>X_test</code>) during <code>transform()</code>.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: INDIVIDUAL BASE CLASSIFIERS
# =============================================================================
elif page == "🤖 Individual Models":
    st.subheader("Individual Base Classifiers Evaluation")
    st.markdown("Inspect the architecture, mathematical principles, metrics, and confusion matrix for each base model:")
    
    base_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'SVM', 'Random Forest']
    selected_base = st.selectbox("Select Individual Base Model:", base_names, index=0)
    
    res = results[selected_base]
    
    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
    m2.metric("Weighted Precision", f"{res['precision']:.4f}")
    m3.metric("Weighted Recall", f"{res['recall']:.4f}")
    m4.metric("Weighted F1-Score", f"{res['f1_score']:.4f}")
    loss_disp = f"{res['log_loss']:.4f}" if res['log_loss'] is not None else "N/A"
    m5.metric("Log Loss", loss_disp)
    
    st.markdown("---")
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown(f"#### 📐 Confusion Matrix: {selected_base}")
        fig_cm = plot_confusion_matrix(res['confusion_matrix'], selected_base, data_pkg['target_names'])
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)
        
    with col_r:
        st.markdown("#### 📄 Classification Report")
        st.code(res['classification_report_text'], language="text")
        
        st.markdown("#### ⚙️ Model Architecture & Theory")
        st.markdown(f"**Pipeline / Configuration:** `{res['architecture']}`")
        st.markdown(f"**Description:** {res['description']}")
        st.markdown(f"**Mathematical Formulation:** {res['theory']}")


# =============================================================================
# PAGE 4: ENSEMBLE MODELS
# =============================================================================
elif page == "⚡ Ensemble Models":
    st.subheader("Ensemble Learning Architecture & In-Depth Analysis")
    
    ens_tabs = st.tabs([
        "🧱 Stacking Ensemble",
        "🚀 AdaBoost Ensemble",
        "🗳️ Hard Voting Ensemble",
        "🌊 Soft Voting Ensemble"
    ])
    
    # Tab 1: Stacking
    with ens_tabs[0]:
        ens_res = results['Stacking Classifier']
        st.markdown("### 🧱 Stacking Classifier (Meta-Learning)")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{ens_res['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{ens_res['precision']:.4f}")
        m3.metric("Recall", f"{ens_res['recall']:.4f}")
        m4.metric("F1-Score", f"{ens_res['f1_score']:.4f}")
        m5.metric("Log Loss", f"{ens_res['log_loss']:.4f}")
        
        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_cm = plot_confusion_matrix(ens_res['confusion_matrix'], "Stacking Classifier", data_pkg['target_names'])
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)
        with c2:
            st.markdown("##### 📄 Classification Report")
            st.code(ens_res['classification_report_text'], language="text")
            
        st.markdown("#### 🔬 How the Stacking Meta-Classifier Combines Predictions")
        st.markdown(r"""
        Stacking is an advanced meta-learning ensemble paradigm:
        1. **Level-0 Base Estimators:** Diverse models with distinct mathematical formulations:
           - *Decision Tree* (Orthogonal axis-aligned recursive partitioning)
           - *KNN* (Local distance-based manifold neighborhood)
           - *SVM* (Infinite-dimensional RBF margin maximization)
           - *Logistic Regression* (Sigmoidal linear probabilistic calibration)
        2. **Internal 5-Fold Cross-Validation:** To prevent meta-feature leakage, the training dataset 
           is split into 5 internal folds. Each base estimator is fit on 4 folds and predicts the holdout fold.
        3. **Meta-Feature Construction:** The out-of-fold predictions concatenate into a meta-matrix \(Z \in \mathbb{R}^{N \times 4}\).
        4. **Level-1 Meta-Classifier:** A *Logistic Regression* classifier is trained on \(Z\). It learns 
           optimal blending weights \(\beta_m\), assigning higher authority to whichever base classifier performs 
           best across different regions of feature space.
        """)
        
    # Tab 2: AdaBoost
    with ens_tabs[1]:
        ens_res = results['AdaBoost Classifier']
        st.markdown("### 🚀 AdaBoost Classifier (Adaptive Boosting)")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{ens_res['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{ens_res['precision']:.4f}")
        m3.metric("Recall", f"{ens_res['recall']:.4f}")
        m4.metric("F1-Score", f"{ens_res['f1_score']:.4f}")
        m5.metric("Log Loss", f"{ens_res['log_loss']:.4f}")
        
        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_cm = plot_confusion_matrix(ens_res['confusion_matrix'], "AdaBoost Classifier", data_pkg['target_names'])
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)
        with c2:
            st.markdown("##### 📄 Classification Report")
            st.code(ens_res['classification_report_text'], language="text")
            
        st.markdown("#### 🔄 How AdaBoost Adjusts Sample Weights for Misclassified Samples")
        st.markdown(r"""
        AdaBoost converts weak learners (Decision Stumps with `max_depth=1`) into a strong classifier through sequential re-weighting:
        - **Initialization:** Every training sample receives equal weight: \(w_i^{(1)} = \frac{1}{N}\).
        - **Iterative Round \(m = 1 \dots 50\):**
          1. Fit decision stump \(h_m(x)\) minimizing weighted training error \(\epsilon_m = \sum_{i=1}^N w_i^{(m)} \mathbb{I}(y_i \neq h_m(x_i))\).
          2. Compute estimator stage weight: \(\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)\).
          3. **Update Sample Weights:**
             $$w_i^{(m+1)} = w_i^{(m)} \exp\left(-\alpha_m y_i h_m(x_i)\right)$$
          4. Normalize weights such that \(\sum w_i^{(m+1)} = 1\).
        - **Core Intuition:** Misclassified instances receive multiplied weights, compelling round \(m+1\)'s stump 
          to prioritize hard-to-classify borderline cases.
        """)

    # Tab 3: Hard Voting
    with ens_tabs[2]:
        ens_res = results['Hard Voting Classifier']
        st.markdown("### 🗳️ Hard Voting Classifier (Majority Rule)")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{ens_res['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{ens_res['precision']:.4f}")
        m3.metric("Recall", f"{ens_res['recall']:.4f}")
        m4.metric("F1-Score", f"{ens_res['f1_score']:.4f}")
        m5.metric("Log Loss", "N/A (Discrete Votes)")
        
        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_cm = plot_confusion_matrix(ens_res['confusion_matrix'], "Hard Voting Classifier", data_pkg['target_names'])
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)
        with c2:
            st.markdown("##### 📄 Classification Report")
            st.code(ens_res['classification_report_text'], language="text")
            
        st.markdown("#### 📊 Working Mechanism of Hard Voting")
        st.markdown(r"""
        Hard Voting applies a strict majority democratic voting rule across 4 base models:
        - **Base Models:** Logistic Regression, Decision Tree, KNN, Random Forest.
        - **Decision Rule:**
          $$\hat{y} = \text{mode}\left(\hat{y}_1, \hat{y}_2, \hat{y}_3, \hat{y}_4\right)$$
        - **Characteristics:**
          - Simple and computationally fast.
          - Invariant to uncalibrated classifier confidence scores.
          - **Limitation:** Does not produce class probability vectors, which is why Cross-Entropy Log Loss cannot be computed.
        """)

    # Tab 4: Soft Voting
    with ens_tabs[3]:
        ens_res = results['Soft Voting Classifier']
        st.markdown("### 🌊 Soft Voting Classifier (Probability Averaging)")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{ens_res['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{ens_res['precision']:.4f}")
        m3.metric("Recall", f"{ens_res['recall']:.4f}")
        m4.metric("F1-Score", f"{ens_res['f1_score']:.4f}")
        m5.metric("Log Loss", f"{ens_res['log_loss']:.4f}")
        
        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_cm = plot_confusion_matrix(ens_res['confusion_matrix'], "Soft Voting Classifier", data_pkg['target_names'])
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)
        with c2:
            st.markdown("##### 📄 Classification Report")
            st.code(ens_res['classification_report_text'], language="text")
            
        st.markdown("#### ⚖️ Difference Between Hard Voting and Soft Voting")
        st.markdown(r"""
        | Feature | Hard Voting | Soft Voting |
        |---|---|---|
        | **Input Signal** | Discrete class labels \(\hat{y} \in \{0, 1\}\) | Continuous posterior probabilities \(P(y=c\|x)\) |
        | **Decision Formula** | \(\hat{y} = \text{mode}(\hat{y}_1, \dots, \hat{y}_M)\) | \(\hat{y} = \text{argmax}_c \frac{1}{M}\sum_{m=1}^M P_m(y=c\|x)\) |
        | **Confidence Sensitivity** | Discards classifier confidence | Highly confident predictions carry proportional weight |
        | **Probability Support** | No (Log Loss undefined) | Yes (Calculates Log Loss = 0.1003) |
        | **Empirical Advantage** | Equal weight to weak/strong calls | Resolves edge ties through probability gradient |
        """)


# =============================================================================
# PAGE 5: PERFORMANCE COMPARISON
# =============================================================================
elif page == "📈 Performance Comparison":
    st.subheader("Comprehensive Model Performance Comparison")
    st.markdown(
        "Side-by-side evaluation of all 9 models on the identical 20% test partition. "
        "All metrics are calculated strictly from actual out-of-sample model predictions."
    )
    
    # Formatted Comparison Table
    st.markdown("#### 📋 Complete Performance Comparison Table")
    df_fmt = get_formatted_comparison_table(df_comparison)
    
    # Highlighting styling for dataframe
    st.dataframe(
        df_fmt[['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Log Loss', 'Train Time (s)']],
        use_container_width=True,
        height=380
    )
    
    # Download CSV Button
    csv_bytes = df_comparison.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Comparison CSV (model_comparison.csv)",
        data=csv_bytes,
        file_name="model_comparison.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Visualizations Row 1: Accuracy & Metrics
    st.markdown("#### 📊 Comparative Metric Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig_acc = plot_accuracy_comparison(df_comparison)
        st.pyplot(fig_acc, use_container_width=True)
        plt.close(fig_acc)
    with col2:
        fig_met = plot_metrics_comparison(df_comparison)
        st.pyplot(fig_met, use_container_width=True)
        plt.close(fig_met)
        
    st.markdown("---")
    
    # Visualizations Row 2: Log Loss & 4-Panel Overview
    col3, col4 = st.columns(2)
    with col3:
        fig_loss = plot_log_loss_comparison(df_comparison)
        st.pyplot(fig_loss, use_container_width=True)
        plt.close(fig_loss)
    with col4:
        st.markdown("##### 📌 Evaluation Takeaways:")
        st.markdown(f"""
        - **Linear & Kernel Models Lead:** Standalone Logistic Regression and SVM achieved top test accuracy ({best_model['accuracy']*100:.2f}%) due to well-scaled high-dimensional continuous features.
        - **Ensemble Robustness:** Stacking, Soft Voting, and Hard Voting all attained stable 97.37% accuracy, shielding against tree instability.
        - **Soft Voting Advantage:** Outperformed hard voting in probability estimation with a low log loss of 0.1003.
        - **Decision Tree Underperformance:** Standalone Decision Tree scored lowest (92.11% accuracy, log loss 1.9708), reflecting high variance on continuous data.
        """)
        
    st.markdown("---")
    st.markdown("#### 🎯 Overall 4-Panel Comparison Matrix")
    fig_overall = plot_overall_model_comparison(df_comparison)
    st.pyplot(fig_overall, use_container_width=True)
    plt.close(fig_overall)


# =============================================================================
# PAGE 6: CONCLUSION AND INSIGHTS
# =============================================================================
elif page == "💡 Conclusion & Insights":
    st.subheader("Conclusion, Theoretical Findings, and Future Scope")
    
    st.markdown("""
    ### 📝 Comprehensive Assignment Findings
    1. **Base vs. Ensemble Comparison:**
       - Individual models exhibited varying variance: Decision Tree showed noticeable variance (92.11%), whereas regularized linear models (Logistic Regression, 98.25%) and margin-maximizing kernels (SVM, 98.25%) excelled.
       - Ensemble methods (Stacking, Hard Voting, Soft Voting) provided strong stability (97.37%), effectively dampening individual model variance.
    2. **Probability vs. Discrete Voting:**
       - Soft Voting provided richer decision boundaries than Hard Voting by taking advantage of classifier confidence levels.
    """)
    
    st.markdown("---")
    st.markdown("### 🧭 Strategic Guide: When to Use Each Ensemble Technique")
    
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div class="kpi-card" style="text-align: left;">
            <h4 style="color:#1d3557;">🧱 Stacking Ensemble</h4>
            <strong>When to Use:</strong>
            <ul>
                <li>When base classifiers come from distinct model families (e.g. tree, kernel, linear).</li>
                <li>When individual models make errors in different regions of the feature space.</li>
                <li>When you have sufficient data to cross-validate Level-0 meta-features.</li>
            </ul>
            <strong>Pros:</strong> Learns non-linear combination weights; high expressiveness.<br>
            <strong>Cons:</strong> Slower training due to internal cross-validation; risk of meta-overfitting.
        </div>
        """, unsafe_allow_html=True)
        
    with g2:
        st.markdown("""
        <div class="kpi-card" style="text-align: left;">
            <h4 style="color:#e63946;">🚀 AdaBoost Ensemble</h4>
            <strong>When to Use:</strong>
            <ul>
                <li>When base models are high-bias weak learners (e.g., decision stumps).</li>
                <li>When dataset has clean labels and minimal outlier contamination.</li>
                <li>When you want a lightweight sequential booster.</li>
            </ul>
            <strong>Pros:</strong> Converts weak stumps into strong classifiers; minimal tuning.<br>
            <strong>Cons:</strong> Vulnerable to noisy outliers (exponential penalty overweights noise).
        </div>
        """, unsafe_allow_html=True)
        
    with g3:
        st.markdown("""
        <div class="kpi-card" style="text-align: left;">
            <h4 style="color:#2a9d8f;">🗳️ Voting Ensemble</h4>
            <strong>When to Use:</strong>
            <ul>
                <li>When multiple well-tuned diverse models achieve comparable accuracy.</li>
                <li>When fast inference and training are required without meta-learner overhead.</li>
                <li>Prefer Soft Voting whenever well-calibrated probabilities are available.</li>
            </ul>
            <strong>Pros:</strong> Low computational cost; smooths decision boundaries.<br>
            <strong>Cons:</strong> Cannot overcome systemic correlated errors if base models are too similar.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 🔭 Limitations and Future Scope")
    st.markdown("""
    - **Hyperparameter Optimization:** All models were trained with standard baseline hyperparameters. Future work can implement Bayesian Search (`Optuna`) or Grid Search for hyperparameter tuning.
    - **Advanced Gradient Boosters:** Extend the ensemble study to modern gradient boosting libraries: **XGBoost**, **LightGBM**, and **CatBoost**.
    - **Feature Selection & Dimensionality Reduction:** Employ Recursive Feature Elimination (RFE) or Principal Component Analysis (PCA) to remove multicollinear features among the 30 continuous predictors.
    - **Calibration:** Apply isotonic regression or Platt scaling to optimize probability calibration curves before soft voting aggregation.
    """)


# =============================================================================
# PAGE 7: INTERACTIVE MODEL PLAYGROUND (BONUS PRACTICAL FEATURE)
# =============================================================================
elif page == "🧪 Interactive Model Playground":
    st.subheader("Interactive Model Playground (Live Inference Test)")
    st.markdown(
        "Select a test sample from the holdout dataset (20% test partition) to test and compare "
        "live predictions and confidence probabilities across **all 9 models side-by-side**."
    )
    
    X_test = data_pkg['X_test']
    y_test = data_pkg['y_test']
    target_names = data_pkg['target_names']
    
    sample_idx = st.slider(
        "Select Test Sample Index:",
        min_value=0,
        max_value=len(X_test) - 1,
        value=12,
        step=1
    )
    
    sample_features = X_test.iloc[sample_idx:sample_idx+1]
    actual_label = y_test.iloc[sample_idx]
    actual_name = target_names[actual_label].capitalize()
    
    # Ground truth pill
    st.markdown(f"""
    <div style="background-color:#e0f2fe; border-left:5px solid #0284c7; padding:12px 16px; border-radius:6px; margin-bottom:15px;">
        <strong>Test Sample #{sample_idx}:</strong> Actual Ground Truth is 
        <span style="font-weight:700; color:{'#dc2626' if actual_name == 'Malignant' else '#16a34a'};">{actual_name}</span>.
    </div>
    """, unsafe_allow_html=True)
    
    # Live Predictions Across All 9 Models
    pred_rows = []
    for name, res in results.items():
        fitted = res['fitted_model']
        p_label = fitted.predict(sample_features)[0]
        p_name = target_names[p_label].capitalize()
        
        prob_str = "N/A"
        if hasattr(fitted, "predict_proba"):
            try:
                probs = fitted.predict_proba(sample_features)[0]
                prob_str = f"Malignant: {probs[0]:.2%}, Benign: {probs[1]:.2%}"
            except Exception:
                prob_str = "N/A"
                
        is_correct = (p_label == actual_label)
        pred_rows.append({
            'Model': name,
            'Model Category': res['model_type'],
            'Predicted Diagnosis': p_name,
            'Confidence Probabilities': prob_str,
            'Verdict': '✅ Correct' if is_correct else '❌ Incorrect'
        })
        
    df_live = pd.DataFrame(pred_rows)
    st.dataframe(df_live, use_container_width=True)
    
    st.markdown("##### Selected Sample Feature Values (First 10 Features):")
    st.dataframe(sample_features.iloc[:, :10], use_container_width=True)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 10px 0;">
    AIML Practical Assignment 5 &bull; Submitted by <strong>Tejas Gopal Jaju</strong> (PRN: 202401040124, Roll: 29, Batch: A2) &bull; Department of Artificial Intelligence & Machine Learning
</div>
""", unsafe_allow_html=True)
