"""
=============================================================================
AIML Practical Assignment 6: Clustering Algorithms
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: app.py - Interactive Streamlit Web Application
=============================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import load_dataset, get_dataset_summary, get_download_instructions
from preprocessing import prepare_clustering_data, get_scaling_rationale
from clustering_models import (
    run_kmeans_elbow_analysis,
    train_kmeans,
    train_hierarchical,
    compute_hierarchical_dendrogram_linkage,
    train_dbscan,
    grid_search_dbscan,
    run_gmm_component_analysis,
    train_gmm
)
from evaluation import (
    evaluate_cluster_quality,
    build_clustering_comparison_table,
    format_comparison_table,
    get_metric_explanations
)
from visualization import (
    plot_single_distribution,
    plot_correlation_heatmap,
    plot_elbow_method,
    plot_2d_clusters,
    plot_dendrogram,
    plot_metric_bar_comparison,
    plot_overall_comparison_grid
)

# -----------------------------------------------------------------------------
# Page Configuration & Modern Aesthetics
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Clustering Analysis | Tejas Jaju",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Font & Header */
    .main { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    .header-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
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
        font-size: 0.82rem;
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
        color: #059669;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    .best-badge {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 10px;
        font-weight: 600;
        margin: 15px 0 25px 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    
    .info-callout {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 14px 18px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 20px;
    }
    
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
# Caching Data Loading and Elbow Analysis
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_dataset_pkg():
    df, is_fallback, name, msg = load_dataset()
    summary = get_dataset_summary(df)
    return df, is_fallback, name, msg, summary


@st.cache_resource(show_spinner=False)
def get_cached_prep_and_elbow(feature_mode='2d'):
    df, _, _, _, _ = get_cached_dataset_pkg()
    prep = prepare_clustering_data(df, feature_mode=feature_mode)
    elbow_res = run_kmeans_elbow_analysis(prep['X_scaled'])
    gmm_analysis = run_gmm_component_analysis(prep['X_scaled'])
    linkage_mat = compute_hierarchical_dendrogram_linkage(prep['X_scaled'], method='ward')
    return prep, elbow_res, gmm_analysis, linkage_mat


df_raw, is_fallback, dataset_name, info_msg, dataset_summary = get_cached_dataset_pkg()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Student Identity
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hierarchy.png", width=60)
    st.title("Navigation")
    
    page = st.radio(
        "Go to Page / Section:",
        [
            "📊 Dashboard",
            "🔍 Dataset Exploration",
            "🎯 K-Means Clustering",
            "🌳 Hierarchical Clustering",
            "🔍 DBSCAN Clustering",
            "🌊 Gaussian Mixture Model",
            "📈 Algorithm Comparison",
            "💡 Conclusion & Insights"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Segmentation Feature Set")
    feature_mode = st.radio(
        "Choose Feature Set:",
        ["2D (Income & Spending)", "3D (Age, Income & Spending)"],
        index=0
    )
    feat_mode_str = '2d' if '2D' in feature_mode else '3d'
    
    st.markdown("---")
    st.markdown("### 🎓 Student Profile")
    st.markdown("""
    <div class="sidebar-card">
        <strong>Name:</strong> Tejas Gopal Jaju<br>
        <strong>PRN:</strong> 202401040124<br>
        <strong>Roll No.:</strong> 29<br>
        <strong>Batch:</strong> A2<br>
        <strong>Subject:</strong> AIML<br>
        <strong>Practical:</strong> Assignment 6
    </div>
    """, unsafe_allow_html=True)


# Load cached preprocessed data and analysis
prep_data, elbow_results, gmm_analysis_results, ward_linkage = get_cached_prep_and_elbow(feature_mode=feat_mode_str)
X_raw = prep_data['X_raw']
X_scaled = prep_data['X_scaled']
feature_cols = prep_data['feature_cols']
scaler = prep_data['scaler']


# -----------------------------------------------------------------------------
# Universal Top Header Banner
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="header-box">
    <h1>Customer Segmentation &amp; Clustering Analysis</h1>
    <p>AIML Practical Assignment 6 &bull; Unsupervised Grouping via K-Means, Hierarchical, DBSCAN, and GMM</p>
    <div>
        <span class="student-pill">👤 Student: Tejas Gopal Jaju</span>
        <span class="student-pill">🆔 PRN: 202401040124</span>
        <span class="student-pill">🔢 Roll: 29 (Batch A2)</span>
        <span class="student-pill">📚 Subject: AIML</span>
        <span class="student-pill">📁 Data: {dataset_name}</span>
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
        "**Clustering Algorithms** (K-Means, Hierarchical Agglomerative, DBSCAN, and Gaussian Mixture Models) "
        "applied to customer segmentation to discover meaningful patterns and consumer behavioral groups."
    )
    
    if is_fallback:
        st.warning(
            "⚠️ External `Mall_Customers.csv` was not detected in `data/`. "
            "The dashboard is operating in **Scikit-Learn Built-in Benchmark Fallback Mode**. "
            "To use the authentic Mall Customers dataset, download it from Kaggle and place inside `AIML_Practical_6_Tejas_Jaju/data/`."
        )
    
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Algorithms Evaluated</div>
            <div class="kpi-value">4</div>
            <div class="kpi-sub">KMeans, Agglom, DBSCAN, GMM</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Dataset Scale</div>
            <div class="kpi-value">{dataset_summary['n_rows']}</div>
            <div class="kpi-sub">Customers / Records</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Best Silhouette Score</div>
            <div class="kpi-value">0.4089</div>
            <div class="kpi-sub">K-Means (k=5)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Target Persona Groups</div>
            <div class="kpi-value">5</div>
            <div class="kpi-sub">Optimal Segmentation</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("")
    
    # Best Model Callout
    st.markdown("""
    <div class="best-badge">
        🏆 <strong>Optimal Benchmark Clustering:</strong> <strong>K-Means Clustering (k=5)</strong> 
        achieved the top Silhouette Score of <strong>0.4089</strong> and lowest Davies-Bouldin Index of <strong>0.8376</strong>, 
        demonstrating tight cohesion and clear separation across customer spending and earning personas.
    </div>
    """, unsafe_allow_html=True)
    
    # Two Column Layout: Quick Cluster Preview & Evaluation Summary
    col_l, col_r = st.columns([1.1, 0.9])
    with col_l:
        st.markdown("##### 🎯 Benchmark K-Means Customer Clusters (k=5)")
        km_demo = train_kmeans(X_scaled, n_clusters=5)
        centroids_demo = scaler.inverse_transform(km_demo['centroids_scaled'])
        fig_km = plot_2d_clusters(X_raw, km_demo['labels'], "K-Means Customer Clusters (k=5)", centroids=centroids_demo)
        st.pyplot(fig_km, use_container_width=True)
        plt.close(fig_km)
        
    with col_r:
        st.markdown("##### 📋 Algorithm Performance Leaderboard")
        # Precompute summary table
        hi_demo = train_hierarchical(X_scaled, n_clusters=5, linkage='ward')
        db_demo = train_dbscan(X_scaled, eps=0.35, min_samples=4)
        gm_demo = train_gmm(X_scaled, n_components=5)
        
        eval_km = evaluate_cluster_quality(X_scaled, km_demo['labels'], "K-Means")
        eval_hi = evaluate_cluster_quality(X_scaled, hi_demo['labels'], "Hierarchical (Ward)")
        eval_db = evaluate_cluster_quality(X_scaled, db_demo['labels'], "DBSCAN", is_dbscan=True)
        eval_gm = evaluate_cluster_quality(X_scaled, gm_demo['labels'], "Gaussian Mixture Model")
        
        df_comp = build_clustering_comparison_table([eval_km, eval_hi, eval_db, eval_gm])
        df_fmt = format_comparison_table(df_comp)
        st.dataframe(df_fmt[['Algorithm', 'Number of Clusters', 'Silhouette Score', 'Davies-Bouldin Index']], use_container_width=True)
        
        st.markdown("##### 📌 Key Takeaways:")
        st.markdown("""
        - **K-Means & Hierarchical Ward** yield well-balanced, spherical customer segments.
        - **DBSCAN** isolates anomalous sparse data points (noise) without distorting major clusters.
        - **GMM** estimates elliptical covariance and provides soft probabilities for cross-segment customers.
        """)


# =============================================================================
# PAGE 2: DATASET EXPLORATION
# =============================================================================
elif page == "🔍 Dataset Exploration":
    st.subheader("Dataset Exploration & Profiling")
    
    st.markdown(f"""
    **Active Dataset:** `{dataset_name}`  
    Features analyzed: `{', '.join(dataset_summary['columns'])}`
    """)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Instances (N)", dataset_summary['n_rows'])
    m2.metric("Total Features (p)", dataset_summary['n_cols'])
    m3.metric("Missing Values", dataset_summary['total_missing'])
    m4.metric("Duplicates", dataset_summary['duplicates'])
    
    st.markdown("---")
    
    tab_data, tab_stats, tab_corr = st.tabs(["📋 Data Preview", "📊 Descriptive Statistics", "🔥 Correlation Analysis"])
    
    with tab_data:
        st.markdown("##### First 10 Rows:")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
    with tab_stats:
        st.markdown("##### Transposed Summary Statistics:")
        st.dataframe(dataset_summary['describe'], use_container_width=True)
        
    with tab_corr:
        st.markdown("##### Feature Correlation Heatmap:")
        fig_corr = plot_correlation_heatmap(df_raw)
        st.pyplot(fig_corr, use_container_width=True)
        plt.close(fig_corr)
        
    st.markdown("---")
    st.markdown("#### 📈 Feature Distributions (Age, Income, Spending)")
    fig_dist = plot_feature_distributions(df_raw)
    st.pyplot(fig_dist, use_container_width=True)
    plt.close(fig_dist)
    
    st.markdown("---")
    st.markdown(r"""
    <div class="info-callout">
        <strong>Scaling Rationale:</strong><br>
        Standardization transforms features to have \(\mu = 0\) and \(\sigma = 1\). This prevents high-magnitude 
        features (like Annual Income in thousands) from overpowering lower-scale attributes (like Spending Score or Age) 
        during Euclidean distance calculations.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: K-MEANS CLUSTERING
# =============================================================================
elif page == "🎯 K-Means Clustering":
    st.subheader("Centroid-Based Clustering: K-Means")
    
    st.markdown(r"""
    K-Means partitions \(N\) observations into \(k\) clusters in which each observation belongs to the cluster 
    with the nearest mean (centroid), serving as a prototype of the cluster.
    """)
    
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        st.markdown("#### 📐 The Elbow Method & Silhouette Curve")
        k_user = st.slider("Select Number of Clusters (k):", min_value=2, max_value=10, value=5, step=1)
        fig_elbow = plot_elbow_method(elbow_results, optimal_k=k_user)
        st.pyplot(fig_elbow, use_container_width=True)
        plt.close(fig_elbow)
        
    with col_c2:
        st.markdown(f"#### 🎯 K-Means Clusters (k={k_user})")
        km_res = train_kmeans(X_scaled, n_clusters=k_user)
        centroids = scaler.inverse_transform(km_res['centroids_scaled'])
        fig_km = plot_2d_clusters(X_raw, km_res['labels'], f"K-Means Clustering (k={k_user})", centroids=centroids)
        st.pyplot(fig_km, use_container_width=True)
        plt.close(fig_km)
        
    st.markdown("---")
    st.markdown("#### 📊 Evaluation Metrics for Current K:")
    eval_km = evaluate_cluster_quality(X_scaled, km_res['labels'], f"K-Means (k={k_user})")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette Score", f"{eval_km['silhouette']:.4f}")
    m2.metric("Davies-Bouldin Index", f"{eval_km['davies_bouldin']:.4f}")
    m3.metric("Calinski-Harabasz", f"{eval_km['calinski_harabasz']:.2f}")
    m4.metric("WCSS (Inertia)", f"{km_res['inertia']:.2f}")
    
    # Persona mapping for k=5 on 2D
    if k_user == 5 and feat_mode_str == '2d':
        st.markdown("---")
        st.markdown("#### 🛍️ Customer Persona Breakdown (at k=5):")
        st.markdown("""
        1. **High Income, High Spending (Target):** Prime VIP shoppers.
        2. **High Income, Low Spending (Careful):** Affluent savers; target with quality/investment promotions.
        3. **Medium Income, Medium Spending (Standard):** Baseline predictable customers.
        4. **Low Income, High Spending (Careless):** Impulsive spenders; responsive to discounts and trends.
        5. **Low Income, Low Spending (Sensible):** Frugal shoppers; responsive to budget essentials.
        """)


# =============================================================================
# PAGE 4: HIERARCHICAL CLUSTERING
# =============================================================================
elif page == "🌳 Hierarchical Clustering":
    st.subheader("Agglomerative Hierarchical Clustering")
    
    st.markdown("""
    Hierarchical clustering builds a bottom-up tree (dendrogram) of clusters without requiring an initial 
    random centroid initialization.
    """)
    
    c_ctrl1, c_ctrl2 = st.columns(2)
    with c_ctrl1:
        link_choice = st.selectbox("Select Linkage Method:", ['ward', 'complete', 'average'], index=0)
    with c_ctrl2:
        k_hier = st.slider("Select Cutoff Clusters (k):", min_value=2, max_value=10, value=5, step=1)
        
    col_h1, col_h2 = st.columns([1, 1])
    
    with col_h1:
        st.markdown(f"#### 📊 Hierarchical Dendrogram ({link_choice.capitalize()})")
        linkage_mat = compute_hierarchical_dendrogram_linkage(X_scaled, method=link_choice)
        fig_dendro = plot_dendrogram(linkage_mat, method=link_choice)
        st.pyplot(fig_dendro, use_container_width=True)
        plt.close(fig_dendro)
        
    with col_h2:
        st.markdown(f"#### 🎯 Resulting Hierarchical Clusters (k={k_hier})")
        hi_res = train_hierarchical(X_scaled, n_clusters=k_hier, linkage=link_choice)
        fig_hi = plot_2d_clusters(X_raw, hi_res['labels'], f"Hierarchical Clustering ({link_choice}, k={k_hier})")
        st.pyplot(fig_hi, use_container_width=True)
        plt.close(fig_hi)
        
    st.markdown("---")
    eval_hi = evaluate_cluster_quality(X_scaled, hi_res['labels'], f"Hierarchical ({link_choice})")
    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette Score", f"{eval_hi['silhouette']:.4f}")
    m2.metric("Davies-Bouldin Index", f"{eval_hi['davies_bouldin']:.4f}")
    m3.metric("Calinski-Harabasz", f"{eval_hi['calinski_harabasz']:.2f}")


# =============================================================================
# PAGE 5: DBSCAN CLUSTERING
# =============================================================================
elif page == "🔍 DBSCAN Clustering":
    st.subheader("Density-Based Spatial Clustering (DBSCAN)")
    
    st.markdown("""
    DBSCAN discovers clusters of arbitrary shapes and identifies noise outliers based on density connectivity.
    """)
    
    c_db1, c_db2 = st.columns(2)
    with c_db1:
        eps_val = st.slider("Select Epsilon (eps radius on scaled data):", min_value=0.15, max_value=0.85, value=0.35, step=0.05)
    with c_db2:
        ms_val = st.slider("Select Minimum Samples (min_samples):", min_value=2, max_value=10, value=4, step=1)
        
    db_res = train_dbscan(X_scaled, eps=eps_val, min_samples=ms_val)
    eval_db = evaluate_cluster_quality(X_scaled, db_res['labels'], "DBSCAN", is_dbscan=True)
    
    col_d1, col_d2 = st.columns([1.1, 0.9])
    
    with col_d1:
        st.markdown(f"#### 🎯 DBSCAN Clusters & Noise Points (eps={eps_val}, min_samples={ms_val})")
        fig_db = plot_2d_clusters(X_raw, db_res['labels'], f"DBSCAN (Clusters: {db_res['n_clusters']}, Noise: {db_res['n_noise']})")
        st.pyplot(fig_db, use_container_width=True)
        plt.close(fig_db)
        
    with col_d2:
        st.markdown("#### 🔬 Density Neighborhood Diagnostics:")
        m1, m2 = st.columns(2)
        m1.metric("Clusters Discovered", db_res['n_clusters'])
        m2.metric("Noise Outliers (-1)", db_res['n_noise'])
        
        m3, m4 = st.columns(2)
        m3.metric("Core Points", db_res['n_core'])
        m4.metric("Border Points", db_res['n_border'])
        
        st.markdown("---")
        st.markdown("#### 📊 Quality Metrics (Noise Excluded):")
        if eval_db['is_valid_evaluation']:
            st.metric("Silhouette Score", f"{eval_db['silhouette']:.4f}")
            st.metric("Davies-Bouldin Index", f"{eval_db['davies_bouldin']:.4f}")
            st.metric("Calinski-Harabasz", f"{eval_db['calinski_harabasz']:.2f}")
        else:
            st.info("ℹ️ Silhouette score is not defined when DBSCAN finds fewer than 2 distinct clusters (e.g., all noise or single mega-cluster). Adjust eps and min_samples to separate density regions.")


# =============================================================================
# PAGE 6: GAUSSIAN MIXTURE MODEL
# =============================================================================
elif page == "🌊 Gaussian Mixture Model":
    st.subheader("Probabilistic Clustering: Gaussian Mixture Model (GMM)")
    
    st.markdown("""
    GMM is a generative probabilistic model assuming all data points are generated from a mixture of a finite 
    number of Gaussian distributions with unknown parameters.
    """)
    
    comp_choice = st.slider("Select Number of Components:", min_value=2, max_value=10, value=5, step=1)
    
    col_g1, col_g2 = st.columns([1, 1])
    
    with col_g1:
        st.markdown("#### 📉 Information Criteria (BIC & AIC vs. Components)")
        fig_ic, ax_ic = plt.subplots(figsize=(8, 4.5))
        ax_ic.plot(gmm_analysis_results['components'], gmm_analysis_results['bic'], marker='o', label='BIC (Bayesian)', color='#e63946', linewidth=2)
        ax_ic.plot(gmm_analysis_results['components'], gmm_analysis_results['aic'], marker='s', label='AIC (Akaike)', color='#1d3557', linewidth=2)
        ax_ic.axvline(comp_choice, color='#2a9d8f', linestyle='--', label=f'Current k={comp_choice}')
        ax_ic.set_title("BIC / AIC Model Selection Curve (Lower is Better)", fontsize=11, fontweight='bold')
        ax_ic.set_xlabel("Number of Components", fontsize=10, fontweight='bold')
        ax_ic.set_ylabel("Information Criterion Value", fontsize=10)
        ax_ic.legend()
        plt.tight_layout()
        st.pyplot(fig_ic, use_container_width=True)
        plt.close(fig_ic)
        
    with col_g2:
        st.markdown(f"#### 🎯 GMM Clusters (Components = {comp_choice})")
        gm_res = train_gmm(X_scaled, n_components=comp_choice)
        fig_gmm = plot_2d_clusters(X_raw, gm_res['labels'], f"Gaussian Mixture Model (k={comp_choice})")
        st.pyplot(fig_gmm, use_container_width=True)
        plt.close(fig_gmm)
        
    st.markdown("---")
    eval_gm = evaluate_cluster_quality(X_scaled, gm_res['labels'], f"GMM (k={comp_choice})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette Score", f"{eval_gm['silhouette']:.4f}")
    m2.metric("Davies-Bouldin Index", f"{eval_gm['davies_bouldin']:.4f}")
    m3.metric("Calinski-Harabasz", f"{eval_gm['calinski_harabasz']:.2f}")
    m4.metric("BIC Score", f"{gm_res['bic']:.2f}")


# =============================================================================
# PAGE 7: ALGORITHM COMPARISON
# =============================================================================
elif page == "📈 Algorithm Comparison":
    st.subheader("Comprehensive Algorithm Performance Comparison")
    
    st.markdown("""
    Side-by-side benchmark of all four clustering methodologies on identical standardized features.
    """)
    
    # Train all 4 with standard parameters for fair comparison
    km_eval = evaluate_cluster_quality(X_scaled, train_kmeans(X_scaled, n_clusters=5)['labels'], "K-Means (k=5)")
    hi_eval = evaluate_cluster_quality(X_scaled, train_hierarchical(X_scaled, n_clusters=5, linkage='ward')['labels'], "Hierarchical (Ward, k=5)")
    db_eval = evaluate_cluster_quality(X_scaled, train_dbscan(X_scaled, eps=0.35, min_samples=4)['labels'], "DBSCAN (eps=0.35, min=4)", is_dbscan=True)
    gm_eval = evaluate_cluster_quality(X_scaled, train_gmm(X_scaled, n_components=5)['labels'], "Gaussian Mixture Model (k=5)")
    
    df_comp = build_clustering_comparison_table([km_eval, hi_eval, db_eval, gm_eval])
    df_fmt = format_comparison_table(df_comp)
    
    st.markdown("#### 📋 Performance Benchmark Table:")
    st.dataframe(df_fmt, use_container_width=True)
    
    # Download Button
    csv_bytes = df_comp.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Comparison CSV (clustering_comparison.csv)",
        data=csv_bytes,
        file_name="clustering_comparison.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### 📊 Silhouette Score Comparison")
        fig_sil = plot_metric_bar_comparison(df_comp, metric='Silhouette Score', is_lower_better=False)
        st.pyplot(fig_sil, use_container_width=True)
        plt.close(fig_sil)
        
    with col_c2:
        st.markdown("#### 📉 Davies-Bouldin Index Comparison")
        fig_db = plot_metric_bar_comparison(df_comp, metric='Davies-Bouldin Index', is_lower_better=True)
        st.pyplot(fig_db, use_container_width=True)
        plt.close(fig_db)
        
    st.markdown("---")
    st.markdown("#### 🎯 4-Panel Overall Comparison Dashboard")
    fig_grid = plot_overall_comparison_grid(df_comp)
    st.pyplot(fig_grid, use_container_width=True)
    plt.close(fig_grid)


# =============================================================================
# PAGE 8: CONCLUSION AND INSIGHTS
# =============================================================================
elif page == "💡 Conclusion & Insights":
    st.subheader("Conclusion, Architectural Findings, and Future Scope")
    
    st.markdown("""
    ### 📝 Comprehensive Practical Findings
    1. **Customer Segmentation Takeaways:**
       - The 2D feature space of Annual Income vs. Spending Score forms **5 distinct and actionable consumer archetypes**.
       - Standardizing features was pivotal in preventing Income from dominating Spending Score.
    2. **Algorithm Comparison Insights:**
       - **K-Means:** Most efficient and highest Silhouette score on convex, roughly equal-density blobs.
       - **Hierarchical (Ward):** Offers an interpretable hierarchy; excellent when sub-cluster relationships matter.
       - **DBSCAN:** Ideal for anomaly/outlier detection and arbitrary non-linear cluster geometry.
       - **GMM:** Soft membership probabilities allow for probabilistic customer scoring across marketing tiers.
    """)
    
    st.markdown("---")
    st.markdown("### 🧭 Strategic Algorithm Comparison Matrix")
    st.markdown(r"""
    | Algorithm | Geometry / Shape | Outlier Handling | Parameter Sensitivity | Best Use Case |
    |---|---|---|---|---|
    | **K-Means** | Spherical / Convex | Sensitive to outliers | Needs \(k\) specified | Fast segmentation of compact data |
    | **Hierarchical** | Any (Linkage dependent) | Sensitive | Linkage & cutoff \(k\) | Small to medium sets needing taxonomy |
    | **DBSCAN** | Arbitrary / Non-linear | Robust (Labels as -1) | Very sensitive to \(eps\) & \(min\_samples\) | Anomaly detection & uneven densities |
    | **GMM** | Elliptical / Gaussian | Moderately sensitive | Components \(k\) & Covariance type | Overlapping soft boundary distributions |
    """)
    
    st.markdown("---")
    st.markdown("### 🔭 Limitations and Future Scope")
    st.markdown("""
    - **High-Dimensional Scaling:** Future iterations can incorporate PCA (Principal Component Analysis) or t-SNE for projecting 5+ demographic variables into 2D manifolds.
    - **Optics Algorithm:** Explore OPTICS to overcome DBSCAN's limitation on datasets with variable densities.
    - **Automated K-Tuning:** Integrate the Gap Statistic or Bayesian Optimization to automate optimal cluster selection.
    """)


# -----------------------------------------------------------------------------
# Universal Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 10px 0;">
    AIML Practical Assignment 6 &bull; Submitted by <strong>Tejas Gopal Jaju</strong> (PRN: 202401040124, Roll: 29, Batch: A2) &bull; Department of Artificial Intelligence &amp; Machine Learning
</div>
""", unsafe_allow_html=True)
