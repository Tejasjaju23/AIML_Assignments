"""
=============================================================================
AIML Practical Assignment 6: Clustering Algorithms
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: visualization.py - Matplotlib and Seaborn Visualizations
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Clean professional styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    sns.set_theme(style="whitegrid")


# Distinct high-contrast palette for clusters
CLUSTER_PALETTE = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261', '#457b9d', '#9b5de5', '#00bbf9', '#f15bb5']


# -----------------------------------------------------------------------------
# Exploratory Data Analysis Plots (1-4)
# -----------------------------------------------------------------------------
def plot_feature_distributions(df, save_path=None):
    """
    Plots individual histograms and KDE density distributions for:
    1. Age
    2. Annual Income
    3. Spending Score
    """
    cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    cols_present = [c for c in cols if c in df.columns]
    
    fig, axes = plt.subplots(1, len(cols_present), figsize=(5 * len(cols_present), 4.5))
    if len(cols_present) == 1:
        axes = [axes]
        
    palette = ['#457b9d', '#2a9d8f', '#e76f51']
    
    for idx, col in enumerate(cols_present):
        sns.histplot(
            df[col],
            kde=True,
            ax=axes[idx],
            color=palette[idx % len(palette)],
            edgecolor='black',
            linewidth=1.1,
            bins=15
        )
        axes[idx].set_title(f"Distribution: {col}", fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel(col, fontsize=10, fontweight='bold')
        axes[idx].set_ylabel("Count / Density", fontsize=10)
        
    fig.suptitle("Feature Distributions (Customer Demographics & Metrics)", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_single_distribution(df, column, color='#2a9d8f', save_path=None):
    """
    Plots distribution for a single individual feature.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df[column], kde=True, ax=ax, color=color, edgecolor='black', linewidth=1.1, bins=18)
    ax.set_title(f"Feature Distribution: {column}", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(column, fontsize=10, fontweight='bold')
    ax.set_ylabel("Count", fontsize=10)
    
    # Annotate mean and median
    mean_val = df[column].mean()
    median_val = df[column].median()
    ax.axvline(mean_val, color='#e63946', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color='#1d3557', linestyle=':', linewidth=1.8, label=f"Median: {median_val:.1f}")
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_correlation_heatmap(df, save_path=None):
    """
    Renders correlation heatmap across all numerical features.
    """
    num_df = df.select_dtypes(include=[np.number])
    # Drop CustomerID from correlation analysis if present
    cols_to_corr = [c for c in num_df.columns if 'id' not in c.lower()]
    corr = num_df[cols_to_corr].corr()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1.5,
        linecolor='white',
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"fontsize": 10, "fontweight": 'bold'}
    )
    ax.set_title("Numerical Features Correlation Heatmap", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# -----------------------------------------------------------------------------
# Clustering Model Visualizations (5-10)
# -----------------------------------------------------------------------------
def plot_elbow_method(elbow_results, optimal_k=5, save_path=None):
    """
    Plots the Elbow Curve (WCSS vs K) and the Silhouette Score curve side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    k_vals = elbow_results['k_values']
    wcss = elbow_results['wcss']
    
    # 1. WCSS Curve
    ax1.plot(k_vals, wcss, marker='o', markersize=8, color='#1d3557', linewidth=2.2, label='WCSS (Inertia)')
    if optimal_k in k_vals:
        opt_idx = k_vals.index(optimal_k)
        ax1.plot(optimal_k, wcss[opt_idx], marker='*', markersize=16, color='#e63946', label=f'Optimal K = {optimal_k}')
        ax1.axvline(optimal_k, color='#e63946', linestyle='--', alpha=0.7)
    ax1.set_title("Elbow Method (WCSS vs. Number of Clusters)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=10, fontweight='bold')
    ax1.set_xticks(k_vals)
    ax1.legend(loc='upper right', frameon=True)
    
    # 2. Silhouette Score Curve
    sil_dict = elbow_results['silhouette_scores']
    if sil_dict:
        k_sil = list(sil_dict.keys())
        v_sil = list(sil_dict.values())
        ax2.plot(k_sil, v_sil, marker='s', markersize=8, color='#2a9d8f', linewidth=2.2, label='Silhouette Score')
        best_k = max(sil_dict, key=sil_dict.get)
        ax2.plot(best_k, sil_dict[best_k], marker='*', markersize=16, color='#e76f51', label=f'Peak Silhouette (k={best_k})')
        ax2.axvline(best_k, color='#e76f51', linestyle='--', alpha=0.7)
        ax2.set_title("Silhouette Score vs. Number of Clusters", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight='bold')
        ax2.set_ylabel("Silhouette Score", fontsize=10, fontweight='bold')
        ax2.set_xticks(k_sil)
        ax2.legend(loc='lower right', frameon=True)
        
    fig.suptitle("K-Means Optimal Cluster Selection Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_2d_clusters(X, labels, title, x_label="Annual Income (k$)", y_label="Spending Score (1-100)",
                    centroids=None, noise_mask=None, save_path=None):
    """
    Renders a clear 2D scatter plot of clusters with optional centroids and noise points.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    # Extract coordinates
    if isinstance(X, pd.DataFrame):
        x_vals = X.iloc[:, 0].values
        y_vals = X.iloc[:, 1].values
    else:
        x_vals = X[:, 0]
        y_vals = X[:, 1]
        
    labels = np.asarray(labels)
    unique_labels = sorted(list(set(labels)))
    
    # Plot non-noise clusters
    for idx, lab in enumerate(unique_labels):
        if lab == -1:
            continue  # Noise handled separately
        mask = (labels == lab)
        color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
        ax.scatter(
            x_vals[mask],
            y_vals[mask],
            s=65,
            color=color,
            edgecolor='black',
            linewidth=0.8,
            alpha=0.85,
            label=f'Cluster {lab + 1} (n={np.sum(mask)})'
        )
        
    # Plot noise points for DBSCAN if present
    if -1 in unique_labels:
        mask_noise = (labels == -1)
        ax.scatter(
            x_vals[mask_noise],
            y_vals[mask_noise],
            s=75,
            color='#000000',
            marker='x',
            linewidth=1.8,
            label=f'Noise Points (n={np.sum(mask_noise)})'
        )
        
    # Plot centroids if supplied
    if centroids is not None:
        if isinstance(centroids, pd.DataFrame):
            c_x = centroids.iloc[:, 0].values
            c_y = centroids.iloc[:, 1].values
        else:
            c_x = centroids[:, 0]
            c_y = centroids[:, 1]
            
        ax.scatter(
            c_x,
            c_y,
            s=220,
            color='#ffff3f',
            edgecolor='black',
            linewidth=2.0,
            marker='P',
            label='Centroids'
        )
        
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_dendrogram(linkage_matrix, method='ward', truncate_level=5, save_path=None):
    """
    Renders hierarchical clustering dendrogram using SciPy.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sch.dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode='level',
        p=truncate_level,
        color_threshold=None,
        above_threshold_color='#457b9d'
    )
    
    ax.set_title(f"Hierarchical Clustering Dendrogram (Linkage Method: {method.capitalize()})",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Sample / Cluster Index (Truncated)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cophentic Distance / Cluster Dissimilarity", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# -----------------------------------------------------------------------------
# Comparison and Benchmark Plots (11-13)
# -----------------------------------------------------------------------------
def plot_metric_bar_comparison(df_comp, metric='Silhouette Score', is_lower_better=False, save_path=None):
    """
    Renders a bar chart comparing a specific metric across algorithms.
    """
    df_valid = df_comp.dropna(subset=[metric]).copy()
    if df_valid.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No valid data for {metric}", ha='center', va='center')
        return fig
        
    df_valid = df_valid.sort_values(by=metric, ascending=is_lower_better).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ['#2a9d8f' if idx == 0 else '#457b9d' for idx in range(len(df_valid))]
    
    bars = ax.barh(df_valid['Algorithm'], df_valid[metric], color=colors, edgecolor='black', linewidth=1.1, height=0.55)
    
    direction = "Lower is Better" if is_lower_better else "Higher is Better"
    ax.set_title(f"Clustering Performance: {metric} ({direction})", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel(metric, fontsize=10, fontweight='bold')
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.4f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha='left',
            va='center',
            fontsize=9,
            fontweight='bold'
        )
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_overall_comparison_grid(df_comp, save_path=None):
    """
    Creates a multi-panel visual summarizing Silhouette Score,
    Davies-Bouldin Index, Calinski-Harabasz Score, and Cluster Counts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Silhouette Score
    df_sil = df_comp.dropna(subset=['Silhouette Score']).sort_values(by='Silhouette Score', ascending=True)
    if not df_sil.empty:
        axes[0, 0].barh(df_sil['Algorithm'], df_sil['Silhouette Score'], color='#2a9d8f', edgecolor='black')
        axes[0, 0].set_title("Silhouette Score (Higher is Better)", fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel("Score")
        
    # 2. Davies-Bouldin Index
    df_db = df_comp.dropna(subset=['Davies-Bouldin Index']).sort_values(by='Davies-Bouldin Index', ascending=False)
    if not df_db.empty:
        axes[0, 1].barh(df_db['Algorithm'], df_db['Davies-Bouldin Index'], color='#e76f51', edgecolor='black')
        axes[0, 1].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel("Score")
        
    # 3. Calinski-Harabasz Score
    df_ch = df_comp.dropna(subset=['Calinski-Harabasz Score']).sort_values(by='Calinski-Harabasz Score', ascending=True)
    if not df_ch.empty:
        axes[1, 0].barh(df_ch['Algorithm'], df_ch['Calinski-Harabasz Score'], color='#457b9d', edgecolor='black')
        axes[1, 0].set_title("Calinski-Harabasz Score (Higher is Better)", fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel("Variance Ratio Score")
        
    # 4. Number of Clusters Identified
    axes[1, 1].barh(df_comp['Algorithm'], df_comp['Number of Clusters'], color='#e9c46a', edgecolor='black')
    axes[1, 1].set_title("Number of Identified Clusters", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Count")
    
    fig.suptitle("AIML Practical 6: Comprehensive Clustering Algorithm Comparison",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    from dataset import load_dataset
    from preprocessing import prepare_clustering_data
    from clustering_models import (
        run_kmeans_elbow_analysis,
        train_kmeans,
        train_hierarchical,
        compute_hierarchical_dendrogram_linkage,
        train_dbscan,
        train_gmm
    )
    from evaluation import evaluate_cluster_quality, build_clustering_comparison_table
    
    print("=" * 70)
    print("AIML Practical 6: Visualization Module Test")
    print("=" * 70)
    df, _, _, _ = load_dataset()
    prep = prepare_clustering_data(df, feature_mode='2d')
    X_raw = prep['X_raw']
    X_scaled = prep['X_scaled']
    
    out_dir = os.path.join("AIML_Practical_6_Tejas_Jaju", "outputs")
    comp_dir = os.path.join(out_dir, "comparison_graphs")
    os.makedirs(comp_dir, exist_ok=True)
    
    print("Generating EDA charts...")
    plot_single_distribution(df, 'Age', save_path=os.path.join(out_dir, "age_distribution.png"))
    plot_single_distribution(df, 'Annual Income (k$)', save_path=os.path.join(out_dir, "income_distribution.png"))
    plot_single_distribution(df, 'Spending Score (1-100)', save_path=os.path.join(out_dir, "spending_distribution.png"))
    plot_correlation_heatmap(df, save_path=os.path.join(out_dir, "correlation_heatmap.png"))
    
    print("Generating Elbow and Dendrogram...")
    elbow_res = run_kmeans_elbow_analysis(X_scaled)
    plot_elbow_method(elbow_res, optimal_k=5, save_path=os.path.join(out_dir, "elbow_method.png"))
    
    linkage_mat = compute_hierarchical_dendrogram_linkage(X_scaled, method='ward')
    plot_dendrogram(linkage_mat, method='ward', save_path=os.path.join(out_dir, "dendrogram.png"))
    
    print("Generating cluster scatter plots...")
    km_res = train_kmeans(X_scaled, n_clusters=5)
    # Inverse transform centroids to original feature scale for intuitive plotting
    centroids_raw = prep['scaler'].inverse_transform(km_res['centroids_scaled'])
    plot_2d_clusters(X_raw, km_res['labels'], "K-Means Customer Clusters (k=5)", centroids=centroids_raw,
                    save_path=os.path.join(out_dir, "kmeans_clusters.png"))
    
    hi_res = train_hierarchical(X_scaled, n_clusters=5, linkage='ward')
    plot_2d_clusters(X_raw, hi_res['labels'], "Hierarchical Customer Clusters (Ward Linkage, k=5)",
                    save_path=os.path.join(out_dir, "hierarchical_clusters.png"))
                    
    db_res = train_dbscan(X_scaled, eps=0.35, min_samples=4)
    plot_2d_clusters(X_raw, db_res['labels'], f"DBSCAN Clusters (eps=0.35, min_samples=4)",
                    save_path=os.path.join(out_dir, "dbscan_clusters.png"))
                    
    gm_res = train_gmm(X_scaled, n_components=5)
    plot_2d_clusters(X_raw, gm_res['labels'], "Gaussian Mixture Model Clusters (k=5)",
                    save_path=os.path.join(out_dir, "gmm_clusters.png"))
                    
    print("Generating comparison charts...")
    eval_km = evaluate_cluster_quality(X_scaled, km_res['labels'], "K-Means")
    eval_hi = evaluate_cluster_quality(X_scaled, hi_res['labels'], "Hierarchical (Ward)")
    eval_db = evaluate_cluster_quality(X_scaled, db_res['labels'], "DBSCAN", is_dbscan=True)
    eval_gm = evaluate_cluster_quality(X_scaled, gm_res['labels'], "Gaussian Mixture Model")
    df_comp = build_clustering_comparison_table([eval_km, eval_hi, eval_db, eval_gm])
    
    plot_overall_comparison_grid(df_comp, save_path=os.path.join(comp_dir, "overall_clustering_comparison.png"))
    plot_metric_bar_comparison(df_comp, metric='Silhouette Score', is_lower_better=False,
                              save_path=os.path.join(comp_dir, "silhouette_score_comparison.png"))
    plot_metric_bar_comparison(df_comp, metric='Davies-Bouldin Index', is_lower_better=True,
                              save_path=os.path.join(comp_dir, "davies_bouldin_comparison.png"))
                              
    print("All 13 visualizations successfully generated and saved.")
