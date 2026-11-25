"""
=============================================================================
AIML Practical Assignment 6: Clustering Algorithms
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: evaluation.py - Cluster Evaluation Metrics and Comparison Table
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_cluster_quality(X_scaled, labels, algorithm_name, is_dbscan=False):
    """
    Computes Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
    
    Robust Handling:
      - Excludes DBSCAN noise points (label == -1) when calculating separation/compactness.
      - Never computes Silhouette or other metrics when fewer than 2 valid clusters exist.
      - Returns cleanly structured dictionary with numerical values or None.
    """
    labels = np.asarray(labels)
    
    if is_dbscan:
        valid_mask = (labels != -1)
        valid_labels = labels[valid_mask]
        valid_X = X_scaled[valid_mask]
        n_noise = int(np.sum(labels == -1))
        unique_valid_clusters = set(valid_labels)
        n_clusters = len(unique_valid_clusters)
        
        # Check if at least 2 distinct clusters exist to compute metrics
        if n_clusters >= 2 and len(valid_X) > n_clusters:
            sil = float(silhouette_score(valid_X, valid_labels))
            db = float(davies_bouldin_score(valid_X, valid_labels))
            ch = float(calinski_harabasz_score(valid_X, valid_labels))
        else:
            sil = None
            db = None
            ch = None
            
        return {
            'algorithm': algorithm_name,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch,
            'is_valid_evaluation': (sil is not None),
            'note': f"Noise points: {n_noise} (excluded from quality metrics)" if n_noise > 0 else "No noise detected"
        }
        
    else:
        unique_clusters = set(labels)
        n_clusters = len(unique_clusters)
        
        if n_clusters >= 2 and len(X_scaled) > n_clusters:
            sil = float(silhouette_score(X_scaled, labels))
            db = float(davies_bouldin_score(X_scaled, labels))
            ch = float(calinski_harabasz_score(X_scaled, labels))
            is_valid = True
        else:
            sil = None
            db = None
            ch = None
            is_valid = False
            
        return {
            'algorithm': algorithm_name,
            'n_clusters': n_clusters,
            'n_noise': 0,
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch,
            'is_valid_evaluation': is_valid,
            'note': "All points assigned to clusters"
        }


def build_clustering_comparison_table(evaluation_results_list):
    """
    Compiles evaluation dictionaries into a clean Pandas DataFrame:
    | Algorithm | Number of Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
    """
    rows = []
    for res in evaluation_results_list:
        rows.append({
            'Algorithm': res['algorithm'],
            'Number of Clusters': res['n_clusters'],
            'Silhouette Score': res['silhouette'] if res['silhouette'] is not None else np.nan,
            'Davies-Bouldin Index': res['davies_bouldin'] if res['davies_bouldin'] is not None else np.nan,
            'Calinski-Harabasz Score': res['calinski_harabasz'] if res['calinski_harabasz'] is not None else np.nan,
            'Noise Points': res['n_noise']
        })
        
    df_comp = pd.DataFrame(rows)
    return df_comp


def format_comparison_table(df_comp):
    """
    Formats the comparison DataFrame for terminal or markdown display.
    """
    df_fmt = df_comp.copy()
    df_fmt['Silhouette Score'] = df_fmt['Silhouette Score'].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A (<2 clusters)"
    )
    df_fmt['Davies-Bouldin Index'] = df_fmt['Davies-Bouldin Index'].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
    )
    df_fmt['Calinski-Harabasz Score'] = df_fmt['Calinski-Harabasz Score'].apply(
        lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
    )
    return df_fmt


def get_metric_explanations():
    """
    Returns mathematical definitions and critical limitations of internal clustering metrics.
    """
    return {
        'Silhouette Score': {
            'formula': 's(i) = (b(i) - a(i)) / max(a(i), b(i))',
            'range': '[-1.0, +1.0]',
            'interpretation': 'Higher is better. Values near +1 indicate dense, well-separated clusters; values near 0 indicate overlapping boundaries; negative values suggest sample misassignment.',
            'limitations': 'Inherently favors convex, spherical clusters (like K-Means). Heavily penalizes complex manifold, non-linear, or concentric density-based cluster geometries.'
        },
        'Davies-Bouldin Index': {
            'formula': 'DB = (1 / k) * sum_{i=1}^k max_{j != i} ((s_i + s_j) / d(c_i, c_j))',
            'range': '[0, +inf)',
            'interpretation': 'Lower is better. A value closer to 0 indicates clusters that are compact within and widely separated between centroids.',
            'limitations': 'Calculates cluster dispersion relative to centroid distances; vulnerable to non-spherical clusters and sensitive to outliers.'
        },
        'Calinski-Harabasz Index': {
            'formula': 'CH = (Tr(B_k) / Tr(W_k)) * ((N - k) / (k - 1))',
            'range': '[0, +inf)',
            'interpretation': 'Higher is better. Defined as the ratio of between-cluster dispersion (B_k) to within-cluster dispersion (W_k) normalized by degrees of freedom.',
            'limitations': 'Tends to increase monotonically as k grows, which can artificially bias selection toward higher cluster counts unless cross-verified with Silhouette scores.'
        }
    }


def save_comparison_csv(df_comp, filepath):
    """
    Exports the numerical comparison DataFrame to CSV.
    """
    df_comp.to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    from dataset import load_dataset
    from preprocessing import prepare_clustering_data
    from clustering_models import train_kmeans, train_hierarchical, train_dbscan, train_gmm
    
    print("=" * 70)
    print("AIML Practical 6: Evaluation Module Test")
    print("=" * 70)
    df, _, _, _ = load_dataset()
    prep = prepare_clustering_data(df, feature_mode='2d')
    X_scaled = prep['X_scaled']
    
    km_res = train_kmeans(X_scaled, n_clusters=5)
    hi_res = train_hierarchical(X_scaled, n_clusters=5, linkage='ward')
    db_res = train_dbscan(X_scaled, eps=0.35, min_samples=4)
    gm_res = train_gmm(X_scaled, n_components=5)
    
    eval_km = evaluate_cluster_quality(X_scaled, km_res['labels'], "K-Means")
    eval_hi = evaluate_cluster_quality(X_scaled, hi_res['labels'], "Hierarchical (Ward)")
    eval_db = evaluate_cluster_quality(X_scaled, db_res['labels'], "DBSCAN", is_dbscan=True)
    eval_gm = evaluate_cluster_quality(X_scaled, gm_res['labels'], "Gaussian Mixture Model")
    
    df_comp = build_clustering_comparison_table([eval_km, eval_hi, eval_db, eval_gm])
    df_fmt = format_comparison_table(df_comp)
    print("\nComparison Table:")
    print(df_fmt.to_string(index=False))
