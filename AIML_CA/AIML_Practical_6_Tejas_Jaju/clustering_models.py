"""
=============================================================================
AIML Practical Assignment 6: Clustering Algorithms
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: clustering_models.py - Model Implementations (K-Means, Hierarchical, DBSCAN, GMM)
=============================================================================
"""

import os
import warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch


# -----------------------------------------------------------------------------
# 1. K-Means Clustering
# -----------------------------------------------------------------------------
def run_kmeans_elbow_analysis(X_scaled, k_range=range(1, 11), random_state=42):
    """
    Computes Within-Cluster Sum of Squares (WCSS / Inertia) and Silhouette Scores
    for k in k_range to identify optimal number of clusters via the Elbow Method.
    """
    wcss = []
    silhouette_scores = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
        kmeans.fit(X_scaled)
        wcss.append(float(kmeans.inertia_))
        
        # Silhouette Score requires at least 2 clusters
        if k >= 2 and k < len(X_scaled):
            score = float(silhouette_score(X_scaled, kmeans.labels_))
            silhouette_scores[k] = score
            
    return {
        'k_values': list(k_range),
        'wcss': wcss,
        'silhouette_scores': silhouette_scores
    }


def train_kmeans(X_scaled, n_clusters=5, random_state=42):
    """
    Trains a K-Means model with specified clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    
    return {
        'model': kmeans,
        'labels': labels,
        'n_clusters': n_clusters,
        'centroids_scaled': kmeans.cluster_centers_,
        'inertia': float(kmeans.inertia_)
    }


# -----------------------------------------------------------------------------
# 2. Hierarchical (Agglomerative) Clustering
# -----------------------------------------------------------------------------
def compute_hierarchical_dendrogram_linkage(X_scaled, method='ward'):
    """
    Computes hierarchical clustering linkage matrix using SciPy.
    Supported methods: 'ward', 'complete', 'average'.
    """
    linkage_matrix = sch.linkage(X_scaled, method=method)
    return linkage_matrix


def train_hierarchical(X_scaled, n_clusters=5, linkage='ward'):
    """
    Trains an Agglomerative Hierarchical Clustering model.
    """
    # Note: for ward linkage, metric must be euclidean
    metric = 'euclidean'
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric
    )
    labels = model.fit_predict(X_scaled)
    
    return {
        'model': model,
        'labels': labels,
        'n_clusters': n_clusters,
        'linkage': linkage
    }


def evaluate_hierarchical_linkages(X_scaled, n_clusters=5, linkages=('ward', 'complete', 'average')):
    """
    Compares different linkage methods for a given cluster count.
    """
    results = {}
    for link in linkages:
        h_res = train_hierarchical(X_scaled, n_clusters=n_clusters, linkage=link)
        results[link] = h_res
    return results


# -----------------------------------------------------------------------------
# 3. DBSCAN Clustering
# -----------------------------------------------------------------------------
def train_dbscan(X_scaled, eps=0.35, min_samples=4):
    """
    Fits DBSCAN clustering and differentiates core, border, and noise points.
    
    Labels:
      - >= 0 : Valid cluster identifier
      - -1   : Noise point (outlier)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # Identify unique clusters excluding noise (-1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    
    # Core sample indices
    core_sample_indices = set(dbscan.core_sample_indices_)
    
    # Differentiate point types:
    # 0: Core Point, 1: Border Point, -1: Noise Point
    point_types = np.zeros(len(labels), dtype=int)
    for idx in range(len(labels)):
        if labels[idx] == -1:
            point_types[idx] = -1  # Noise
        elif idx in core_sample_indices:
            point_types[idx] = 1   # Core
        else:
            point_types[idx] = 0   # Border
            
    n_noise = int(np.sum(labels == -1))
    n_core = len(core_sample_indices)
    n_border = len(labels) - n_noise - n_core
    
    return {
        'model': dbscan,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_core': n_core,
        'n_border': n_border,
        'point_types': point_types,
        'eps': eps,
        'min_samples': min_samples
    }


def grid_search_dbscan(X_scaled, eps_range=np.linspace(0.2, 0.7, 6), min_samples_range=(3, 4, 5, 6)):
    """
    Explores hyperparameter grid for DBSCAN to inspect resulting clusters and noise points.
    """
    grid_records = []
    for eps in eps_range:
        for ms in min_samples_range:
            res = train_dbscan(X_scaled, eps=float(eps), min_samples=int(ms))
            grid_records.append({
                'eps': round(float(eps), 3),
                'min_samples': int(ms),
                'n_clusters': res['n_clusters'],
                'n_noise': res['n_noise'],
                'noise_ratio': round(res['n_noise'] / len(X_scaled), 3)
            })
    return pd.DataFrame(grid_records)


# -----------------------------------------------------------------------------
# 4. Gaussian Mixture Model (GMM)
# -----------------------------------------------------------------------------
def run_gmm_component_analysis(X_scaled, n_components_range=range(2, 11), random_state=42):
    """
    Computes AIC, BIC, and Silhouette Scores across component counts
    to determine optimal probabilistic clusters.
    """
    bic_scores = []
    aic_scores = []
    sil_scores = {}
    
    for k in n_components_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=5, random_state=random_state)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        
        bic = float(gmm.bic(X_scaled))
        aic = float(gmm.aic(X_scaled))
        bic_scores.append(bic)
        aic_scores.append(aic)
        
        if len(set(labels)) >= 2:
            sil = float(silhouette_score(X_scaled, labels))
            sil_scores[k] = sil
            
    return {
        'components': list(n_components_range),
        'bic': bic_scores,
        'aic': aic_scores,
        'silhouette_scores': sil_scores
    }


def train_gmm(X_scaled, n_components=5, random_state=42):
    """
    Trains a Gaussian Mixture Model and computes soft class membership probabilities.
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=5, random_state=random_state)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    return {
        'model': gmm,
        'labels': labels,
        'probabilities': probabilities,
        'n_clusters': n_components,
        'means': gmm.means_,
        'covariances': gmm.covariances_,
        'bic': float(gmm.bic(X_scaled)),
        'aic': float(gmm.aic(X_scaled))
    }


if __name__ == "__main__":
    from dataset import load_dataset
    from preprocessing import prepare_clustering_data
    
    print("=" * 70)
    print("AIML Practical 6: Clustering Models Module Test")
    print("=" * 70)
    df, _, name, _ = load_dataset()
    prep = prepare_clustering_data(df, feature_mode='2d')
    X_scaled = prep['X_scaled']
    
    print("\n[1] Testing K-Means...")
    km = train_kmeans(X_scaled, n_clusters=5)
    print(f"K-Means clusters: {km['n_clusters']}, Inertia: {km['inertia']:.2f}")
    
    print("\n[2] Testing Hierarchical...")
    hi = train_hierarchical(X_scaled, n_clusters=5, linkage='ward')
    print(f"Hierarchical clusters: {hi['n_clusters']}, Linkage: {hi['linkage']}")
    
    print("\n[3] Testing DBSCAN...")
    db = train_dbscan(X_scaled, eps=0.35, min_samples=4)
    print(f"DBSCAN clusters: {db['n_clusters']}, Noise points: {db['n_noise']}, Core points: {db['n_core']}")
    
    print("\n[4] Testing GMM...")
    gmm = train_gmm(X_scaled, n_components=5)
    print(f"GMM clusters: {gmm['n_clusters']}, BIC: {gmm['bic']:.2f}, AIC: {gmm['aic']:.2f}")
    
    print("\nAll 4 clustering algorithms successfully verified.")
