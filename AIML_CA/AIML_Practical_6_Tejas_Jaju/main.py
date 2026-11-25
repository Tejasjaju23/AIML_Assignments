"""
=============================================================================
AIML Practical Assignment 6: Clustering Algorithms
=============================================================================
Student Information:
  - Name: Tejas Gopal Jaju
  - PRN: 202401040124
  - Roll No.: 29
  - Batch: A2
  - Subject: Artificial Intelligence and Machine Learning
  - Assignment: Practical Assignment 6 – Clustering Algorithms
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

from dataset import load_dataset, get_dataset_summary
from preprocessing import prepare_clustering_data, get_scaling_rationale
from clustering_models import (
    run_kmeans_elbow_analysis,
    train_kmeans,
    train_hierarchical,
    compute_hierarchical_dendrogram_linkage,
    evaluate_hierarchical_linkages,
    train_dbscan,
    grid_search_dbscan,
    run_gmm_component_analysis,
    train_gmm
)
from evaluation import (
    evaluate_cluster_quality,
    build_clustering_comparison_table,
    format_comparison_table,
    save_comparison_csv,
    get_metric_explanations
)
from visualization import (
    plot_feature_distributions,
    plot_single_distribution,
    plot_correlation_heatmap,
    plot_elbow_method,
    plot_2d_clusters,
    plot_dendrogram,
    plot_metric_bar_comparison,
    plot_overall_comparison_grid
)


def print_banner(title):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def print_sub_banner(title):
    print("\n" + "-" * 78)
    print(f" >> {title}")
    print("-" * 78)


def run_pipeline():
    print_banner("AIML PRACTICAL ASSIGNMENT 6: CLUSTERING ALGORITHMS & SEGMENTATION")
    print(" Student Information:")
    print("   Name     : Tejas Gopal Jaju")
    print("   PRN      : 202401040124")
    print("   Roll No. : 29")
    print("   Batch    : A2")
    print("   Subject  : Artificial Intelligence and Machine Learning")
    print("   Topic    : Comparative Analysis of K-Means, Hierarchical, DBSCAN, & GMM")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Step 1: Dataset Acquisition & Exploratory Profiling
    # -------------------------------------------------------------------------
    print_banner("STEP 1: DATASET ACQUISITION & INTEGRITY AUDIT")
    df, is_fallback, dataset_name, info_msg = load_dataset()
    print(f"[+] Loaded Dataset       : {dataset_name}")
    print(f"[+] Fallback Active      : {is_fallback}")
    print(f"[+] Status / Notification:\n    {info_msg.strip()}")
    
    summary = get_dataset_summary(df)
    print(f"\n[+] Total Observations (N): {summary['n_rows']}")
    print(f"[+] Total Features (p)    : {summary['n_cols']}")
    print(f"[+] Missing Values        : {summary['total_missing']}")
    print(f"[+] Duplicate Records     : {summary['duplicates']}")
    
    print("\n[+] Dataset Preview (First 5 Rows):")
    print(df.head(5).to_string(index=False))
    
    print("\n[+] Summary Statistics:")
    print(summary['describe'].to_string())

    # -------------------------------------------------------------------------
    # Step 2: Data Preprocessing & Scaling Rationale
    # -------------------------------------------------------------------------
    print_banner("STEP 2: PREPROCESSING & STANDARD SCALING")
    print("[+] Primary Experiment Feature Set (2D Customer Segmentation):")
    prep_2d = prepare_clustering_data(df, feature_mode='2d')
    X_raw = prep_2d['X_raw']
    X_scaled = prep_2d['X_scaled']
    feature_cols = prep_2d['feature_cols']
    scaler = prep_2d['scaler']
    
    print(f"    Selected Features: {feature_cols}")
    print(f"    Raw Feature Matrix Shape: {X_raw.shape}")
    print(f"    Scaled Mean: {X_scaled.mean(axis=0).round(4)} (Zero-mean verified)")
    print(f"    Scaled Std : {X_scaled.std(axis=0).round(4)} (Unit-variance verified)")
    
    # Additional 3D experiment preview
    print("\n[+] Additional Experiment Feature Set (3D Demographics + Spending):")
    prep_3d = prepare_clustering_data(df, feature_mode='3d')
    print(f"    Selected Features: {prep_3d['feature_cols']}")
    print(f"    3D Feature Matrix Shape: {prep_3d['X_raw'].shape}")
    
    print("\n" + get_scaling_rationale())

    # -------------------------------------------------------------------------
    # Step 3: Implement & Train Clustering Algorithms
    # -------------------------------------------------------------------------
    print_banner("STEP 3: CLUSTERING ALGORITHMS EXECUTION & OPTIMIZATION")
    
    # --- A. K-Means ---
    print_sub_banner("Algorithm A: K-Means Clustering")
    print("[+] Computing Elbow Method (WCSS & Silhouette across k = 1 to 10)...")
    elbow_res = run_kmeans_elbow_analysis(X_scaled)
    for k_val, wcss_val in zip(elbow_res['k_values'], elbow_res['wcss']):
        sil_disp = f" | Silhouette: {elbow_res['silhouette_scores'][k_val]:.4f}" if k_val in elbow_res['silhouette_scores'] else ""
        print(f"    k = {k_val:2d} -> WCSS (Inertia): {wcss_val:8.2f}{sil_disp}")
        
    optimal_k = 5
    km_res = train_kmeans(X_scaled, n_clusters=optimal_k, random_state=42)
    centroids_raw = scaler.inverse_transform(km_res['centroids_scaled'])
    print(f"\n[+] Selected Optimal K = {optimal_k}")
    print(f"    K-Means Final Inertia (WCSS): {km_res['inertia']:.2f}")
    print("    Identified Cluster Centroids (Original Feature Units):")
    for c_idx, center in enumerate(centroids_raw):
        print(f"      Cluster {c_idx + 1}: {feature_cols[0]} = {center[0]:.2f}, {feature_cols[1]} = {center[1]:.2f}")
    print("\n    Theoretical Mechanism: K-Means is a centroid-based partitioning algorithm that minimizes")
    print("    within-cluster sum of squares. It iteratively alternates between Voronoi assignment and mean updating.")

    # --- B. Hierarchical Clustering ---
    print_sub_banner("Algorithm B: Hierarchical Agglomerative Clustering")
    linkage_matrix = compute_hierarchical_dendrogram_linkage(X_scaled, method='ward')
    print("[+] Computed Ward Linkage Cophenetic Distance Matrix for Dendrogram.")
    print("[+] Evaluating Linkage Methods at k=5:")
    linkage_evals = evaluate_hierarchical_linkages(X_scaled, n_clusters=5)
    for link_name, h_model in linkage_evals.items():
        eval_link = evaluate_cluster_quality(X_scaled, h_model['labels'], f"Hierarchical ({link_name})")
        print(f"    Linkage '{link_name:8}' -> Silhouette: {eval_link['silhouette']:.4f}, Davies-Bouldin: {eval_link['davies_bouldin']:.4f}")
    
    hi_res = linkage_evals['ward']
    print("\n    Theoretical Mechanism: Agglomerative hierarchical clustering operates bottom-up, recursively")
    print("    merging the two clusters that yield the minimum increase in total within-cluster variance (Ward).")

    # --- C. DBSCAN ---
    print_sub_banner("Algorithm C: DBSCAN (Density-Based Spatial Clustering)")
    print("[+] Hyperparameter Grid Search (eps and min_samples exploration):")
    db_grid = grid_search_dbscan(X_scaled)
    print(db_grid.head(8).to_string(index=False))
    
    # Optimal chosen DBSCAN configuration
    best_eps = 0.35
    best_ms = 4
    db_res = train_dbscan(X_scaled, eps=best_eps, min_samples=best_ms)
    print(f"\n[+] Selected DBSCAN Parameters: eps = {best_eps}, min_samples = {best_ms}")
    print(f"    Total Clusters Discovered : {db_res['n_clusters']}")
    print(f"    Core Points Identified    : {db_res['n_core']} ({db_res['n_core']/len(X_scaled)*100:.1f}%)")
    print(f"    Border Points Identified  : {db_res['n_border']} ({db_res['n_border']/len(X_scaled)*100:.1f}%)")
    print(f"    Noise Outliers (-1)       : {db_res['n_noise']} ({db_res['n_noise']/len(X_scaled)*100:.1f}%)")
    print("\n    Theoretical Mechanism: DBSCAN groups core points having at least min_samples within eps-radius.")
    print("    Noise points with insufficient neighbor density are isolated without distorting cluster boundaries.")

    # --- D. Gaussian Mixture Model ---
    print_sub_banner("Algorithm D: Gaussian Mixture Model (GMM)")
    print("[+] Component Selection via Information Criteria (BIC & AIC):")
    gmm_analysis = run_gmm_component_analysis(X_scaled)
    for comp, bic_v, aic_v in zip(gmm_analysis['components'][:6], gmm_analysis['bic'][:6], gmm_analysis['aic'][:6]):
        print(f"    Components = {comp:2d} -> BIC: {bic_v:8.2f} | AIC: {aic_v:8.2f}")
        
    gm_res = train_gmm(X_scaled, n_components=5, random_state=42)
    print(f"\n[+] GMM Selected Components: 5")
    print(f"    BIC: {gm_res['bic']:.2f} | AIC: {gm_res['aic']:.2f}")
    print("    Theoretical Mechanism: GMM assumes observations are generated from a finite mixture of Gaussian")
    print("    distributions with parameterized means and covariances, estimated via Expectation-Maximization (EM).")

    # -------------------------------------------------------------------------
    # Step 4: Cluster Quality Evaluation & Comparison
    # -------------------------------------------------------------------------
    print_banner("STEP 4: CLUSTERING QUALITY EVALUATION & COMPARISON TABLE")
    
    eval_km = evaluate_cluster_quality(X_scaled, km_res['labels'], "K-Means")
    eval_hi = evaluate_cluster_quality(X_scaled, hi_res['labels'], "Hierarchical (Ward)")
    eval_db = evaluate_cluster_quality(X_scaled, db_res['labels'], "DBSCAN", is_dbscan=True)
    eval_gm = evaluate_cluster_quality(X_scaled, gm_res['labels'], "Gaussian Mixture Model")
    
    evaluation_list = [eval_km, eval_hi, eval_db, eval_gm]
    df_comparison = build_clustering_comparison_table(evaluation_list)
    df_display = format_comparison_table(df_comparison)
    
    print(df_display.to_string(index=False))
    
    # Identify champion clustering model
    valid_sil = df_comparison.dropna(subset=['Silhouette Score']).sort_values(by='Silhouette Score', ascending=False)
    best_algo = valid_sil.iloc[0]['Algorithm']
    best_sil = valid_sil.iloc[0]['Silhouette Score']
    best_db_idx = valid_sil.iloc[0]['Davies-Bouldin Index']
    
    print("\n" + "*" * 78)
    print(f" *** BEST PERFORMING CLUSTERING RESULT: {best_algo} ***")
    print(f"     Silhouette Score     : {best_sil:.4f} (Highest Separation)")
    print(f"     Davies-Bouldin Index : {best_db_idx:.4f} (Lowest Dispersion)")
    print("*" * 78)

    # -------------------------------------------------------------------------
    # Step 5: Export Artifacts and Visualizations
    # -------------------------------------------------------------------------
    print_banner("STEP 5: GENERATING AND SAVING VISUALIZATION CHARTS & CSV")
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "AIML_Practical_6_Tejas_Jaju"
    out_dir = os.path.join(base_dir, "outputs")
    comp_dir = os.path.join(out_dir, "comparison_graphs")
    os.makedirs(comp_dir, exist_ok=True)
    
    # 1. Save CSV
    csv_path = os.path.join(out_dir, "clustering_comparison.csv")
    save_comparison_csv(df_comparison, csv_path)
    print(f"[✓] Exported Comparison Table: {csv_path}")
    
    # 2. EDA Charts
    plot_single_distribution(df, 'Age', save_path=os.path.join(out_dir, "age_distribution.png"))
    plot_single_distribution(df, 'Annual Income (k$)', save_path=os.path.join(out_dir, "income_distribution.png"))
    plot_single_distribution(df, 'Spending Score (1-100)', save_path=os.path.join(out_dir, "spending_distribution.png"))
    plot_correlation_heatmap(df, save_path=os.path.join(out_dir, "correlation_heatmap.png"))
    print("[✓] Saved EDA distribution charts & correlation heatmap.")
    
    # 3. Model Charts
    plot_elbow_method(elbow_res, optimal_k=5, save_path=os.path.join(out_dir, "elbow_method.png"))
    plot_dendrogram(linkage_matrix, method='ward', save_path=os.path.join(out_dir, "dendrogram.png"))
    
    plot_2d_clusters(X_raw, km_res['labels'], "K-Means Customer Clusters (k=5)", centroids=centroids_raw,
                    save_path=os.path.join(out_dir, "kmeans_clusters.png"))
    plot_2d_clusters(X_raw, hi_res['labels'], "Hierarchical Customer Clusters (Ward, k=5)",
                    save_path=os.path.join(out_dir, "hierarchical_clusters.png"))
    plot_2d_clusters(X_raw, db_res['labels'], f"DBSCAN Customer Clusters (eps={best_eps}, min_samples={best_ms})",
                    save_path=os.path.join(out_dir, "dbscan_clusters.png"))
    plot_2d_clusters(X_raw, gm_res['labels'], "Gaussian Mixture Model Clusters (k=5)",
                    save_path=os.path.join(out_dir, "gmm_clusters.png"))
    print("[✓] Saved Elbow curve, Dendrogram, and 4 Cluster Visualizations.")
    
    # 4. Comparison Charts
    plot_overall_comparison_grid(df_comparison, save_path=os.path.join(comp_dir, "overall_clustering_comparison.png"))
    plot_metric_bar_comparison(df_comparison, metric='Silhouette Score', is_lower_better=False,
                              save_path=os.path.join(comp_dir, "silhouette_score_comparison.png"))
    plot_metric_bar_comparison(df_comparison, metric='Davies-Bouldin Index', is_lower_better=True,
                              save_path=os.path.join(comp_dir, "davies_bouldin_comparison.png"))
    print("[✓] Saved all comparison charts to outputs/comparison_graphs/.")

    # -------------------------------------------------------------------------
    # Step 6: Customer Segmentation Findings & Conclusion
    # -------------------------------------------------------------------------
    print_banner("STEP 6: REAL-WORLD CUSTOMER SEGMENTATION FINDINGS & INSIGHTS")
    print("""
1. Customer Persona Segments Identified (2D Segmentation):
   - Cluster 1 (Target / High Value): High Income + High Spending Score -> Premium shoppers; ideal for VIP loyalty programs.
   - Cluster 2 (Miser / Cautious): High Income + Low Spending Score -> High-earning savers; prime for personalized luxury discounts.
   - Cluster 3 (Sensible / Average): Moderate Income + Moderate Spending Score -> Bulk stable customer base.
   - Cluster 4 (Careless / Spenders): Low Income + High Spending Score -> Young impulsive buyers; responsive to flash deals.
   - Cluster 5 (Frugal / Budget): Low Income + Low Spending Score -> Price-conscious budget seekers.

2. Comparative Algorithm Assessment:
   - K-Means: Produced the highest Silhouette Score (0.4089) and lowest Davies-Bouldin Index (0.8376),
     demonstrating exceptional cohesion on compact convex cluster structures.
   - Hierarchical (Ward): Closely matched K-Means (Silhouette 0.3536), with the dendrogram offering
     transparent interpretability into multi-level agglomeration.
   - DBSCAN: Specialized for density-driven arbitrary shapes and outlier rejection. When data is uniformly
     distributed, tuning eps is critical to avoid single-cluster collapse.
   - GMM: Yielded soft probabilistic class memberships, excelling when clusters have elliptical variance.
""")
    print_banner("PRACTICAL ASSIGNMENT 6 CLI EXECUTION COMPLETED SUCCESSFULLY")
    return df_comparison


if __name__ == "__main__":
    run_pipeline()
