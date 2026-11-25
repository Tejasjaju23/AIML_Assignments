"""
=============================================================================
AIML Practical Assignment 6: Clustering Algorithms
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: preprocessing.py - Feature Selection, Encoding, and Standard Scaling
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Define standard feature sets
FEATURES_2D = ['Annual Income (k$)', 'Spending Score (1-100)']
FEATURES_3D = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']


def get_scaling_rationale():
    """
    Returns an academic and mathematical explanation of why feature scaling
    is critical for distance-based clustering algorithms.
    """
    return (
        "Why Feature Scaling is Essential for Clustering Algorithms:\n"
        "1. Distance-Metric Sensitivity: Algorithms such as K-Means, Hierarchical (Agglomerative), "
        "and DBSCAN rely heavily on pairwise geometric distances (typically Euclidean: d(p,q) = sqrt(sum (p_i - q_i)^2)).\n"
        "2. Magnitude Dominance: If features operate on vastly different scales (e.g., Annual Income spanning "
        "$15k to $140k vs. Spending Score spanning 1 to 100), the feature with the larger raw numeric variance "
        "will artificially dominate the distance calculation, rendering smaller-scale features negligible.\n"
        "3. Geometric Isotropy: Applying StandardScaler transforms each feature to have a mean of 0 (mu=0) and "
        "unit variance (sigma=1), ensuring that all orthogonal feature dimensions contribute equally and fairly "
        "to the formation of cluster boundaries and Voronoi cells.\n"
    )


def encode_categorical_features(df):
    """
    Encodes categorical features (such as Gender) into numeric representation.
    """
    df_clean = df.copy()
    if 'Gender' in df_clean.columns:
        # 0: Female, 1: Male
        df_clean['Gender_Encoded'] = df_clean['Gender'].map({'Female': 0, 'Male': 1}).fillna(0).astype(int)
    return df_clean


def select_features(df, feature_mode='2d'):
    """
    Extracts numerical features based on the experiment mode.
    
    Args:
        df (DataFrame): Input customer data.
        feature_mode (str): '2d' for [Income, Spending] or '3d' for [Age, Income, Spending].
        
    Returns:
        X (DataFrame): Selected raw feature matrix.
        feature_cols (list): Selected feature column names.
    """
    if feature_mode.lower() == '3d':
        cols = [col for col in FEATURES_3D if col in df.columns]
    else:
        cols = [col for col in FEATURES_2D if col in df.columns]
        
    if not cols:
        # Fallback to any numeric columns except ID
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric_cols if 'id' not in c.lower()][:2]
        
    X = df[cols].copy()
    return X, cols


def scale_features(X):
    """
    Applies StandardScaler to feature matrix X.
    
    Returns:
        X_scaled (ndarray): Standardized feature matrix (zero mean, unit variance).
        scaler (StandardScaler): The fitted scaler instance.
        X_scaled_df (DataFrame): Standardized features as a pandas DataFrame.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled, scaler, X_scaled_df


def prepare_clustering_data(df, feature_mode='2d'):
    """
    High-level convenience function: selects features, applies scaling,
    and returns both raw and scaled structures alongside metadata.
    """
    df_encoded = encode_categorical_features(df)
    X_raw, feature_cols = select_features(df_encoded, feature_mode=feature_mode)
    X_scaled, scaler, X_scaled_df = scale_features(X_raw)
    
    return {
        'df_processed': df_encoded,
        'X_raw': X_raw,
        'X_scaled': X_scaled,
        'X_scaled_df': X_scaled_df,
        'feature_cols': feature_cols,
        'feature_mode': feature_mode,
        'scaler': scaler
    }


if __name__ == "__main__":
    from dataset import load_dataset
    
    print("=" * 70)
    print("AIML Practical 6: Preprocessing Module Test")
    print("=" * 70)
    df, _, name, _ = load_dataset()
    print(f"Loaded: {name}")
    
    prep_2d = prepare_clustering_data(df, feature_mode='2d')
    print(f"\n2D Features: {prep_2d['feature_cols']}")
    print(f"X_raw shape: {prep_2d['X_raw'].shape}")
    print(f"X_scaled mean: {prep_2d['X_scaled'].mean(axis=0).round(4)}")
    print(f"X_scaled std: {prep_2d['X_scaled'].std(axis=0).round(4)}")
    
    prep_3d = prepare_clustering_data(df, feature_mode='3d')
    print(f"\n3D Features: {prep_3d['feature_cols']}")
    print(f"X_raw shape: {prep_3d['X_raw'].shape}")
    
    print("\n" + get_scaling_rationale())
