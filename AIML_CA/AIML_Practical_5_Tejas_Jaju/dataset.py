"""
=============================================================================
AIML Practical Assignment 5: Ensemble Learning and Performance Evaluation
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: dataset.py - Dataset Loading, Preprocessing, and Splitting
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_raw_dataset():
    """
    Load the Breast Cancer Wisconsin dataset from Scikit-Learn.
    Returns:
        bunch: The raw sklearn Bunch object containing data, target, and metadata.
    """
    cancer = load_breast_cancer(as_frame=True)
    return cancer


def get_dataset_dataframe():
    """
    Returns a unified pandas DataFrame with all features and the target column.
    """
    cancer = load_raw_dataset()
    df = cancer.frame.copy()
    # Add human-readable target label column
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    return df, cancer.feature_names, cancer.target_names


def get_dataset_summary():
    """
    Computes and returns comprehensive metadata and summary statistics
    for the Breast Cancer Wisconsin dataset.
    """
    cancer = load_raw_dataset()
    df, feature_names, target_names = get_dataset_dataframe()
    
    # Class distribution
    class_counts = df['target'].value_counts().to_dict()
    # Map numerical 0 -> Malignant, 1 -> Benign
    class_dist = {
        target_names[0].capitalize(): {
            'count': class_counts.get(0, 0),
            'percentage': round((class_counts.get(0, 0) / len(df)) * 100, 2)
        },
        target_names[1].capitalize(): {
            'count': class_counts.get(1, 0),
            'percentage': round((class_counts.get(1, 0) / len(df)) * 100, 2)
        }
    }
    
    # Missing values check
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())
    
    summary = {
        'dataset_name': 'Breast Cancer Wisconsin (Diagnostic)',
        'n_samples': int(df.shape[0]),
        'n_features': int(len(feature_names)),
        'feature_names': list(feature_names),
        'target_names': list(target_names),
        'class_distribution': class_dist,
        'total_missing_values': total_missing,
        'missing_per_feature': missing_series.to_dict(),
        'first_five_rows': df.head(5),
        'statistical_summary': df.describe().T
    }
    return summary


def prepare_data(test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing partitions using an 80:20 ratio.
    Stratified sampling ensures the proportion of malignant and benign cases
    remains identical in both train and test splits.
    
    Feature scaling is NOT performed globally here to prevent data leakage.
    Instead, StandardScaler is wrapped inside a Pipeline for models that require
    scaling, ensuring the scaler is fit ONLY on the training split.
    
    Returns:
        X_train (DataFrame): Training feature matrix (80%)
        X_test (DataFrame): Testing feature matrix (20%)
        y_train (Series): Training target labels (80%)
        y_test (Series): Testing target labels (20%)
        feature_names (list): List of 30 feature names
        target_names (list): List of class names ('malignant', 'benign')
    """
    cancer = load_raw_dataset()
    X = cancer.data
    y = cancer.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, list(cancer.feature_names), list(cancer.target_names)


if __name__ == "__main__":
    print("=" * 70)
    print("AIML Practical 5: Dataset Exploration Test")
    print("=" * 70)
    summary = get_dataset_summary()
    print(f"Dataset: {summary['dataset_name']}")
    print(f"Total Samples: {summary['n_samples']}")
    print(f"Total Features: {summary['n_features']}")
    print(f"Missing Values: {summary['total_missing_values']}")
    print("\nClass Distribution:")
    for cls, val in summary['class_distribution'].items():
        print(f"  - {cls}: {val['count']} samples ({val['percentage']}%)")
    
    X_tr, X_te, y_tr, y_te, feats, tgts = prepare_data()
    print(f"\nTrain Split Shape: {X_tr.shape}, Test Split Shape: {X_te.shape}")
    print("Dataset module successfully loaded and verified.")
