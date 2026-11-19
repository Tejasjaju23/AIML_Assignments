"""
Data Preprocessing Module for AIML Practical Assignment 4.
Handles dataset loading, exploratory data summary, train-test splitting,
and proper feature scaling preventing data leakage.

Student: Tejas Gopal Jaju (PRN: 202401040124, Roll: 29, Batch: A2)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():
    """
    Load the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn.
    
    Returns:
        tuple: (df, X, y, feature_names, target_names, raw_data)
            - df: Complete pandas DataFrame with features and target
            - X: Feature DataFrame (30 numerical features)
            - y: Target Series (0 = malignant, 1 = benign)
            - feature_names: List of feature names
            - target_names: List of class names ['malignant', 'benign']
            - raw_data: Original Bunch object from scikit-learn
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    
    # Map target integer values to class names for clear visualization
    # 0 = malignant (cancerous), 1 = benign (non-cancerous)
    df['class_name'] = df['target'].map({0: 'malignant', 1: 'benign'})
    
    X = data.data.copy()
    y = data.target.copy()
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    
    return df, X, y, feature_names, target_names, data


def get_dataset_summary(df, target_col='target'):
    """
    Generate comprehensive statistical summary of the dataset.
    
    Parameters:
        df: Pandas DataFrame containing dataset
        target_col: Name of target column
        
    Returns:
        dict: Summary metrics including shape, missing values, class distribution
    """
    total_samples, total_features = df.shape[0], df.shape[1] - 2  # excluding target and class_name
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    class_counts = df['class_name'].value_counts()
    class_percentages = (df['class_name'].value_counts(normalize=True) * 100).round(2)
    
    summary = {
        'total_rows': total_samples,
        'total_columns': df.shape[1],
        'total_features': total_features,
        'missing_values_count': total_missing,
        'has_missing_values': total_missing > 0,
        'class_counts': class_counts.to_dict(),
        'class_percentages': class_percentages.to_dict(),
        'target_distribution': pd.DataFrame({
            'Count': class_counts,
            'Percentage (%)': class_percentages
        })
    }
    return summary


def prepare_data(X, y, test_size=0.20, random_state=42):
    """
    Split the dataset into training and testing sets (80:20 ratio),
    and apply StandardScaler strictly fitted on training data to prevent data leakage.
    
    Important Preprocessing Rules:
    - Distance-based algorithms (SVM, KNN) require standardized features (mean=0, std=1).
    - Tree-based algorithms (Decision Tree) are scale-invariant and operate directly on raw features.
    - Gaussian Naive Bayes computes per-feature mean and variance independently, so original features are ideal.
    - Fitting the scaler ONLY on X_train prevents test set information from leaking into training.
    
    Parameters:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of test split (default 0.20 = 20%)
        random_state: Seed for reproducible results (default 42)
        
    Returns:
        dict: Processed datasets and scaler
    """
    # 1. Stratified Train-Test Split (80:20 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # 2. Fit StandardScaler strictly on X_train to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # 3. Transform training and test feature sets
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }


def get_dataset_rationale():
    """
    Returns an academic explanation of why Breast Cancer Wisconsin dataset
    is well-suited for classification model evaluation.
    """
    return (
        "Why Breast Cancer Wisconsin Dataset is Suitable for Classification:\n"
        "1. Real-World Clinical Relevance: Distinguishing benign (non-cancerous) from malignant "
        "(cancerous) tumors is a high-stakes binary classification problem where precision and recall "
        "have direct clinical significance (minimizing false negatives prevents untreated cancer).\n"
        "2. Clean & High-Quality Attributes: Contains 569 instances with 30 continuous, real-valued numerical "
        "features derived from cell nuclei images, with zero missing values.\n"
        "3. Well-Balanced Class Distribution: 357 benign (62.74%) and 212 malignant (37.26%) cases offer a "
        "natural clinical balance without severe skew, allowing objective metric comparison.\n"
        "4. Diverse Feature Correlations: Contains both highly correlated features (e.g., radius, perimeter, area) "
        "and non-linear boundary characteristics, allowing effective stress-testing of different algorithmic "
        "assumptions (e.g., Naive Bayes independence vs. SVM margin boundary vs. Decision Tree orthogonal splits)."
    )
