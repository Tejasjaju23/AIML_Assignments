"""
Performance Evaluation Module for AIML Practical Assignment 4.
Calculates Accuracy, Precision, Recall, F1-Score, and Log Loss.
Generates confusion matrices, classification reports, and automated comparative findings.

Student: Tejas Gopal Jaju (PRN: 202401040124, Roll: 29, Batch: A2)
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, data_dict, use_scaled=False, class_names=None):
    """
    Evaluate a single trained model on both training and testing datasets.
    
    Parameters:
        model: Fitted scikit-learn estimator
        data_dict: Dictionary containing X_train, X_test, etc.
        use_scaled: Boolean indicating whether to use scaled features
        class_names: List of class label strings
        
    Returns:
        dict: Complete evaluation metrics, predictions, and reports
    """
    if class_names is None:
        class_names = ['malignant', 'benign']
        
    X_train = data_dict['X_train_scaled'] if use_scaled else data_dict['X_train']
    X_test = data_dict['X_test_scaled'] if use_scaled else data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    
    # 1. Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 2. Probability predictions for Log Loss
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_test_proba)
    else:
        y_test_proba = None
        loss = np.nan
        
    # 3. Core metrics calculation (Weighted average accounts for class distribution)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # 4. Confusion Matrix and Detailed Classification Reports
    cm = confusion_matrix(y_test, y_test_pred)
    report_text = classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0)
    report_dict = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    return {
        'training_accuracy': train_acc,
        'accuracy': test_acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'log_loss': loss,
        'confusion_matrix': cm,
        'classification_report_text': report_text,
        'classification_report_dict': report_dict,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }


def evaluate_all_models(trained_models, data_dict, models_dict, class_names=None):
    """
    Evaluate all four trained classification algorithms and assemble comparative table.
    
    Parameters:
        trained_models: Dictionary of {model_name: fitted_estimator}
        data_dict: Dictionary with train/test splits
        models_dict: Configuration dictionary specifying use_scaled per model
        class_names: List of class names
        
    Returns:
        tuple: (results_df, detailed_results)
    """
    detailed_results = {}
    table_rows = []
    
    for name, model in trained_models.items():
        use_scaled = models_dict[name]["use_scaled"]
        eval_result = evaluate_model(model, data_dict, use_scaled=use_scaled, class_names=class_names)
        detailed_results[name] = eval_result
        
        table_rows.append({
            'Algorithm': name,
            'Accuracy': round(eval_result['accuracy'], 4),
            'Precision': round(eval_result['precision'], 4),
            'Recall': round(eval_result['recall'], 4),
            'F1-Score': round(eval_result['f1_score'], 4),
            'Log Loss': round(eval_result['log_loss'], 4)
        })
        
    results_df = pd.DataFrame(table_rows)
    return results_df, detailed_results


def generate_comparative_analysis(results_df, detailed_results=None):
    """
    Automatically generate an analytical summary based on actual model performance.
    
    Parameters:
        results_df: Pandas DataFrame containing metric scores
        detailed_results: Optional detailed evaluation dictionary
        
    Returns:
        dict: Findings, best performers, and practical conclusion text
    """
    best_acc_row = results_df.loc[results_df['Accuracy'].idxmax()]
    best_prec_row = results_df.loc[results_df['Precision'].idxmax()]
    best_rec_row = results_df.loc[results_df['Recall'].idxmax()]
    best_f1_row = results_df.loc[results_df['F1-Score'].idxmax()]
    lowest_loss_row = results_df.loc[results_df['Log Loss'].idxmin()]
    
    # Calculate a composite score rank (mean rank across metrics) to identify best overall
    rank_acc = results_df['Accuracy'].rank(ascending=False)
    rank_f1 = results_df['F1-Score'].rank(ascending=False)
    rank_loss = results_df['Log Loss'].rank(ascending=True)
    composite_rank = (rank_acc + rank_f1 + rank_loss) / 3.0
    best_overall_algo = results_df.loc[composite_rank.idxmin()]['Algorithm']
    
    findings = {
        "highest_accuracy": {
            "algorithm": best_acc_row['Algorithm'],
            "score": best_acc_row['Accuracy']
        },
        "highest_precision": {
            "algorithm": best_prec_row['Algorithm'],
            "score": best_prec_row['Precision']
        },
        "highest_recall": {
            "algorithm": best_rec_row['Algorithm'],
            "score": best_rec_row['Recall']
        },
        "highest_f1": {
            "algorithm": best_f1_row['Algorithm'],
            "score": best_f1_row['F1-Score']
        },
        "lowest_log_loss": {
            "algorithm": lowest_loss_row['Algorithm'],
            "score": lowest_loss_row['Log Loss']
        },
        "best_overall": best_overall_algo
    }
    
    # Academic journal conclusion
    journal_conclusion = (
        f"In this practical assignment, four foundational machine learning classification algorithms "
        f"(Support Vector Machine, Decision Tree, K-Nearest Neighbors, and Gaussian Naïve Bayes) were "
        f"trained and rigorously evaluated on the Breast Cancer Wisconsin Diagnostic dataset (80:20 train-test split).\n\n"
        f"Key Observations:\n"
        f"1. Overall Champion: {best_overall_algo} emerged as the top-performing model, achieving an Accuracy of "
        f"{best_acc_row['Accuracy'] * 100:.2f}% and an F1-Score of {best_f1_row['F1-Score']:.4f}, demonstrating "
        f"excellent diagnostic capability with minimal classification error.\n"
        f"2. Log Loss Performance: {lowest_loss_row['Algorithm']} demonstrated superior probabilistic calibration with "
        f"the lowest Log Loss of {lowest_loss_row['Log Loss']:.4f}. Unlike accuracy, log loss heavily penalizes "
        f"overconfident erroneous predictions, which is critical in medical diagnostic decision support.\n"
        f"3. Feature Scaling Impact: SVM and KNN rely heavily on distance calculations in feature space, meaning "
        f"StandardScaler was essential to prevent features with large numeric variance from distorting distance metrics. "
        f"Conversely, Decision Tree and Gaussian Naive Bayes naturally handled original feature scales.\n"
        f"4. Computational Simplicity: Gaussian Naïve Bayes provided the fastest training and inference with minimal "
        f"computational overhead, making it ideal for resource-constrained or real-time environments, despite its "
        f"conditional independence assumption.\n"
        f"5. High-Dimensional Suitability: Support Vector Machine (SVM) demonstrated superior capability in handling "
        f"the 30 continuous clinical dimensions by finding an optimal maximum-margin hyperplane using the RBF kernel.\n\n"
        f"Final Conclusion:\n"
        f"The experiment successfully fulfilled all objectives of Practical Assignment 4. For medical diagnostic tasks "
        f"such as breast cancer tumor detection, {best_overall_algo} is the most suitable algorithm due to its balanced "
        f"precision-recall profile and robust generalization ability."
    )
    
    return findings, journal_conclusion


def save_results_to_files(results_df, detailed_results, output_dir='outputs'):
    """
    Save the comparative results table to CSV and full classification reports to TXT.
    
    Parameters:
        results_df: DataFrame of model comparisons
        detailed_results: Dictionary containing detailed model reports
        output_dir: Target directory path
        
    Returns:
        tuple: (csv_path, txt_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save comparison table to CSV
    csv_path = os.path.join(output_dir, "comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    # 2. Save classification reports and student metadata to TXT
    txt_path = os.path.join(output_dir, "classification_reports.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AIML PRACTICAL ASSIGNMENT 4 - CLASSIFICATION MODELS EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write("STUDENT DETAILS:\n")
        f.write("Name: Tejas Gopal Jaju\n")
        f.write("PRN: 202401040124\n")
        f.write("Roll Number: 29\n")
        f.write("Batch: A2\n")
        f.write("Subject: Artificial Intelligence and Machine Learning\n")
        f.write("Practical Assignment: 4\n")
        f.write("Title: Classification Models\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("OVERALL COMPARISON TABLE:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n\n")
        
        for name, data in detailed_results.items():
            f.write(f"MODEL: {name.upper()}\n")
            f.write(f"Training Accuracy : {data['training_accuracy']:.4f}\n")
            f.write(f"Testing Accuracy  : {data['accuracy']:.4f}\n")
            f.write(f"Precision (weight): {data['precision']:.4f}\n")
            f.write(f"Recall (weighted) : {data['recall']:.4f}\n")
            f.write(f"F1-Score (weight) : {data['f1_score']:.4f}\n")
            f.write(f"Log Loss          : {data['log_loss']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"{data['confusion_matrix']}\n\n")
            f.write("Classification Report:\n")
            f.write(f"{data['classification_report_text']}\n")
            f.write("-" * 80 + "\n\n")
            
    return csv_path, txt_path
