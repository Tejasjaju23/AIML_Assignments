"""
=============================================================================
AIML Practical Assignment 5: Ensemble Learning and Performance Evaluation
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: evaluation.py - Model Training, Metric Computation, and Comparison
=============================================================================
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)


def evaluate_single_model(model_entry, X_train, y_train, X_test, y_test, model_name, target_names):
    """
    Fits a single model on training data, evaluates predictions and probabilities
    on testing data, and computes comprehensive classification metrics.
    
    Args:
        model_entry (dict): Dictionary from models.py with 'model' and metadata.
        X_train, y_train: Training split.
        X_test, y_test: Testing split.
        model_name (str): Identifier name for the model.
        target_names (list): Class label names (e.g., ['malignant', 'benign']).
        
    Returns:
        dict: Complete evaluation payload containing metrics, confusion matrix, 
              classification reports, training duration, and predictions.
    """
    model = model_entry['model']
    
    # 1. Train the model and measure latency
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_duration = time.perf_counter() - start_time
    
    # 2. Generate predictions on test dataset
    y_pred = model.predict(X_test)
    
    # 3. Probabilistic predictions for Log Loss (if supported)
    y_prob = None
    loss_value = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            loss_value = float(log_loss(y_test, y_prob))
        except (AttributeError, NotImplementedError):
            y_prob = None
            loss_value = None
    
    # 4. Compute core classification metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_weighted = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    
    # Binary class specific metrics (Positive class = Benign / 1)
    precision_binary = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    recall_binary = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_binary = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    
    # 5. Confusion matrix & Classification report
    cm = confusion_matrix(y_test, y_pred)
    clf_report_dict = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    
    result = {
        'model_name': model_name,
        'model_type': model_entry.get('type', 'Unknown'),
        'family': model_entry.get('family', 'Unknown'),
        'fitted_model': model,
        'accuracy': accuracy,
        'precision': precision_weighted,
        'recall': recall_weighted,
        'f1_score': f1_weighted,
        'precision_binary': precision_binary,
        'recall_binary': recall_binary,
        'f1_binary': f1_binary,
        'log_loss': loss_value,
        'confusion_matrix': cm,
        'classification_report_dict': clf_report_dict,
        'classification_report_text': clf_report_text,
        'train_time_sec': train_duration,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'description': model_entry.get('description', ''),
        'architecture': model_entry.get('architecture', ''),
        'theory': model_entry.get('theory', '')
    }
    return result


def evaluate_all_models(models_dict, X_train, y_train, X_test, y_test, target_names):
    """
    Sequentially trains and evaluates all models provided in models_dict.
    """
    results = {}
    for name, entry in models_dict.items():
        res = evaluate_single_model(
            model_entry=entry,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            target_names=target_names
        )
        results[name] = res
    return results


def build_comparison_dataframe(results):
    """
    Constructs a clean comparison DataFrame containing:
    | Model | Accuracy | Precision | Recall | F1-Score | Log Loss |
    
    Sorts models by accuracy in descending order.
    """
    rows = []
    for name, res in results.items():
        loss_val = res['log_loss']
        rows.append({
            'Model': name,
            'Type': res['model_type'],
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1_score'],
            'Log Loss': loss_val if loss_val is not None else np.nan,
            'Train Time (s)': round(res['train_time_sec'], 4)
        })
    
    df = pd.DataFrame(rows)
    # Sort models by accuracy descending
    df_sorted = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    return df_sorted


def get_formatted_comparison_table(df_comparison):
    """
    Returns a copy of the comparison DataFrame formatted with rounded decimals
    and string 'N/A' for models that do not produce probability outputs.
    """
    df_display = df_comparison.copy()
    df_display['Accuracy'] = df_display['Accuracy'].map(lambda x: f"{x:.4f}")
    df_display['Precision'] = df_display['Precision'].map(lambda x: f"{x:.4f}")
    df_display['Recall'] = df_display['Recall'].map(lambda x: f"{x:.4f}")
    df_display['F1-Score'] = df_display['F1-Score'].map(lambda x: f"{x:.4f}")
    df_display['Log Loss'] = df_display['Log Loss'].map(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A (Hard Voting)"
    )
    return df_display


def identify_best_model(df_comparison):
    """
    Identifies the best-performing model based on highest Accuracy,
    breaking ties using highest F1-Score and lowest Log Loss.
    """
    # Sort primarily by Accuracy, then F1-Score, then ascending Log Loss
    sorted_df = df_comparison.sort_values(
        by=['Accuracy', 'F1-Score', 'Log Loss'],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    
    best_row = sorted_df.iloc[0]
    return {
        'model_name': best_row['Model'],
        'model_type': best_row['Type'],
        'accuracy': float(best_row['Accuracy']),
        'precision': float(best_row['Precision']),
        'recall': float(best_row['Recall']),
        'f1_score': float(best_row['F1-Score']),
        'log_loss': float(best_row['Log Loss']) if pd.notnull(best_row['Log Loss']) else None
    }


def save_comparison_csv(df_comparison, filepath):
    """
    Saves the numerical comparison DataFrame to CSV.
    """
    df_comparison.to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    from dataset import prepare_data
    from models import get_all_models
    
    print("=" * 70)
    print("AIML Practical 5: Evaluation Module Verification")
    print("=" * 70)
    X_tr, X_te, y_tr, y_te, feats, tgts = prepare_data()
    models = get_all_models()
    
    print(f"Training and evaluating {len(models)} models...")
    res = evaluate_all_models(models, X_tr, y_tr, X_te, y_te, tgts)
    
    df_comp = build_comparison_dataframe(res)
    df_fmt = get_formatted_comparison_table(df_comp)
    print("\nModel Comparison Table:")
    print(df_fmt[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Log Loss']].to_string(index=False))
    
    best = identify_best_model(df_comp)
    print(f"\nBest Performing Model: {best['model_name']} with Accuracy: {best['accuracy']:.4f}")
