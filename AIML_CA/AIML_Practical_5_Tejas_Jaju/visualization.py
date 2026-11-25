"""
=============================================================================
AIML Practical Assignment 5: Ensemble Learning and Performance Evaluation
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: visualization.py - Matplotlib and Seaborn Visualizations
=============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    sns.set_theme(style="whitegrid")


def sanitize_filename(name):
    """Converts a model name into a clean filesystem-friendly slug."""
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
    return re.sub(r'_+', '_', clean).strip('_')


def plot_class_distribution(y, target_names, save_path=None):
    """
    Plots the target class distribution (Malignant vs Benign) with both count
    and percentage annotations.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    counts = pd.Series(y).value_counts().sort_index()
    labels = [target_names[i].capitalize() for i in counts.index]
    colors = ['#e63946', '#2a9d8f']
    
    # 1. Bar Chart
    bars = ax1.bar(labels, counts.values, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    ax1.set_title("Class Counts", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Number of Samples", fontsize=11)
    ax1.set_ylim(0, max(counts.values) * 1.15)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height} ({height/len(y)*100:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Donut Chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor='black', linewidth=1.2)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
        at.set_color('white')
    ax2.set_title("Class Proportions", fontsize=13, fontweight='bold', pad=12)
    
    fig.suptitle("Breast Cancer Wisconsin Dataset - Target Class Distribution",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_confusion_matrix(cm, model_name, target_names, save_path=None):
    """
    Renders a publication-grade confusion matrix heatmap with cell counts,
    percentages, and quadrant annotations (TN, FP, FN, TP).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    cm_sum = np.sum(cm)
    group_counts = [f"{val:0.0f}" for val in cm.flatten()]
    group_percentages = [f"{val/cm_sum:.1%}" for val in cm.flatten()]
    group_labels = ["True Malignant (TN)", "False Benign (FP)", "False Malignant (FN)", "True Benign (TP)"]
    
    box_labels = [f"{l}\n{c} ({p})" for l, c, p in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(2, 2)
    
    sns.heatmap(
        cm,
        annot=box_labels,
        fmt="",
        cmap='Blues',
        cbar=True,
        square=True,
        ax=ax,
        linewidths=1.5,
        linecolor='white',
        annot_kws={"fontsize": 10, "fontweight": 'medium'}
    )
    
    tick_labels = [name.capitalize() for name in target_names]
    ax.set_xticklabels(tick_labels, fontsize=10, fontweight='bold')
    ax.set_yticklabels(tick_labels, fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold', pad=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def save_all_confusion_matrices(results, target_names, output_dir):
    """
    Iterates through all evaluated model results and exports individual confusion
    matrix figures to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}
    for name, res in results.items():
        slug = sanitize_filename(name)
        file_path = os.path.join(output_dir, f"{slug}_cm.png")
        fig = plot_confusion_matrix(res['confusion_matrix'], name, target_names, save_path=file_path)
        plt.close(fig)
        saved_paths[name] = file_path
    return saved_paths


def plot_accuracy_comparison(df_comparison, save_path=None):
    """
    Generates a horizontal bar chart of all models sorted by Accuracy,
    distinguishing Individual Models from Ensemble Models by color.
    """
    df_sorted = df_comparison.sort_values(by='Accuracy', ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette: Individual base models vs Ensemble models
    colors = []
    for m_type in df_sorted['Type']:
        if 'Ensemble' in m_type:
            colors.append('#2a9d8f')  # Teal for ensembles
        else:
            colors.append('#457b9d')  # Blue for base classifiers
            
    bars = ax.barh(df_sorted['Model'], df_sorted['Accuracy'], color=colors, height=0.6, edgecolor='black', linewidth=1)
    
    ax.set_xlim(0.85, 1.02)
    ax.set_xlabel("Accuracy Score", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title("Model Accuracy Comparison (Ensembles vs Base Classifiers)", fontsize=14, fontweight='bold', pad=14)
    
    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width * 100:.2f}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2a9d8f', edgecolor='black', label='Ensemble Models'),
        Patch(facecolor='#457b9d', edgecolor='black', label='Base Classifiers')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_metrics_comparison(df_comparison, save_path=None):
    """
    Renders a grouped bar chart comparing Precision, Recall, and F1-Score
    across all 9 evaluated models.
    """
    df_plot = df_comparison.copy()
    
    models = df_plot['Model'].tolist()
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    b1 = ax.bar(x - width, df_plot['Precision'], width, label='Precision', color='#1d3557', edgecolor='black')
    b2 = ax.bar(x, df_plot['Recall'], width, label='Recall', color='#457b9d', edgecolor='black')
    b3 = ax.bar(x + width, df_plot['F1-Score'], width, label='F1-Score', color='#e76f51', edgecolor='black')
    
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Performance Metrics Comparison (Precision, Recall, F1-Score)", fontsize=14, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc='lower left', frameon=True, fontsize=11)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_log_loss_comparison(df_comparison, save_path=None):
    """
    Plots Log Loss across models that output probabilities (lower is better).
    """
    df_valid = df_comparison.dropna(subset=['Log Loss']).sort_values(by='Log Loss', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_valid['Model'], df_valid['Log Loss'], color='#e63946', height=0.55, edgecolor='black')
    
    ax.set_xlabel("Cross-Entropy Log Loss (Lower is Better)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title("Log Loss Comparison Across Probabilistic Models", fontsize=13, fontweight='bold', pad=12)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.4f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_overall_model_comparison(df_comparison, save_path=None):
    """
    Creates a comprehensive 2x2 multi-panel comparison dashboard:
    Panel 1: Accuracy comparison
    Panel 2: F1-Score ranking
    Panel 3: Log Loss comparison
    Panel 4: Training Latency / Speed comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy
    df_acc = df_comparison.sort_values(by='Accuracy', ascending=True)
    axes[0, 0].barh(df_acc['Model'], df_acc['Accuracy'] * 100, color='#2a9d8f', edgecolor='black')
    axes[0, 0].set_xlim(85, 102)
    axes[0, 0].set_title("Model Accuracy (%)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Accuracy (%)")
    
    # 2. F1-Score
    df_f1 = df_comparison.sort_values(by='F1-Score', ascending=True)
    axes[0, 1].barh(df_f1['Model'], df_f1['F1-Score'], color='#457b9d', edgecolor='black')
    axes[0, 1].set_xlim(0.85, 1.02)
    axes[0, 1].set_title("F1-Score (Weighted)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("F1-Score")
    
    # 3. Log Loss
    df_loss = df_comparison.dropna(subset=['Log Loss']).sort_values(by='Log Loss', ascending=False)
    axes[1, 0].barh(df_loss['Model'], df_loss['Log Loss'], color='#e76f51', edgecolor='black')
    axes[1, 0].set_title("Log Loss (Lower is Better)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Log Loss")
    
    # 4. Training Time
    df_time = df_comparison.sort_values(by='Train Time (s)', ascending=True)
    axes[1, 1].barh(df_time['Model'], df_time['Train Time (s)'], color='#f4a261', edgecolor='black')
    axes[1, 1].set_title("Training Duration (Seconds)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Time (s)")
    
    fig.suptitle("AIML Practical 5: Comprehensive Model Performance Overview",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    from dataset import prepare_data, get_dataset_dataframe
    from models import get_all_models
    from evaluation import evaluate_all_models, build_comparison_dataframe
    
    print("=" * 70)
    print("AIML Practical 5: Visualization Module Test")
    print("=" * 70)
    
    df, feat_names, target_names = get_dataset_dataframe()
    X_tr, X_te, y_tr, y_te, feats, tgts = prepare_data()
    models = get_all_models()
    results = evaluate_all_models(models, X_tr, y_tr, X_te, y_te, tgts)
    df_comp = build_comparison_dataframe(results)
    
    out_dir = "outputs"
    cm_dir = os.path.join(out_dir, "confusion_matrices")
    
    print(f"Generating confusion matrices in '{cm_dir}'...")
    save_all_confusion_matrices(results, tgts, cm_dir)
    
    print("Generating comparison charts in 'outputs'...")
    plot_class_distribution(df['target'], tgts, save_path=os.path.join(out_dir, "class_distribution.png"))
    plot_accuracy_comparison(df_comp, save_path=os.path.join(out_dir, "accuracy_comparison.png"))
    plot_metrics_comparison(df_comp, save_path=os.path.join(out_dir, "metrics_comparison.png"))
    plot_log_loss_comparison(df_comp, save_path=os.path.join(out_dir, "log_loss_comparison.png"))
    plot_overall_model_comparison(df_comp, save_path=os.path.join(out_dir, "overall_model_comparison.png"))
    
    print("Visualization module successfully verified and charts generated.")
