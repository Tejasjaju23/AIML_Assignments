"""
Visualization Module for AIML Practical Assignment 4.
Generates:
1. Individual & combined metric comparison bar charts (Accuracy, Precision, Recall, F1, Log Loss)
2. Confusion matrix heatmaps (individual and 2x2 grid for all four models)
3. Decision tree graphical representation
4. Target class distribution
5. Train vs Test accuracy comparison

Student: Tejas Gopal Jaju (PRN: 202401040124, Roll: 29, Batch: A2)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import plot_tree


# Set standard aesthetic configuration
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def plot_individual_metric_bar(results_df, metric_col, title=None, color='#2563eb'):
    """
    Generate an individual vertical bar chart for a single metric across all 4 algorithms.
    
    Parameters:
        results_df: DataFrame with model results
        metric_col: Name of metric column
        title: Custom title
        color: Hex color string
        
    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
    
    bars = ax.bar(
        results_df['Algorithm'],
        results_df[metric_col],
        color=color,
        edgecolor='black',
        width=0.55,
        alpha=0.85
    )
    
    # Add value annotations on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.4f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),  # 4 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom',
            fontweight='bold', fontsize=10
        )
        
    chart_title = title if title else f"Algorithm Comparison - {metric_col}"
    ax.set_title(chart_title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel(metric_col, fontsize=11, fontweight='bold')
    ax.set_xlabel("Classification Algorithm", fontsize=11, fontweight='bold')
    
    # Adjust y-limits for aesthetic padding
    max_val = results_df[metric_col].max()
    min_val = results_df[metric_col].min()
    if metric_col == 'Log Loss':
        ax.set_ylim(0, max_val * 1.25)
    else:
        ax.set_ylim(max(0, min_val - 0.15), 1.05)
        
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    return fig


def plot_metric_comparison(results_df, metric='Accuracy'):
    """
    Convenience wrapper to plot individual metric comparison.
    """
    colors = {
        'Accuracy': '#2563eb',
        'Precision': '#059669',
        'Recall': '#d97706',
        'F1-Score': '#7c3aed',
        'Log Loss': '#dc2626'
    }
    color = colors.get(metric, '#2563eb')
    return plot_individual_metric_bar(results_df, metric, color=color)


def plot_all_metrics_comparison(results_df):
    """
    Generate a 2x3 comprehensive dashboard grid of bar charts for all 5 metrics.
    
    Parameters:
        results_df: DataFrame containing model evaluations
        
    Returns:
        matplotlib.figure.Figure
    """
    metrics = [
        ('Accuracy', '#2563eb', True),
        ('Precision', '#059669', True),
        ('Recall', '#d97706', True),
        ('F1-Score', '#7c3aed', True),
        ('Log Loss', '#dc2626', False)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=100)
    axes = axes.flatten()
    
    for idx, (metric, color, bounded) in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(
            results_df['Algorithm'],
            results_df[metric],
            color=color,
            edgecolor='black',
            width=0.5,
            alpha=0.85
        )
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold', fontsize=9
            )
            
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
        
        if bounded:
            min_val = results_df[metric].min()
            ax.set_ylim(max(0, min_val - 0.12), 1.05)
        else:
            max_val = results_df[metric].max()
            ax.set_ylim(0, max_val * 1.3)
            
    # Hide the 6th unused subplot in the 2x3 grid and add a summary legend/note
    axes[5].axis('off')
    summary_text = (
        "Evaluation Metrics Legend:\n\n"
        "• Accuracy: (TP + TN) / Total\n"
        "• Precision: TP / (TP + FP) [Weighted Avg]\n"
        "• Recall: TP / (TP + FN) [Weighted Avg]\n"
        "• F1-Score: 2 * (Prec * Rec) / (Prec + Rec)\n"
        "• Log Loss: -1/N * Σ log(P_correct)\n\n"
        "Note: For Accuracy, Precision, Recall, and\n"
        "F1-Score, higher is better (max = 1.0).\n"
        "For Log Loss, lower is better (min = 0.0)."
    )
    axes[5].text(
        0.1, 0.5, summary_text,
        fontsize=11,
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9)
    )
    
    plt.suptitle("Comprehensive Performance Comparison Across Classification Algorithms",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", ax=None):
    """
    Plot single confusion matrix heatmap with count and percentage annotations.
    
    Parameters:
        cm: Confusion matrix 2D array
        class_names: List of class labels
        title: Chart title
        ax: Optional matplotlib axis
        
    Returns:
        matplotlib.figure.Figure (if ax is None) or ax
    """
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=(5, 4.2), dpi=100)
    else:
        fig = ax.figure
        
    # Calculate percentages for cell text
    cm_sum = np.sum(cm)
    cm_perc = cm / cm_sum * 100
    
    annot_labels = np.array([
        [f"{cm[0, 0]}\n({cm_perc[0, 0]:.1f}%)", f"{cm[0, 1]}\n({cm_perc[0, 1]:.1f}%)"],
        [f"{cm[1, 0]}\n({cm_perc[1, 0]:.1f}%)", f"{cm[1, 1]}\n({cm_perc[1, 1]:.1f}%)"]
    ])
    
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap='Blues',
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=1.2,
        linecolor='white',
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        ax=ax
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Actual Label', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    if create_fig:
        plt.tight_layout()
        return fig
    return ax


def plot_all_confusion_matrices(detailed_results, class_names):
    """
    Create a 2x2 grid containing the confusion matrices for all 4 models.
    
    Parameters:
        detailed_results: Dictionary with model evaluations
        class_names: Class labels ['malignant', 'benign']
        
    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    axes = axes.flatten()
    
    models = list(detailed_results.keys())
    cmaps = ['Blues', 'Greens', 'Purples', 'Oranges']
    
    for idx, name in enumerate(models):
        ax = axes[idx]
        cm = detailed_results[name]['confusion_matrix']
        acc = detailed_results[name]['accuracy'] * 100
        
        cm_sum = np.sum(cm)
        cm_perc = cm / cm_sum * 100
        annot_labels = np.array([
            [f"{cm[0, 0]}\n({cm_perc[0, 0]:.1f}%)", f"{cm[0, 1]}\n({cm_perc[0, 1]:.1f}%)"],
            [f"{cm[1, 0]}\n({cm_perc[1, 0]:.1f}%)", f"{cm[1, 1]}\n({cm_perc[1, 1]:.1f}%)"]
        ])
        
        sns.heatmap(
            cm,
            annot=annot_labels,
            fmt="",
            cmap=cmaps[idx % len(cmaps)],
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=1.5,
            linecolor='white',
            annot_kws={"fontsize": 11, "fontweight": "bold"},
            ax=ax
        )
        ax.set_title(f"{name}\n(Testing Accuracy: {acc:.2f}%)", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Actual Label', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        
    plt.suptitle("Confusion Matrix Heatmaps for All Classification Algorithms",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    return fig


def plot_decision_tree_vis(dt_model, feature_names, class_names, max_depth=3):
    """
    Generate clean, publication-ready visual representation of the Decision Tree.
    
    Parameters:
        dt_model: Fitted DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class labels
        max_depth: Maximum tree depth to visualize for readability
        
    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(18, 9), dpi=120)
    
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10,
        ax=ax,
        precision=2,
        impurity=True,
        proportion=False
    )
    
    ax.set_title(f"Decision Tree Visualization (Max Display Depth = {max_depth})",
                 fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


def plot_class_distribution(y, target_names):
    """
    Generate a bar chart showing the target class distribution.
    
    Parameters:
        y: Target Series
        target_names: List of class names
        
    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    counts = y.value_counts().sort_index()
    labels = [f"{target_names[i]} (Class {i})" for i in counts.index]
    
    colors = ['#ef4444', '#10b981']  # red for malignant, green for benign
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', width=0.45)
    
    total = len(y)
    for bar in bars:
        height = bar.get_height()
        perc = (height / total) * 100
        ax.annotate(
            f"{height} ({perc:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha='center', va='bottom',
            fontweight='bold', fontsize=11
        )
        
    ax.set_title("Target Class Distribution (Wisconsin Breast Cancer)", fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    return fig


def plot_train_vs_test_accuracy(detailed_results):
    """
    Generate side-by-side grouped bar chart comparing Training vs Testing Accuracy
    to evaluate overfitting and underfitting.
    
    Parameters:
        detailed_results: Dictionary of model evaluation results
        
    Returns:
        matplotlib.figure.Figure
    """
    algorithms = list(detailed_results.keys())
    train_acc = [detailed_results[m]['training_accuracy'] for m in algorithms]
    test_acc = [detailed_results[m]['accuracy'] for m in algorithms]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=100)
    rects1 = ax.bar(x - width/2, train_acc, width, label='Training Accuracy', color='#3b82f6', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, test_acc, width, label='Testing Accuracy', color='#10b981', edgecolor='black', alpha=0.9)
    
    # Value annotations
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    ax.set_ylabel('Accuracy Score', fontsize=11, fontweight='bold')
    ax.set_title('Training vs Testing Accuracy (Overfitting vs Generalization Analysis)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha='right', fontsize=10)
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(0.85, 1.05)
    plt.tight_layout()
    return fig
