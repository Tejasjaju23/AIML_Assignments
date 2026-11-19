"""
Visualizations package for AIML Practical Assignment 4.
Contains plotting utilities for metrics, confusion matrices, and decision tree.
"""
from .plots import (
    plot_metric_comparison,
    plot_all_metrics_comparison,
    plot_confusion_matrix,
    plot_all_confusion_matrices,
    plot_decision_tree_vis,
    plot_class_distribution,
    plot_train_vs_test_accuracy
)

__all__ = [
    "plot_metric_comparison",
    "plot_all_metrics_comparison",
    "plot_confusion_matrix",
    "plot_all_confusion_matrices",
    "plot_decision_tree_vis",
    "plot_class_distribution",
    "plot_train_vs_test_accuracy"
]
