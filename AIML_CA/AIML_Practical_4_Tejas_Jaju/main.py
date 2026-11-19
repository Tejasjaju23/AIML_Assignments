"""
AIML Practical Assignment 4 – Classification Models
Main Execution Script

Student Details:
- Name: Tejas Gopal Jaju
- PRN: 202401040124
- Roll Number: 29
- Batch: A2
- Subject: Artificial Intelligence and Machine Learning
- Practical Assignment: 4
- Title: Classification Models
"""

import os
import sys

# Add current workspace directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ensure local writable cache directories and clean execution on Windows
os.environ.setdefault('MPLCONFIGDIR', os.path.join(current_dir, '.matplotlib'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import matplotlib.pyplot as plt
import pandas as pd

from utils.preprocessing import (
    load_dataset,
    prepare_data,
    get_dataset_summary,
    get_dataset_rationale
)
from models.classification_models import (
    get_models_dict,
    train_models
)
from utils.evaluation import (
    evaluate_all_models,
    generate_comparative_analysis,
    save_results_to_files
)
from visualizations.plots import (
    plot_metric_comparison,
    plot_all_metrics_comparison,
    plot_all_confusion_matrices,
    plot_decision_tree_vis,
    plot_class_distribution,
    plot_train_vs_test_accuracy
)


def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def print_student_details():
    print_banner("Student Details & Practical Information")
    print("  Name                 : Tejas Gopal Jaju")
    print("  PRN                  : 202401040124")
    print("  Roll Number          : 29")
    print("  Batch                : A2")
    print("  Subject              : Artificial Intelligence and Machine Learning (AIML)")
    print("  Practical Assignment : 4")
    print("  Assignment Title     : Classification Models")
    print("=" * 80)


def main():
    # 1. Print Student Header
    print_student_details()

    # 2. Dataset Loading & Inspection
    print_banner("Step 1: Dataset Loading & Inspection")
    df, X, y, feature_names, target_names, raw_data = load_dataset()
    summary = get_dataset_summary(df)

    print(f"Dataset Name            : Breast Cancer Wisconsin (Diagnostic) Dataset")
    print(f"Total Number of Rows    : {summary['total_rows']}")
    print(f"Total Number of Columns : {summary['total_columns']}")
    print(f"Number of Features      : {summary['total_features']} (all continuous numerical)")
    print(f"Missing Values Found    : {summary['missing_values_count']} (Clean dataset with zero missing values)")
    print(f"Target Classes          : 0 -> {target_names[0]} (Malignant), 1 -> {target_names[1]} (Benign)")
    print("\nTarget Class Distribution:")
    print(summary['target_distribution'].to_string())

    print("\nFirst 5 Records (Features subset):")
    print(X.iloc[:5, :5].to_string())
    print("\nTotal 30 Features in Dataset:")
    for i, col in enumerate(feature_names, 1):
        print(f"  {i:2d}. {col}", end="\n" if i % 3 == 0 or i == len(feature_names) else "\t")

    print("\n" + "-" * 80)
    print(get_dataset_rationale())

    # 3. Data Preprocessing
    print_banner("Step 2: Data Preprocessing & Leakage Prevention")
    print("Splitting dataset into Training (80%) and Testing (20%) sets with random_state=42...")
    data_dict = prepare_data(X, y, test_size=0.20, random_state=42)

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    print(f"Training feature set shape : {X_train.shape} (80% of samples)")
    print(f"Testing feature set shape  : {X_test.shape} (20% of samples)")
    print("StandardScaler fitted STRICTLY on X_train to prevent data leakage into X_test.")
    print("• Scaled features assigned to: Support Vector Machine (SVM) & K-Nearest Neighbors (KNN)")
    print("• Original features assigned to: Decision Tree & Gaussian Naïve Bayes")

    # 4. Model Training
    print_banner("Step 3: Training All Four Classification Models")
    models_dict = get_models_dict()
    for name, info in models_dict.items():
        scaled_text = "StandardScaler (Scaled)" if info["use_scaled"] else "Original Raw Features"
        print(f"Training {name:<23} | Feature Representation: {scaled_text}")

    trained_models = train_models(data_dict, models_dict)
    print("All four classification models trained successfully!")

    # 5. Performance Evaluation
    print_banner("Step 4: Performance Evaluation (5 Metrics)")
    print("Calculating Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, and Log Loss...")
    results_df, detailed_results = evaluate_all_models(
        trained_models, data_dict, models_dict, class_names=target_names
    )

    print("\n" + "-" * 80)
    print("PERFORMANCE COMPARISON TABLE:")
    print("-" * 80)
    print(results_df.to_string(index=False))
    print("-" * 80)
    print("Note: Log loss evaluates the quality of predicted probabilities (predict_proba).")
    print("      Lower log loss indicates superior probabilistic confidence (0 is ideal).")

    # Display individual model metrics and classification reports
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORTS & CONFUSION MATRICES:")
    print("=" * 80)
    for name in models_dict.keys():
        res = detailed_results[name]
        print(f"\n>>> Algorithm: {name}")
        print(f"  Training Accuracy : {res['training_accuracy']:.4f} ({res['training_accuracy']*100:.2f}%)")
        print(f"  Testing Accuracy  : {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
        print(f"  Weighted Precision: {res['precision']:.4f}")
        print(f"  Weighted Recall   : {res['recall']:.4f}")
        print(f"  Weighted F1-Score : {res['f1_score']:.4f}")
        print(f"  Log Loss          : {res['log_loss']:.4f}")
        print("  Confusion Matrix [[TN, FP], [FN, TP]]:")
        print(f"    {res['confusion_matrix'][0]}")
        print(f"    {res['confusion_matrix'][1]}")
        print("  Classification Report:")
        for line in res['classification_report_text'].splitlines():
            print(f"    {line}")

    # 6. Save Results
    print_banner("Step 5: Saving Reports & Exporting Visualizations")
    output_dir = os.path.join(current_dir, "outputs")
    csv_file, txt_file = save_results_to_files(results_df, detailed_results, output_dir=output_dir)
    print(f"✓ Comparison table saved to : {csv_file}")
    print(f"✓ Full report saved to      : {txt_file}")

    # Generate and save all plots
    # 1. Individual metric plots
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Log Loss']:
        fig_m = plot_metric_comparison(results_df, metric=metric)
        safe_name = metric.lower().replace("-", "_").replace(" ", "_")
        fig_m.savefig(os.path.join(output_dir, f"metric_{safe_name}.png"))
        plt.close(fig_m)

    # 2. Combined metrics dashboard
    fig_dash = plot_all_metrics_comparison(results_df)
    fig_dash.savefig(os.path.join(output_dir, "metrics_comparison_dashboard.png"))
    plt.close(fig_dash)

    # 3. Confusion matrices (2x2 grid)
    fig_cm = plot_all_confusion_matrices(detailed_results, target_names)
    fig_cm.savefig(os.path.join(output_dir, "all_confusion_matrices.png"))
    plt.close(fig_cm)

    # 4. Decision Tree Visualization
    dt_estimator = trained_models["Decision Tree"]
    fig_dt = plot_decision_tree_vis(dt_estimator, feature_names, target_names, max_depth=3)
    fig_dt.savefig(os.path.join(output_dir, "decision_tree_visualization.png"))
    plt.close(fig_dt)

    # 5. Class Distribution
    fig_dist = plot_class_distribution(y, target_names)
    fig_dist.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close(fig_dist)

    # 6. Train vs Test Accuracy
    fig_tt = plot_train_vs_test_accuracy(detailed_results)
    fig_tt.savefig(os.path.join(output_dir, "train_vs_test_accuracy.png"))
    plt.close(fig_tt)

    print("✓ All 10 visualization figures generated and saved in outputs/ directory.")

    # 7. Automated Comparative Findings & Practical Journal Conclusion
    print_banner("Step 6: Automated Comparative Findings & Discussion")
    findings, journal_conclusion = generate_comparative_analysis(results_df, detailed_results)

    print(f"1. Highest Accuracy Model : {findings['highest_accuracy']['algorithm']} ({findings['highest_accuracy']['score']:.4f})")
    print(f"2. Highest Precision Model: {findings['highest_precision']['algorithm']} ({findings['highest_precision']['score']:.4f})")
    print(f"3. Highest Recall Model   : {findings['highest_recall']['algorithm']} ({findings['highest_recall']['score']:.4f})")
    print(f"4. Highest F1-Score Model : {findings['highest_f1']['algorithm']} ({findings['highest_f1']['score']:.4f})")
    print(f"5. Lowest Log Loss Model  : {findings['lowest_log_loss']['algorithm']} ({findings['lowest_log_loss']['score']:.4f})")
    print(f"6. Best Overall Model     : {findings['best_overall']}")

    print("\nALGORITHM SUITABILITY ANALYSIS:")
    print("• Computational Simplicity: Gaussian Naïve Bayes has O(N*D) training complexity and closed-form parameter estimation.")
    print("• High-Dimensional Data   : Support Vector Machine (SVM) maximizes geometric margins, excelling with many dimensions.")
    print("• Small Datasets          : Naïve Bayes & SVM perform well because Naïve Bayes has low variance and SVM generalizes effectively.")
    print("• Interpretability        : Decision Tree provides intuitive, visualizable if-then decision rules.")

    print_banner("Step 7: Final Practical Journal Conclusion")
    print(journal_conclusion)

    print_banner("Execution Complete")
    print("To launch the interactive Streamlit Web Application, run:")
    print("  python -m streamlit run app.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
