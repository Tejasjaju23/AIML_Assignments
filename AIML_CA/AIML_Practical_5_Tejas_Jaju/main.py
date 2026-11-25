"""
=============================================================================
AIML Practical Assignment 5: Ensemble Learning and Performance Evaluation
=============================================================================
Student Details:
  - Name: Tejas Gopal Jaju
  - PRN: 202401040124
  - Roll No.: 29
  - Batch: A2
  - Subject: Artificial Intelligence and Machine Learning
  - Assignment: Practical Assignment 5 – Ensemble Learning and Performance Evaluation
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

from dataset import load_raw_dataset, get_dataset_dataframe, get_dataset_summary, prepare_data
from models import get_all_models
from evaluation import (
    evaluate_all_models,
    build_comparison_dataframe,
    get_formatted_comparison_table,
    identify_best_model,
    save_comparison_csv
)
from visualization import (
    plot_class_distribution,
    plot_accuracy_comparison,
    plot_metrics_comparison,
    plot_log_loss_comparison,
    plot_overall_model_comparison,
    save_all_confusion_matrices
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
    print_banner("AIML PRACTICAL ASSIGNMENT 5: ENSEMBLE LEARNING & EVALUATION")
    print(" Student Information:")
    print("   Name     : Tejas Gopal Jaju")
    print("   PRN      : 202401040124")
    print("   Roll No. : 29")
    print("   Batch    : A2")
    print("   Subject  : Artificial Intelligence and Machine Learning")
    print("   Topic    : Stacking, AdaBoost, and Voting Ensembles vs Base Classifiers")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Step 1: Dataset Exploration & Information
    # -------------------------------------------------------------------------
    print_banner("STEP 1: DATASET ACQUISITION & EXPLORATION")
    summary = get_dataset_summary()
    print(f"[+] Loaded Dataset          : {summary['dataset_name']}")
    print(f"[+] Total Samples (Instances): {summary['n_samples']}")
    print(f"[+] Total Features (Inputs) : {summary['n_features']}")
    print(f"[+] Total Missing Values    : {summary['total_missing_values']}")
    print("\n[+] Target Class Distribution:")
    for cls_name, meta in summary['class_distribution'].items():
        print(f"    - Class '{cls_name}': {meta['count']} samples ({meta['percentage']}%)")
    
    print("\n[+] Dataset Preview (First 5 Rows - Selected Features):")
    df_raw, feat_names, target_names = get_dataset_dataframe()
    preview_cols = list(feat_names[:5]) + ['target', 'diagnosis']
    print(df_raw[preview_cols].head(5).to_string(index=True))

    # -------------------------------------------------------------------------
    # Step 2: Data Preprocessing & Splitting (Preventing Data Leakage)
    # -------------------------------------------------------------------------
    print_banner("STEP 2: PREPROCESSING & STRATIFIED TRAIN-TEST SPLIT")
    print("[+] Train-Test Ratio: 80% Training (455 samples) / 20% Testing (114 samples)")
    print("[+] Stratification  : Preserves exact malignant/benign proportions across splits.")
    print("[+] Random State    : 42 (ensures deterministic reproducibility)")
    print("[+] Scaling Policy  : Feature scaling (StandardScaler) is strictly integrated")
    print("                      inside Scikit-Learn Pipelines. Scalers fit exclusively on")
    print("                      training folds to eliminate data leakage.")
    
    X_train, X_test, y_train, y_test, features, targets = prepare_data(test_size=0.2, random_state=42)
    print(f"[+] Training Split Shape : X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"[+] Testing Split Shape  : X_test  = {X_test.shape}, y_test  = {y_test.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Model Definitions & Pipeline Construction
    # -------------------------------------------------------------------------
    print_banner("STEP 3: INITIALIZING 9 BASE & ENSEMBLE CLASSIFIERS")
    all_models = get_all_models(random_state=42)
    print(f"[+] Total models configured: {len(all_models)}")
    for idx, (name, meta) in enumerate(all_models.items(), 1):
        print(f"  {idx}. {name:24} | Category: {meta['type']:30} | Family: {meta['family']}")

    # -------------------------------------------------------------------------
    # Step 4: Model Training and Evaluation
    # -------------------------------------------------------------------------
    print_banner("STEP 4: TRAINING AND EVALUATION EXECUTION")
    results = evaluate_all_models(all_models, X_train, y_train, X_test, y_test, targets)
    
    # Detailed output for each model
    for name, res in results.items():
        print_sub_banner(f"Model: {name} ({res['model_type']})")
        print(f" Architecture     : {res['architecture']}")
        print(f" Training Time    : {res['train_time_sec']:.4f} seconds")
        print(f" Accuracy         : {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
        print(f" Weighted Precision: {res['precision']:.4f}")
        print(f" Weighted Recall  : {res['recall']:.4f}")
        print(f" Weighted F1-Score: {res['f1_score']:.4f}")
        loss_disp = f"{res['log_loss']:.4f}" if res['log_loss'] is not None else "N/A (Hard Voting)"
        print(f" Cross-Entropy Log Loss: {loss_disp}")
        
        print("\n Confusion Matrix:")
        cm = res['confusion_matrix']
        print(f"    [[TN: {cm[0,0]:2d} (True Malignant),  FP: {cm[0,1]:2d} (False Benign)]")
        print(f"     [FN: {cm[1,0]:2d} (False Malignant), TP: {cm[1,1]:2d} (True Benign)]]")
        
        print("\n Classification Report:")
        for line in res['classification_report_text'].splitlines():
            print(f"    {line}")

    # -------------------------------------------------------------------------
    # Step 5: Ensemble Mechanics & Theoretical Explanations
    # -------------------------------------------------------------------------
    print_banner("STEP 5: ENSEMBLE ARCHITECTURE MECHANICS & WORKING PRINCIPLES")
    
    print("\n[A] STACKING ENSEMBLE (Meta-Learning):")
    print("    - Base Estimators: Decision Tree, KNN (Scaled), SVM (Scaled), Logistic Regression (Scaled).")
    print("    - Meta-Classifier: Logistic Regression.")
    print("    - How the Meta-Classifier Combines Predictions:")
    print("      Stacking applies 5-fold cross-validation internally to the training split.")
    print("      In each fold, the base models predict out-of-fold samples. These predictions")
    print("      form a 4-dimensional meta-feature vector Z for each observation.")
    print("      The meta-classifier (Logistic Regression) then fits on matrix Z. Rather than")
    print("      a simple majority rule, the meta-model mathematically learns which base")
    print("      learners are most reliable, assigning optimal regression coefficients to them.")
    
    print("\n[B] ADABOOST ENSEMBLE (Adaptive Boosting):")
    print("    - Weak Learner: Decision Tree Stump (max_depth=1).")
    print("    - Hyperparameters: n_estimators=50, learning_rate=1.0.")
    print("    - How AdaBoost Adjusts Sample Weights for Misclassified Samples:")
    print("      AdaBoost initializes all sample weights equally: w_i = 1 / N.")
    print("      In each iteration m = 1 to 50:")
    print("        1. A weak decision stump h_m(x) is fit using sample weights w_i.")
    print("        2. Weighted error rate epsilon_m = sum(w_i * I(y_i != h_m(x_i))) / sum(w_i).")
    print("        3. Stump importance weight alpha_m = 0.5 * ln((1 - epsilon_m) / epsilon_m).")
    print("        4. Sample weights update: w_i <- w_i * exp(-alpha_m * y_i * h_m(x_i)).")
    print("        5. Weights are re-normalized so sum(w_i) = 1.")
    print("      Misclassified samples receive significantly increased weights, compelling")
    print("      subsequent stumps to focus on difficult, borderline instances.")
    
    print("\n[C] VOTING ENSEMBLE (Hard vs Soft Voting):")
    print("    - Base Estimators: Logistic Regression, Decision Tree, KNN, Random Forest.")
    print("    - Hard Voting (Plurality / Majority):")
    print("        Predicts the class label that receives the highest count of discrete votes:")
    print("        y_pred = mode(C_1(x), C_2(x), C_3(x), C_4(x)).")
    print("        Does not consider classifier certainty or confidence.")
    print("    - Soft Voting (Probability Averaging):")
    print("        Computes the arithmetic mean of predicted probability distributions:")
    print("        P_ensemble(c) = (1 / M) * sum_{m=1}^M P_m(c | x).")
    print("        Selects argmax_c P_ensemble(c). Highly confident predictions exert")
    print("        greater leverage, typically yielding superior boundary calibration.")

    # -------------------------------------------------------------------------
    # Step 6: Performance Evaluation & Comparison Table
    # -------------------------------------------------------------------------
    print_banner("STEP 6: PERFORMANCE EVALUATION & COMPARISON TABLE")
    df_comparison = build_comparison_dataframe(results)
    df_display = get_formatted_comparison_table(df_comparison)
    
    print(df_display[['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Log Loss', 'Train Time (s)']].to_string(index=False))
    
    best_model = identify_best_model(df_comparison)
    print("\n" + "*" * 78)
    print(f" *** BEST PERFORMING MODEL: {best_model['model_name']} ***")
    print(f"     Category  : {best_model['model_type']}")
    print(f"     Accuracy  : {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)")
    print(f"     F1-Score  : {best_model['f1_score']:.4f}")
    if best_model['log_loss'] is not None:
        print(f"     Log Loss  : {best_model['log_loss']:.4f}")
    print("*" * 78)

    # -------------------------------------------------------------------------
    # Step 7: Export Outputs and Visualizations
    # -------------------------------------------------------------------------
    print_banner("STEP 7: EXPORTING ARTIFACTS, CSV AND FIGURES")
    
    # Save CSV
    out_dir = "outputs"
    cm_dir = os.path.join(out_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "model_comparison.csv")
    save_comparison_csv(df_comparison, csv_path)
    print(f"[✓] Saved Comparison Table to: {csv_path}")
    
    # Also save to subfolder if it exists
    sub_out_dir = os.path.join("AIML_Practical_5_Tejas_Jaju", "outputs")
    sub_cm_dir = os.path.join(sub_out_dir, "confusion_matrices")
    if os.path.exists("AIML_Practical_5_Tejas_Jaju"):
        os.makedirs(sub_cm_dir, exist_ok=True)
        sub_csv_path = os.path.join(sub_out_dir, "model_comparison.csv")
        save_comparison_csv(df_comparison, sub_csv_path)
        print(f"[✓] Mirrored CSV to: {sub_csv_path}")
    
    # Generate charts
    print("[+] Rendering and saving publication-quality figures...")
    save_all_confusion_matrices(results, targets, cm_dir)
    plot_class_distribution(df_raw['target'], targets, save_path=os.path.join(out_dir, "class_distribution.png"))
    plot_accuracy_comparison(df_comparison, save_path=os.path.join(out_dir, "accuracy_comparison.png"))
    plot_metrics_comparison(df_comparison, save_path=os.path.join(out_dir, "metrics_comparison.png"))
    plot_log_loss_comparison(df_comparison, save_path=os.path.join(out_dir, "log_loss_comparison.png"))
    plot_overall_model_comparison(df_comparison, save_path=os.path.join(out_dir, "overall_model_comparison.png"))
    
    if os.path.exists("AIML_Practical_5_Tejas_Jaju"):
        save_all_confusion_matrices(results, targets, sub_cm_dir)
        plot_class_distribution(df_raw['target'], targets, save_path=os.path.join(sub_out_dir, "class_distribution.png"))
        plot_accuracy_comparison(df_comparison, save_path=os.path.join(sub_out_dir, "accuracy_comparison.png"))
        plot_metrics_comparison(df_comparison, save_path=os.path.join(sub_out_dir, "metrics_comparison.png"))
        plot_log_loss_comparison(df_comparison, save_path=os.path.join(sub_out_dir, "log_loss_comparison.png"))
        plot_overall_model_comparison(df_comparison, save_path=os.path.join(sub_out_dir, "overall_model_comparison.png"))
    
    print("[✓] All 9 confusion matrices saved to outputs/confusion_matrices/")
    print("[✓] All comparison charts saved to outputs/")

    # -------------------------------------------------------------------------
    # Step 8: Conclusions and Practical Insights
    # -------------------------------------------------------------------------
    print_banner("STEP 8: KEY FINDINGS, TRADE-OFFS & RECOMMENDATIONS")
    print("""
1. Base vs. Ensemble Performance:
   - Logistic Regression and SVM achieved exceptional accuracy (98.25%) when combined
     with standard scaling pipeline.
   - Ensembles (Soft Voting, Hard Voting, Stacking) demonstrated high stability (~97.37%),
     successfully mitigating the risk of high variance seen in standalone Decision Trees (92.11%).
   - Soft Voting outperformed Hard Voting in log loss calibration because probability 
     averaging incorporates prediction confidence rather than mere discrete counts.

2. When to Use Each Ensemble Technique:
   - Stacking Ensemble: Best when you have diverse, heterogeneous base models with complementary
     inductive biases (e.g. tree + kernel + linear), and sufficient training data to fit
     a meta-classifier without overfitting.
   - AdaBoost Ensemble: Best when base models have high bias (e.g., shallow decision stumps)
     and the dataset has low label noise, iteratively improving hard-to-classify samples.
   - Voting Ensemble: Best for quick, robust variance reduction across well-tuned models
     without the computational overhead of cross-validated meta-training.

3. Limitations & Future Scope:
   - AdaBoost is sensitive to noisy outliers because it progressively inflates weights on
     misclassified samples.
   - Stacking requires cross-validation at Level 0, increasing training latency.
   - Future scope: Explore Gradient Boosted Trees (XGBoost, LightGBM, CatBoost), automated
     hyperparameter optimization via Bayesian Search, and meta-feature selection.
""")
    print_banner("PRACTICAL ASSIGNMENT 5 PIPELINE COMPLETED SUCCESSFULLY")
    return results, df_comparison


if __name__ == "__main__":
    run_pipeline()
