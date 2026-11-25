"""
=============================================================================
AIML Practical Assignment 5: Ensemble Learning and Performance Evaluation
Student: Tejas Gopal Jaju | PRN: 202401040124 | Roll No: 29 | Batch: A2
Module: models.py - Model Definitions, Pipelines, and Ensembles
=============================================================================
"""

import warnings
import os
# Suppress loky CPU count warning on Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)


def get_base_pipeline(model_type, random_state=42):
    """
    Constructs an individual classifier. Classifiers requiring standardized
    feature distributions (Logistic Regression, KNN, SVM) are wrapped in a 
    Pipeline with StandardScaler to strictly prevent data leakage.
    """
    if model_type == 'lr':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000, C=1.0))
        ])
    elif model_type == 'dt':
        return DecisionTreeClassifier(
            random_state=random_state,
            max_depth=5,
            criterion='gini'
        )
    elif model_type == 'knn':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))
        ])
    elif model_type == 'svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=random_state))
        ])
    elif model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=None,
            n_jobs=1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def build_stacking_ensemble(random_state=42):
    """
    Builds a Stacking Classifier.
    Base Estimators:
      - Decision Tree
      - KNN (with StandardScaler Pipeline)
      - SVM (with StandardScaler Pipeline, probability=True)
      - Logistic Regression (with StandardScaler Pipeline)
    Meta-Classifier:
      - Logistic Regression (final_estimator)
    
    Cross-validation (cv=5) is used internally to generate out-of-fold 
    meta-features on the training set, avoiding target leakage to the meta-model.
    """
    base_estimators = [
        ('dt', DecisionTreeClassifier(random_state=random_state, max_depth=5)),
        ('knn', Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ])),
        ('svm', Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=random_state))
        ])),
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ]))
    ]
    
    meta_classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_classifier,
        cv=5,
        passthrough=False,
        n_jobs=1
    )
    return stacking_clf


def build_adaboost_ensemble(random_state=42):
    """
    Builds an AdaBoost Classifier.
    Weak Learner:
      - Decision Tree with max_depth=1 (Decision Stump)
    Hyperparameters:
      - n_estimators = 50
      - learning_rate = 1.0
    
    Handles compatibility across modern Scikit-Learn (estimator=...)
    and legacy versions (base_estimator=...).
    """
    stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    
    try:
        adaboost_clf = AdaBoostClassifier(
            estimator=stump,
            n_estimators=50,
            learning_rate=1.0,
            random_state=random_state
        )
    except TypeError:
        # Fallback for scikit-learn < 1.2
        adaboost_clf = AdaBoostClassifier(
            base_estimator=stump,
            n_estimators=50,
            learning_rate=1.0,
            random_state=random_state
        )
    
    return adaboost_clf


def build_voting_ensemble(voting_type='hard', random_state=42):
    """
    Builds a Voting Classifier.
    Base Estimators:
      - Logistic Regression (Pipeline with StandardScaler)
      - Decision Tree
      - KNN (Pipeline with StandardScaler)
      - Random Forest
    
    Parameters:
      - voting_type: 'hard' (majority label count) or 'soft' (argmax of average class probabilities)
    """
    base_estimators = [
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ])),
        ('dt', DecisionTreeClassifier(random_state=random_state, max_depth=5)),
        ('knn', Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ])),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1))
    ]
    
    voting_clf = VotingClassifier(
        estimators=base_estimators,
        voting=voting_type,
        n_jobs=1
    )
    return voting_clf


def get_all_models(random_state=42):
    """
    Returns an ordered dictionary containing all 9 models (5 Base + 4 Ensembles)
    and their theoretical descriptions, architectures, and explanations.
    """
    models = {
        'Logistic Regression': {
            'model': get_base_pipeline('lr', random_state=random_state),
            'type': 'Individual Base Classifier',
            'family': 'Linear Model',
            'description': 'A linear model utilizing the sigmoid link function to estimate class probabilities via maximum likelihood estimation with L2 regularization.',
            'architecture': 'StandardScaler -> LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")',
            'supports_proba': True,
            'theory': (
                "Logistic Regression applies the logistic sigmoid function sigma(z) = 1 / (1 + exp(-z)) "
                "to a linear combination of input features. Features are standardized so coefficients "
                "are regularized uniformly."
            )
        },
        'Decision Tree': {
            'model': get_base_pipeline('dt', random_state=random_state),
            'type': 'Individual Base Classifier',
            'family': 'Tree-based Model',
            'description': 'A non-parametric tree model that recursively splits features to maximize Gini impurity reduction or information gain.',
            'architecture': 'DecisionTreeClassifier(criterion="gini", max_depth=5)',
            'supports_proba': True,
            'theory': (
                "Decision Trees partition the feature space orthogonally using threshold rules. At each node, "
                "the feature and split point minimizing Gini Impurity = 1 - sum(p_i^2) are selected. "
                "A max_depth of 5 prevents overfitting on the 30 continuous features."
            )
        },
        'KNN': {
            'model': get_base_pipeline('knn', random_state=random_state),
            'type': 'Individual Base Classifier',
            'family': 'Instance-based (Lazy Learner)',
            'description': 'An instance-based learner assigning test instances to the majority class among their 5 nearest neighbors in Euclidean space.',
            'architecture': 'StandardScaler -> KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)',
            'supports_proba': True,
            'theory': (
                "K-Nearest Neighbors computes the Minkowski (Euclidean) distance to all training samples. "
                "Standardization is critical because features with large raw numerical variance "
                "(e.g., area vs smoothness) would otherwise dominate the distance metric."
            )
        },
        'SVM': {
            'model': get_base_pipeline('svm', random_state=random_state),
            'type': 'Individual Base Classifier',
            'family': 'Kernel Method',
            'description': 'Support Vector Machine projecting inputs into infinite-dimensional Hilbert space via the Radial Basis Function (RBF) kernel to construct a maximum-margin hyperplane.',
            'architecture': 'StandardScaler -> SVC(kernel="rbf", C=1.0, probability=True)',
            'supports_proba': True,
            'theory': (
                "SVM solves a convex quadratic optimization problem to maximize the geometric margin 2 / ||w|| "
                "subject to slack penalties (C=1.0). The RBF kernel handles non-linear boundaries. Platt scaling "
                "(probability=True) fits a logistic sigmoid over raw decision values for probabilistic calibration."
            )
        },
        'Random Forest': {
            'model': get_base_pipeline('rf', random_state=random_state),
            'type': 'Individual Base Classifier',
            'family': 'Bagging Ensemble',
            'description': 'An ensemble of 100 decorrelated decision trees built on bootstrap samples with random feature subsampling at every split.',
            'architecture': 'RandomForestClassifier(n_estimators=100, criterion="gini", n_jobs=-1)',
            'supports_proba': True,
            'theory': (
                "Random Forest applies Bootstrap Aggregating (Bagging) combined with the Random Subspace method. "
                "By training 100 deep trees on bootstrap replicates and selecting random subsets of sqrt(p) "
                "features per split, tree variance is substantially reduced without increasing bias."
            )
        },
        'Stacking Classifier': {
            'model': build_stacking_ensemble(random_state=random_state),
            'type': 'Ensemble Model (Stacking)',
            'family': 'Meta-Learning Ensemble',
            'description': 'Heterogeneous stacking combining Decision Tree, KNN, SVM, and Logistic Regression via a Logistic Regression meta-classifier.',
            'architecture': 'Level 0: [DT, KNN+Scaler, SVM+Scaler, LR+Scaler] -> Level 1 (Meta): LogisticRegression(C=1.0)',
            'supports_proba': True,
            'theory': (
                "Stacking (Wolpert, 1992) uses 5-fold cross-validation on the training set to generate out-of-fold "
                "predictions for each base learner. These predictions form a new meta-feature matrix Z. The meta-classifier "
                "(Logistic Regression) learns optimal blending weights for each base model's strengths, discovering "
                "which classifier is most trustworthy across regions of the feature space."
            )
        },
        'AdaBoost Classifier': {
            'model': build_adaboost_ensemble(random_state=random_state),
            'type': 'Ensemble Model (Boosting)',
            'family': 'Sequential Adaptive Boosting',
            'description': 'Sequential ensemble boosting 50 Decision Stumps (depth=1) with adaptive sample re-weighting.',
            'architecture': 'AdaBoostClassifier(estimator=DecisionTree(max_depth=1), n_estimators=50, learning_rate=1.0)',
            'supports_proba': True,
            'theory': (
                "AdaBoost (Freund & Schapire, 1997) initializes equal weights w_i = 1/N. In each round m, a weak learner "
                "(decision stump) is fit. Its weighted error epsilon_m is calculated, and model weight alpha_m = 0.5 * ln((1 - eps)/eps) "
                "is assigned. Correctly classified samples have their weights decreased, while misclassified samples "
                "have weights increased: w_i <- w_i * exp(-alpha * y_i * h_m(x_i)). Successive stumps are thus forced to focus "
                "on borderline and hard-to-classify samples."
            )
        },
        'Hard Voting Classifier': {
            'model': build_voting_ensemble('hard', random_state=random_state),
            'type': 'Ensemble Model (Voting)',
            'family': 'Plurality / Majority Voting',
            'description': 'Ensemble combining Logistic Regression, Decision Tree, KNN, and Random Forest by majority class label counting.',
            'architecture': 'VotingClassifier(estimators=[LR, DT, KNN, RF], voting="hard")',
            'supports_proba': False,
            'theory': (
                "Hard Voting collects crisp class predictions y_hat_m in {0, 1} from all 4 diverse base estimators. "
                "The ensemble outputs mode(y_hat_1, ..., y_hat_4). It treats each classifier's vote equally, regardless of "
                "confidence. Note that hard voting does not output continuous probabilities, so log loss is not applicable."
            )
        },
        'Soft Voting Classifier': {
            'model': build_voting_ensemble('soft', random_state=random_state),
            'type': 'Ensemble Model (Voting)',
            'family': 'Probability-Averaging Voting',
            'description': 'Ensemble combining Logistic Regression, Decision Tree, KNN, and Random Forest by averaging predicted class probabilities.',
            'architecture': 'VotingClassifier(estimators=[LR, DT, KNN, RF], voting="soft")',
            'supports_proba': True,
            'theory': (
                "Soft Voting averages the predicted class probability vectors P(y=c|x) across all 4 base models: "
                "P_ensemble(c) = (1/M) * sum(P_m(c)). The class maximizing this average probability is selected. "
                "Soft voting typically outperforms hard voting because high-confidence predictions contribute more "
                "influence than borderline, low-confidence predictions."
            )
        }
    }
    return models


if __name__ == "__main__":
    print("=" * 70)
    print("AIML Practical 5: Models & Ensembles Definition Test")
    print("=" * 70)
    models = get_all_models()
    print(f"Total Models Defined: {len(models)}")
    for name, meta in models.items():
        print(f"\n[+] {name} ({meta['type']})")
        print(f"    Architecture: {meta['architecture']}")
        print(f"    Supports Proba: {meta['supports_proba']}")
    print("\nModels module successfully defined and verified.")
