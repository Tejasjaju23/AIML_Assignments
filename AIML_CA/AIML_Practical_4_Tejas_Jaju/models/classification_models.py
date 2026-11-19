"""
Classification Models Module for AIML Practical Assignment 4.
Implements:
1. Support Vector Machine (SVM) - RBF Kernel
2. Decision Tree Classifier - Gini Criterion
3. K-Nearest Neighbors (KNN) - K=5
4. Gaussian Naive Bayes Classifier

Student: Tejas Gopal Jaju (PRN: 202401040124, Roll: 29, Batch: A2)
"""

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def get_models_dict():
    """
    Returns dictionary containing instances of the 4 required classification models,
    along with configuration details, scaling requirements, and theoretical notes.
    
    Returns:
        dict: Model configuration dictionary
    """
    models = {
        "Support Vector Machine": {
            "estimator": SVC(
                kernel='rbf',
                probability=True,  # Mandatory for predict_proba & log loss
                random_state=42
            ),
            "use_scaled": True,
            "description": "SVM with Radial Basis Function (RBF) kernel projects features into high-dimensional space to construct an optimal separating hyperplane.",
            "scaling_reason": "Distance/Margin based: SVM maximizes margin between support vectors using Euclidean/radial distances. Unscaled features with large ranges would dominate the margin.",
            "key_assumptions": "Non-linear separable boundaries can be mapped to linear hyperplanes in infinite-dimensional Hilbert space via the kernel trick.",
            "strengths": "Effective in high-dimensional spaces, robust against overfitting when C is regularized, memory efficient (uses only support vectors).",
            "limitations": "Computationally intensive for large datasets, sensitive to hyperparameter tuning (C, gamma), requires feature scaling."
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                criterion='gini',
                random_state=42
            ),
            "use_scaled": False,
            "description": "Non-parametric supervised model that partitions feature space recursively into axis-aligned rectangular regions using Gini Impurity splits.",
            "scaling_reason": "Scale-invariant: Splitting thresholds are calculated independently on individual features, so monotonic transformations do not affect splits.",
            "key_assumptions": "Features can be recursively partitioned by orthogonal axis-aligned decision boundaries without parametric distribution assumptions.",
            "strengths": "Highly interpretable and visualizable, handles non-linear relationships natively, scale-invariant, requires minimal data preparation.",
            "limitations": "Prone to overfitting (high variance) if depth is unconstrained, unstable to small perturbations in training data."
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='minkowski',
                p=2  # Euclidean distance
            ),
            "use_scaled": True,
            "description": "Instance-based, non-parametric lazy learner classifying samples based on majority vote of the 5 closest training instances in feature space.",
            "scaling_reason": "Distance-based: Minkowski/Euclidean metric treats differences equally across all dimensions; unscaled features distort neighborhood distances.",
            "key_assumptions": "Instances that are close in feature space share similar class labels (locality assumption).",
            "strengths": "Simple and intuitive, no explicit training phase (lazy learning), naturally handles multi-class and non-linear decision boundaries.",
            "limitations": "Computationally expensive during inference O(N*D), sensitive to irrelevant features, sensitive to choice of K (bias-variance trade-off)."
        },
        "Naïve Bayes": {
            "estimator": GaussianNB(),
            "use_scaled": False,
            "description": "Probabilistic classifier based on Bayes' Theorem assuming Gaussian distribution for continuous features and conditional independence between features.",
            "scaling_reason": "Independent feature likelihoods: GaussianNB computes mean and variance per feature separately. Linear scaling does not change probability density ratios.",
            "key_assumptions": "Feature Independence: Assumes all 30 features are mutually independent given the class label: P(X1, X2, ... | Y) = Π P(Xi | Y).",
            "strengths": "Extremely fast training and prediction, works well with small training sets, immune to curse of dimensionality, robust probabilistic outputs.",
            "limitations": "The conditional independence assumption is violated when features correlate (e.g. tumor radius, perimeter, and area), which can skew probabilities."
        }
    }
    return models


def train_single_model(model_name, model_info, data_dict):
    """
    Train a single classification model with the appropriate feature representation.
    
    Parameters:
        model_name: Name string of the model
        model_info: Config dictionary for the model
        data_dict: Dictionary containing X_train, X_train_scaled, y_train, etc.
        
    Returns:
        Fitted scikit-learn estimator
    """
    estimator = model_info["estimator"]
    use_scaled = model_info["use_scaled"]
    
    # Select appropriate training features
    X_train_to_use = data_dict['X_train_scaled'] if use_scaled else data_dict['X_train']
    y_train = data_dict['y_train']
    
    estimator.fit(X_train_to_use, y_train)
    return estimator


def train_models(data_dict, models_dict=None):
    """
    Train all four classification models on their respective feature sets.
    
    Parameters:
        data_dict: Dictionary from prepare_data()
        models_dict: Optional custom models dictionary (defaults to get_models_dict())
        
    Returns:
        dict: {model_name: fitted_estimator}
    """
    if models_dict is None:
        models_dict = get_models_dict()
        
    trained_models = {}
    for name, info in models_dict.items():
        trained_models[name] = train_single_model(name, info, data_dict)
        
    return trained_models
