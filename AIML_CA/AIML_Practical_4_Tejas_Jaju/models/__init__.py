"""
Models package for AIML Practical Assignment 4.
Contains classification model definitions and training logic.
"""
from .classification_models import get_models_dict, train_models, train_single_model

__all__ = ["get_models_dict", "train_models", "train_single_model"]
