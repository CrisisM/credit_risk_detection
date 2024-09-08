# credit_risk/__init__.py

# Import from the preprocess module
from .preprocess import preprocess_data, separate_features_and_target, preprocess_numeric_features, preprocess_categorical_features, build_preprocessor, apply_preprocessor, random_oversample, smote_oversample

# Import from the virtualization module
from .virtualization import plot_feature_distribution, plot_correlation_heatmap, plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_default_distribution, plot_default_amount_distribution

# Import from the evaluation module
from .evaluation import evaluate_model_classification, plot_confusion_matrix, classification_report

# Import from the ML_model module
from .ML_model import model_gs, run_all_models

# Import from the boost_model module
from .boost_model import AdaBoostModel, CatBoostModel, XGBoostModel, LightGBMModel

# Define __all__ to make package imports clearer
__all__ = [
    # preprocess.py
    'preprocess_data',
    'separate_features_and_target',
    'preprocess_numeric_features',
    'preprocess_categorical_features',
    'build_preprocessor',
    'apply_preprocessor',
    'random_oversample',
    'smote_oversample',
    
    # virtualization.py
    'plot_feature_distribution',
    'plot_correlation_heatmap',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_feature_importance',
    'plot_default_distribution',
    'plot_default_amount_distribution',
    
    # evaluation.py
    'evaluate_model_classification',
    'classification_report',
    
    # ML_model.py
    'model_gs',
    'run_all_models',

    # boost_model.py
    'AdaBoostModel',
    'CatBoostModel',
    'XGBoostModel',
    'LightGBMModel'
]
