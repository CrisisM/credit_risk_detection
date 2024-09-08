import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier

def AdaBoostModel(train_df, val_df, predictors, target, random_state=2024, algorithm='SAMME.R', learning_rate=0.8, n_estimators=100):
    """
    A wrapped function for training and predicting using AdaBoost classifier.

    Parameters:
    - train_df: The training dataset as a DataFrame.
    - val_df: The validation dataset as a DataFrame.
    - predictors: List of feature column names.
    - target: Name of the target column.
    - random_state: Seed for randomness to ensure reproducibility.
    - algorithm: Algorithm type, 'SAMME' or 'SAMME.R' (default is 'SAMME.R').
    - learning_rate: Contribution of each tree to the final result.
    - n_estimators: Number of base learners (usually decision trees).

    Returns:
    - preds: Prediction results on the validation set.
    - clf: The trained AdaBoost model.
    """
    
    # Initialize the AdaBoost model
    clf = AdaBoostClassifier(random_state=random_state,
                             algorithm=algorithm,
                             learning_rate=learning_rate,
                             n_estimators=n_estimators)
    
    # Train the model
    clf.fit(train_df[predictors], train_df[target].values)
    
    # Predict on the validation set
    preds = clf.predict(val_df[predictors])
    
    return preds, clf  # Ensure both values are returned


# %%
from catboost import CatBoostClassifier

def CatBoostModel(train_df, val_df, predictors, target, iterations=500, learning_rate=0.02, depth=12, eval_metric='AUC',
                  random_seed=2024, bagging_temperature=0.2, od_type='Iter', metric_period=50, od_wait=100, verbose=True):
    """
    A wrapped function for training and predicting using CatBoost classifier.

    Parameters:
    - train_df: The training dataset as a DataFrame.
    - val_df: The validation dataset as a DataFrame.
    - predictors: List of feature column names.
    - target: Name of the target column.
    - iterations: Number of training iterations (number of decision trees).
    - learning_rate: Learning rate.
    - depth: Maximum depth of each decision tree.
    - eval_metric: Evaluation metric, default is 'AUC'.
    - random_seed: Seed for randomness to ensure reproducibility.
    - bagging_temperature: Controls the diversity of subsamples.
    - od_type: Type of overfitting detection, default is 'Iter'.
    - metric_period: Frequency of metric output.
    - od_wait: Number of iterations to wait before stopping for overfitting.
    - verbose: Controls the verbosity of the training process.

    Returns:
    - preds: Prediction results on the validation set.
    - clf: The trained CatBoost model.
    """
    
    # Initialize the CatBoost model
    clf = CatBoostClassifier(iterations=iterations,
                             learning_rate=learning_rate,
                             depth=depth,
                             eval_metric=eval_metric,
                             random_seed=random_seed,
                             bagging_temperature=bagging_temperature,
                             od_type=od_type,
                             metric_period=metric_period,
                             od_wait=od_wait)
    
    # Train the model
    clf.fit(train_df[predictors], train_df[target].values, verbose=verbose)
    
    # Predict on the validation set
    preds = clf.predict(val_df[predictors])
    
    return preds, clf


# %%
import xgboost as xgb

def XGBoostModel(train_df, val_df, predictors, target, num_boost_round=500, early_stopping_rounds=50, verbose_eval=100,
                 eta=0.039, max_depth=3, subsample=0.8, colsample_bytree=0.9, eval_metric='auc', random_state=2024):
    """
    A wrapped function for training and predicting using XGBoost classifier.

    Parameters:
    - train_df: The training dataset as a DataFrame.
    - val_df: The validation dataset as a DataFrame.
    - predictors: List of feature column names.
    - target: Name of the target column.
    - num_boost_round: Total number of boosting rounds.
    - early_stopping_rounds: Number of rounds to wait before stopping early if no improvement.
    - verbose_eval: Controls the frequency of output during training.
    - eta: Learning rate, controls the impact of each tree on the final result.
    - max_depth: Maximum depth of each tree.
    - subsample: Proportion of training samples used for each tree.
    - colsample_bytree: Proportion of features used for each tree.
    - eval_metric: Evaluation metric, commonly 'auc' for binary classification tasks.
    - random_state: Seed for randomness to ensure reproducibility.

    Returns:
    - preds: Prediction results on the validation set.
    - model: The trained XGBoost model.
    - model_params: Dictionary of the model's parameters.
    """
    
    # Prepare training and validation datasets
    dtrain = xgb.DMatrix(train_df[predictors], label=train_df[target].values)
    dvalid = xgb.DMatrix(val_df[predictors], label=val_df[target].values)
    
    # Monitor evaluation metrics during training
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eta': eta,
        'verbosity': 1,  # Use verbosity instead of silent
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'eval_metric': eval_metric,
        'random_state': random_state
    }
    
    # Train the model
    model = xgb.train(params, 
                      dtrain, 
                      num_boost_round=num_boost_round, 
                      evals=watchlist, 
                      early_stopping_rounds=early_stopping_rounds, 
                      maximize=True if eval_metric == 'auc' else False, 
                      verbose_eval=verbose_eval)
    
    # Check if the model has the best_ntree_limit attribute
    if hasattr(model, 'best_ntree_limit'):
        preds = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    else:
        preds = model.predict(dvalid)
    
    # Model parameters
    model_params = params.copy()
    model_params.update({
        'num_boost_round': num_boost_round,
        'early_stopping_rounds': early_stopping_rounds,
        'verbose_eval': verbose_eval
    })
    
    return preds, model, model_params


# %%
import lightgbm as lgb
import gc

def LightGBMModel(train_df, val_df, predictors, target, categorical_features=None, params=None, num_boost_round=500, early_stopping_rounds=50, verbose_eval=100):
    """
    A wrapped function for training and predicting using LightGBM classifier.

    Parameters:
    - train_df: The training dataset as a DataFrame.
    - val_df: The validation dataset as a DataFrame.
    - predictors: List of feature column names.
    - target: Name of the target column.
    - categorical_features: List of categorical feature column names.
    - params: Dictionary of LightGBM model parameters.
    - num_boost_round: Total number of boosting rounds.
    - early_stopping_rounds: Number of rounds to wait before early stopping.
    - verbose_eval: Controls the frequency of output during training.

    Returns:
    - preds: Prediction results on the validation set.
    - model: The trained LightGBM model.
    - model_params: Dictionary of the model's parameters.
    - final_auc_train: Final AUC on the training set.
    - final_auc_valid: Final AUC on the validation set.
    """
    # If no params are provided, use default parameters
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',  # Can also be set to 'auc'
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': 2024,
            'verbosity': -1  # Set to -1 to suppress warnings
        }
    
    # Prepare training and validation datasets
    dtrain = lgb.Dataset(train_df[predictors].values, 
                         label=train_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical_features)
    
    dvalid = lgb.Dataset(val_df[predictors].values,
                         label=val_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical_features)
    
    # Define a dictionary to store evaluation results
    evals_results = {}
    
    # Define callbacks
    from lightgbm import early_stopping, log_evaluation, record_evaluation

    callbacks = [
        early_stopping(stopping_rounds=early_stopping_rounds),
        log_evaluation(period=verbose_eval),
        record_evaluation(evals_results)
    ]
    
    # Train the model
    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train', 'valid'], 
                      callbacks=callbacks, 
                      num_boost_round=num_boost_round)
    
    # Make predictions
    preds = model.predict(val_df[predictors].values)
    
    # Model parameters
    model_params = params.copy()
    model_params.update({
        'num_boost_round': num_boost_round,
        'early_stopping_rounds': early_stopping_rounds,
        'verbose_eval': verbose_eval
    })

    # Delete dvalid object to free memory
    del dvalid

    # Perform garbage collection to free memory
    gc.collect()

    return preds, model, model_params
