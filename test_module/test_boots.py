# test_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import model functions from your package
from credit_risk_detection.boost_model import AdaBoostModel, CatBoostModel, XGBoostModel, LightGBMModel

# Create mock data
def create_mock_data():
    np.random.seed(2024)
    data_size = 1000
    X = np.random.rand(data_size, 10)  # Feature data with 1000 rows and 10 columns
    y = np.random.randint(0, 2, size=data_size)  # Binary classification target variable
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

# Split data
df = create_mock_data()
predictors = [col for col in df.columns if col != 'target']
target = 'target'
train_df, val_df = train_test_split(df, test_size=0.2, random_state=2024)

# Test AdaBoostModel
def test_adaboost():   
    preds, clf = AdaBoostModel(train_df, val_df, predictors, target)
    accuracy = accuracy_score(val_df[target], preds)
    print("AdaBoost Accuracy:", accuracy)

# Test CatBoostModel
def test_catboost():
    preds, clf = CatBoostModel(train_df, val_df, predictors, target)
    accuracy = accuracy_score(val_df[target], preds)
    print("CatBoost Accuracy:", accuracy)

# Test XGBoostModel
def test_xgboost():
    preds, model, model_params = XGBoostModel(train_df, val_df, predictors, target)  # Get model, predictions, and parameters
    accuracy = accuracy_score(val_df[target], np.round(preds))  # Round predictions to the nearest integer (0 or 1)
    print("XGBoost Accuracy:", accuracy)
    print("Model Parameters:", model_params)

# Test LightGBMModel
def test_lightgbm():
    preds, model, model_params = LightGBMModel(train_df, val_df, predictors, target)  # Get model and evaluation results
    
    # Predict on the validation set
    preds = model.predict(val_df[predictors])
    
    # Convert continuous probability values to binary classification labels
    binary_preds = [1 if prob >= 0.5 else 0 for prob in preds]
    
    # Calculate accuracy
    accuracy = accuracy_score(val_df[target], binary_preds)
    print("LightGBM Accuracy:", accuracy)

# Run tests
if __name__ == "__main__":
    test_adaboost()
    test_catboost()
    test_xgboost()
    test_lightgbm()
