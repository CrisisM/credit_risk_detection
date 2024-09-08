# test_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 从你的包中导入模型函数
from credit_risk_detection.credit_risk_main import AdaBoostModel, CatBoostModel, XGBoostModel, LightGBMModel

# 创建模拟数据
def create_mock_data():
    np.random.seed(2024)
    data_size = 1000
    X = np.random.rand(data_size, 10)  # 1000 行 10 列的特征数据
    y = np.random.randint(0, 2, size=data_size)  # 二分类目标变量
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

# 分割数据
df = create_mock_data()
predictors = [col for col in df.columns if col != 'target']
target = 'target'
train_df, val_df = train_test_split(df, test_size=0.2, random_state=2024)

# 测试 AdaBoostModel
def test_adaboost():
    preds, clf = AdaBoostModel(train_df, val_df, predictors, target)
    accuracy = accuracy_score(val_df[target], preds)
    print("AdaBoost Accuracy:", accuracy)

# 测试 CatBoostModel
def test_catboost():
    preds, clf = CatBoostModel(train_df, val_df, predictors, target)
    accuracy = accuracy_score(val_df[target], preds)
    print("CatBoost Accuracy:", accuracy)

# 测试 XGBoostModel
def test_xgboost():
    model = XGBoostModel(train_df, val_df, predictors, target)
    # XGBoost 返回的是模型，不直接返回预测结果
    dvalid = xgb.DMatrix(val_df[predictors])
    preds = model.predict(dvalid)
    accuracy = accuracy_score(val_df[target], preds)
    print("XGBoost Accuracy:", accuracy)

# 测试 LightGBMModel
def test_lightgbm():
    params = {'objective': 'binary', 'metric': 'binary_logloss'}
    model, final_auc_train, final_auc_valid = LightGBMModel(
        train_df, val_df, predictors, target, categorical_features=[], params=params)
    print("LightGBM Final AUC Train:", final_auc_train)
    print("LightGBM Final AUC Valid:", final_auc_valid)

# 运行测试
if __name__ == "__main__":
    test_adaboost()
    test_catboost()
    test_xgboost()
    test_lightgbm()
