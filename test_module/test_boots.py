# test_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 从你的包中导入模型函数
from credit_risk_detection.boost_model import AdaBoostModel, CatBoostModel, XGBoostModel, LightGBMModel

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
    preds, clf= AdaBoostModel(train_df, val_df, predictors, target)
    accuracy = accuracy_score(val_df[target], preds)
    print("AdaBoost Accuracy:", accuracy)


# 测试 CatBoostModel
def test_catboost():
    preds, clf = CatBoostModel(train_df, val_df, predictors, target)
    accuracy = accuracy_score(val_df[target], preds)
    print("CatBoost Accuracy:", accuracy)

# 测试 XGBoostModel
def test_xgboost():
    preds, model, model_params = XGBoostModel(train_df, val_df, predictors, target)  # 直接获得模型、预测值和参数
    accuracy = accuracy_score(val_df[target], np.round(preds))  # 将预测结果四舍五入到最近的整数（0 或 1）
    print("XGBoost Accuracy:", accuracy)
    print("Model Parameters:", model_params)

# 测试 LightGBMModel
def test_lightgbm():
    preds, model, model_params = LightGBMModel(train_df, val_df, predictors, target)  # 获取模型和评估结果
    
    # 预测验证集
    preds = model.predict(val_df[predictors])
    
    # 将连续的概率值转换为二进制分类标签
    binary_preds = [1 if prob >= 0.5 else 0 for prob in preds]
    
    # 计算准确率
    accuracy = accuracy_score(val_df[target], binary_preds)
    print("LightGBM Accuracy:", accuracy)


# 运行测试
if __name__ == "__main__":
    test_adaboost()
    test_catboost()
    test_xgboost()
    test_lightgbm()
