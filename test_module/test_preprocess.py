import unittest
import pandas as pd
import numpy as np
from collections import Counter

from credit_risk_detection.preprocess import (
    preprocess_data, separate_features_and_target, preprocess_numeric_features,
    preprocess_categorical_features, build_preprocessor, apply_preprocessor,
    random_oversample, smote_oversample, split_data
)

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        # 创建用于测试的数据
        self.data = pd.DataFrame({
            'age': [25, 34, 28, 45, 52, 29],
            'income': [50000, 64000, 58000, 72000, 69000, 61000],
            'loan_amount': [10000, 20000, 15000, 25000, 18000, 16000],
            'default': [0, 1, 0, 0, 1, 0]
        })
        self.target_column = 'default'
        self.X, self.y = separate_features_and_target(self.data, self.target_column)

    def test_preprocess_data(self):
        # 测试完整的预处理流程
        X_train, X_test, y_train, y_test = preprocess_data(self.data, self.target_column, test_size=0.3)
        self.assertEqual(X_train.shape[0], len(y_train))  # 测试 X 和 y 长度是否一致
        self.assertEqual(X_test.shape[0], len(y_test))

    def test_separate_features_and_target(self):
        # 测试特征和目标的分离
        X, y = separate_features_and_target(self.data, self.target_column)
        self.assertEqual(X.shape[1], 3)  # 确保特征数正确
        self.assertEqual(len(y), len(self.data))  # 确保 y 的长度正确

    def test_preprocess_numeric_features(self):
        # 测试数值特征的预处理
        numeric_transformer, numeric_features = preprocess_numeric_features(self.X)
        self.assertIn('age', numeric_features)  # 确保 age 在数值特征中
        self.assertIn('income', numeric_features)

    def test_preprocess_categorical_features(self):
        # 测试类别特征的预处理
        categorical_transformer, categorical_features = preprocess_categorical_features(self.X)
        self.assertEqual(len(categorical_features), 0)  # 没有类别特征时应该为空

    def test_build_preprocessor(self):
        # 测试构建预处理步骤
        numeric_transformer, numeric_features = preprocess_numeric_features(self.X)
        categorical_transformer, categorical_features = preprocess_categorical_features(self.X)
        preprocessor = build_preprocessor(numeric_transformer, categorical_transformer, numeric_features, categorical_features)
        X_preprocessed = apply_preprocessor(preprocessor, self.X)
        self.assertEqual(X_preprocessed.shape[1], len(numeric_features))  # 确保预处理后的特征数正确

    def test_random_oversample(self):
        # 测试随机过采样
        X_resampled, y_resampled = random_oversample(self.X, self.y)
        self.assertEqual(len(X_resampled), len(y_resampled))
        self.assertGreaterEqual(Counter(y_resampled)[1], Counter(y_resampled)[0])  # 少数类应该被过采样

    def test_smote_oversample(self):
        # 测试 SMOTE 过采样
        X_resampled, y_resampled = smote_oversample(self.X, self.y)
        self.assertEqual(len(X_resampled), len(y_resampled))
        self.assertGreaterEqual(Counter(y_resampled)[1], Counter(y_resampled)[0])  # SMOTE 应该平衡类别

    def test_split_data(self):
        # 测试数据拆分
        X_train, X_test, y_train, y_test = split_data(self.X, self.y, test_size=0.3)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

if __name__ == '__main__':
    unittest.main()
