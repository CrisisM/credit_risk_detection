# tests/test_preprocess.py

import unittest
from credit_risk_detection import preprocess_data
import pandas as pd

class TestPreprocessData(unittest.TestCase):

    def test_preprocess_data(self):
        # 创建一些测试数据
        data = pd.DataFrame({
            'age': [25, 34, 28, 45, 52, 29],
            'income': [50000, 64000, 58000, 72000, 69000, 61000],
            'loan_amount': [10000, 20000, 15000, 25000, 18000, 16000],
            'default': [0, 1, 0, 0, 1, 0]
        })

        # 测试 preprocess_data 函数
        X_train, X_test, y_train, y_test = preprocess_data(data, target_column='default')

        # 检查输出是否正确
        self.assertEqual(X_train.shape[1], 3)  # 测试特征数是否为 3
        self.assertEqual(len(X_train), len(y_train))  # X 和 y 的长度应一致

if __name__ == '__main__':
    unittest.main()
