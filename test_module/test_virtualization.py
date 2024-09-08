import unittest
import pandas as pd
import numpy as np
from credit_risk_detection.virtualization import (
    plot_default_distribution, plot_default_amount_distribution, plot_feature_distribution, 
    plot_correlation_heatmap, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
)
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式，防止测试中弹出图形窗口

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # 设置测试所需的数据
        self.data = pd.DataFrame({
            'age': [25, 34, 28, 45, 52, 29],
            'income': [50000, 64000, 58000, 72000, 69000, 61000],
            'loan_amount': [10000, 20000, 15000, 25000, 18000, 16000],
            'default': [0, 1, 0, 0, 1, 0]
        })

        self.y_true = np.array([0, 1, 0, 0, 1, 0])
        self.y_pred = np.array([0, 1, 0, 0, 1, 1])
        self.y_score = np.array([0.1, 0.8, 0.2, 0.4, 0.9, 0.6])

    def test_plot_default_distribution(self):
        # 测试默认条形图
        plot_default_distribution(self.data, 'default', plot_type='bar')
        # 测试饼图
        plot_default_distribution(self.data, 'default', plot_type='pie')

    def test_plot_default_amount_distribution(self):
        # 测试金额分布直方图
        plot_default_amount_distribution(self.data, 'loan_amount', 'default', plot_type='hist', bins=5)
        # 测试箱线图
        plot_default_amount_distribution(self.data, 'loan_amount', 'default', plot_type='box')

    def test_plot_feature_distribution(self):
        # 测试特征分布
        plot_feature_distribution(self.data, 'income', 'default')

    def test_plot_correlation_heatmap(self):
        # 测试相关性热图
        plot_correlation_heatmap(self.data)

    def test_plot_confusion_matrix(self):
        # 测试混淆矩阵
        plot_confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])

    def test_plot_roc_curve(self):
        # 测试ROC曲线
        plot_roc_curve(self.y_true, self.y_score)

    def test_plot_feature_importance(self):
        # 创建一个简单的模型对象并设置特征重要性
        class DummyModel:
            feature_importances_ = np.array([0.2, 0.4, 0.1, 0.3])

        model = DummyModel()
        feature_names = ['age', 'income', 'loan_amount', 'other_feature']

        # 测试特征重要性图
        plot_feature_importance(model, feature_names)

if __name__ == '__main__':
    unittest.main()
