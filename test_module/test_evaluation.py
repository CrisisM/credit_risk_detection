import unittest
import numpy as np
from credit_risk_detection.evaluation import (
    evaluate_model_classification, plot_confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式，防止测试中弹出图形窗口

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # 设置测试数据
        self.y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        self.y_prob = np.array([0.2, 0.8, 0.1, 0.7, 0.9, 0.4, 0.2, 0.85])  # 用于AUC计算

    def test_evaluate_model_classification(self):
        # 测试分类模型评估函数
        metrics = evaluate_model_classification(self.y_true, self.y_pred, self.y_prob)
        self.assertIn('Accuracy', metrics)
        self.assertIn('Precision', metrics)
        self.assertIn('Recall', metrics)
        self.assertIn('F1 Score', metrics)
        self.assertIn('AUC', metrics)

    def test_plot_confusion_matrix(self):
        # 测试混淆矩阵的绘制
        plot_confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])

    def test_classification_report(self):
        # 测试分类报告的生成
        report = classification_report(self.y_true, self.y_pred)
        self.assertIsInstance(report, str)  # 确保返回的是字符串报告

if __name__ == '__main__':
    unittest.main()
