import unittest
import pandas as pd
import numpy as np
from credit_risk_detection.virtualization import (
    plot_default_distribution, plot_default_amount_distribution, plot_feature_distribution, 
    plot_correlation_heatmap, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive mode to prevent figures from popping up during tests

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # Set up data for testing
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
        # Test default bar plot
        plot_default_distribution(self.data, 'default', plot_type='bar')
        # Test pie chart
        plot_default_distribution(self.data, 'default', plot_type='pie')

    def test_plot_default_amount_distribution(self):
        # Test loan amount distribution histogram
        plot_default_amount_distribution(self.data, 'loan_amount', 'default', plot_type='hist', bins=5)
        # Test box plot
        plot_default_amount_distribution(self.data, 'loan_amount', 'default', plot_type='box')

    def test_plot_feature_distribution(self):
        # Test feature distribution
        plot_feature_distribution(self.data, 'income', 'default')

    def test_plot_correlation_heatmap(self):
        # Test correlation heatmap
        plot_correlation_heatmap(self.data)

    def test_plot_confusion_matrix(self):
        # Test confusion matrix
        plot_confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])

    def test_plot_roc_curve(self):
        # Test ROC curve
        plot_roc_curve(self.y_true, self.y_score)

    def test_plot_feature_importance(self):
        # Create a simple model object with feature importance
        class DummyModel:
            feature_importances_ = np.array([0.2, 0.4, 0.1, 0.3])

        model = DummyModel()
        feature_names = ['age', 'income', 'loan_amount', 'other_feature']

        # Test feature importance plot
        plot_feature_importance(model, feature_names)

if __name__ == '__main__':
    unittest.main()
