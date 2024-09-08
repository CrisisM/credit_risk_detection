import unittest
import numpy as np
from credit_risk_detection.evaluation import (
    evaluate_model_classification, plot_confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive mode to prevent figures from popping up during tests

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        self.y_prob = np.array([0.2, 0.8, 0.1, 0.7, 0.9, 0.4, 0.2, 0.85])  # For AUC calculation

    def test_evaluate_model_classification(self):
        # Test classification model evaluation function
        metrics = evaluate_model_classification(self.y_true, self.y_pred, self.y_prob)
        self.assertIn('Accuracy', metrics)
        self.assertIn('Precision', metrics)
        self.assertIn('Recall', metrics)
        self.assertIn('F1 Score', metrics)
        self.assertIn('AUC', metrics)

    def test_plot_confusion_matrix(self):
        # Test confusion matrix plotting
        plot_confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])

    def test_classification_report(self):
        # Test generation of classification report
        report = classification_report(self.y_true, self.y_pred)
        self.assertIsInstance(report, str)  # Ensure the report is a string

if __name__ == '__main__':
    unittest.main()
