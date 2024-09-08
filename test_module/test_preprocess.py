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
        # Create data for testing
        self.data = pd.DataFrame({
            'age': [25, 34, 28, 45, 52, 29],
            'income': [50000, 64000, 58000, 72000, 69000, 61000],
            'loan_amount': [10000, 20000, 15000, 25000, 18000, 16000],
            'default': [0, 1, 0, 0, 1, 0]
        })
        self.target_column = 'default'
        self.X, self.y = separate_features_and_target(self.data, self.target_column)

    def test_preprocess_data(self):
        # Test the complete preprocessing workflow
        X_train, X_test, y_train, y_test = preprocess_data(self.data, self.target_column, test_size=0.3)
        self.assertEqual(X_train.shape[0], len(y_train))  # Test if X and y lengths are consistent
        self.assertEqual(X_test.shape[0], len(y_test))

    def test_separate_features_and_target(self):
        # Test separating features and target
        X, y = separate_features_and_target(self.data, self.target_column)
        self.assertEqual(X.shape[1], 3)  # Ensure the number of features is correct
        self.assertEqual(len(y), len(self.data))  # Ensure the length of y is correct

    def test_preprocess_numeric_features(self):
        # Test numeric feature preprocessing
        numeric_transformer, numeric_features = preprocess_numeric_features(self.X)
        self.assertIn('age', numeric_features)  # Ensure 'age' is in numeric features
        self.assertIn('income', numeric_features)

    def test_preprocess_categorical_features(self):
        # Test categorical feature preprocessing
        categorical_transformer, categorical_features = preprocess_categorical_features(self.X)
        self.assertEqual(len(categorical_features), 0)  # Should be empty when there are no categorical features

    def test_build_preprocessor(self):
        # Test building the preprocessing pipeline
        numeric_transformer, numeric_features = preprocess_numeric_features(self.X)
        categorical_transformer, categorical_features = preprocess_categorical_features(self.X)
        preprocessor = build_preprocessor(numeric_transformer, categorical_transformer, numeric_features, categorical_features)
        X_preprocessed = apply_preprocessor(preprocessor, self.X)
        self.assertEqual(X_preprocessed.shape[1], len(numeric_features))  # Ensure the number of preprocessed features is correct

    def test_random_oversample(self):
        # Test random oversampling
        X_resampled, y_resampled = random_oversample(self.X, self.y)
        self.assertEqual(len(X_resampled), len(y_resampled))
        self.assertGreaterEqual(Counter(y_resampled)[1], Counter(y_resampled)[0])  # Minority class should be oversampled

    def test_smote_oversample(self):
        # Test SMOTE oversampling
        X_resampled, y_resampled = smote_oversample(self.X, self.y)
        self.assertEqual(len(X_resampled), len(y_resampled))
        self.assertGreaterEqual(Counter(y_resampled)[1], Counter(y_resampled)[0])  # SMOTE should balance the classes

    def test_split_data(self):
        # Test data splitting
        X_train, X_test, y_train, y_test = split_data(self.X, self.y, test_size=0.3)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

if __name__ == '__main__':
    unittest.main()
