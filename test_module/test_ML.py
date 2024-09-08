import unittest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from credit_risk_detection import model_gs, run_all_models

class TestModelFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=10, n_informative=8, n_classes=2, random_state=2024)
        cls.train_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        cls.train_df['target'] = y
        cls.val_df = cls.train_df.copy()
        
        cls.predictors = [f'feature_{i}' for i in range(10)]
        cls.target = 'target'

    def test_model_gs_lr(self):
        best_model, val_preds_proba = model_gs(
            self.train_df, self.val_df, self.predictors, self.target, 'LR'
        )
        self.assertIsNotNone(best_model)
        self.assertEqual(len(val_preds_proba), len(self.val_df))
        val_roc_auc = roc_auc_score(self.val_df[self.target], val_preds_proba)
        self.assertGreater(val_roc_auc, 0.5)  # Assuming a non-trivial ROC-AUC

    def test_model_gs_knn(self):
        best_model, val_preds_proba = model_gs(
            self.train_df, self.val_df, self.predictors, self.target, 'KNN'
        )
        self.assertIsNotNone(best_model)
        self.assertEqual(len(val_preds_proba), len(self.val_df))
        val_roc_auc = roc_auc_score(self.val_df[self.target], val_preds_proba)
        self.assertGreater(val_roc_auc, 0.5)

    def test_model_gs_svc(self):
        best_model, val_preds_proba = model_gs(
            self.train_df, self.val_df, self.predictors, self.target, 'SVC'
        )
        self.assertIsNotNone(best_model)
        self.assertEqual(len(val_preds_proba), len(self.val_df))
        val_roc_auc = roc_auc_score(self.val_df[self.target], val_preds_proba)
        self.assertGreater(val_roc_auc, 0.5)

    def test_model_gs_dt(self):
        best_model, val_preds_proba = model_gs(
            self.train_df, self.val_df, self.predictors, self.target, 'DT'
        )
        self.assertIsNotNone(best_model)
        self.assertEqual(len(val_preds_proba), len(self.val_df))
        val_roc_auc = roc_auc_score(self.val_df[self.target], val_preds_proba)
        self.assertGreater(val_roc_auc, 0.5)

    def test_model_gs_rfc(self):
        best_model, val_preds_proba = model_gs(
            self.train_df, self.val_df, self.predictors, self.target, 'RFC'
        )
        self.assertIsNotNone(best_model)
        self.assertEqual(len(val_preds_proba), len(self.val_df))
        val_roc_auc = roc_auc_score(self.val_df[self.target], val_preds_proba)
        self.assertGreater(val_roc_auc, 0.5)

    def test_run_all_models(self):
        results = run_all_models(self.train_df, self.val_df, self.predictors, self.target)
        self.assertEqual(len(results), 5)  # We are testing 5 models
        for model_name, model in results.items():
            self.assertIsNotNone(model)
            print(f"Model {model_name} tested successfully")

if __name__ == '__main__':
    unittest.main()
