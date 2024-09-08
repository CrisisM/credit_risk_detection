# Credit Risk Analysis

This Python package provides tools for credit card default risk analysis, including data preprocessing, model evaluation, and visualization.

## Installation

To install the package, run:

```bash
pip install -r Requirement.txt
```

## Usage

```python
from credit_risk_analysis import preprocess_data, evaluate_model_classification, plot_confusion_matrix

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(data, target_column='default')

# Train model...
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
metrics = evaluate_model_classification(y_test, y_pred)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred)
```

## Modules

- **[`preprocess`](./preprocess.py)**: Provides data preprocessing functions.
- **[`evaluation`](./evaluation.py)**: Provides model evaluation functions.
- **[`virtualization`](./virtualization.py)**: Provides various plotting functions.