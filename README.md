# Credit Risk Detection

A Python package for analyzing and predicting credit card default risk using machine learning models such as AdaBoost, CatBoost, XGBoost, and LightGBM. The package also provides tools for preprocessing data, handling imbalanced datasets, and visualizing model performance.

## Features

- **Data Preprocessing**: Clean and transform credit card transaction data, including handling missing values and encoding categorical variables.
- **Data Balancing**: Handle imbalanced datasets using techniques such as random oversampling and SMOTE.
- **Machine Learning Models**: Train and evaluate models using AdaBoost, CatBoost, XGBoost, and LightGBM.
- **Evaluation**: Evaluate models with metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.
- **Visualization**: Visualize model performance, feature importance, and data distributions.

## Installation

To install the `credit_risk_detection` package, follow these steps:

1. Clone the repository or download the source code.
2. Install the package and dependencies by running:

```bash
pip install .
```

Alternatively, you can install the package in a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the package
pip install .
```
## Dependencies

The following dependencies are required for the package to work:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imbalanced-learn`
- `catboost`
- `xgboost`
- `lightgbm`

These dependencies will be automatically installed when you run the `pip install` command.

## Usage
1.Preprocessing the data
```python
from credit_risk_detection import preprocess_data

# Load your dataset
data = pd.read_csv("credit_card_data.csv")

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(data, target_column='default')
```

2.Model training and evaluation

AdaBoost
```python
from credit_risk_detection import AdaBoostModel

# Train the model
preds, clf = AdaBoostModel(X_train, X_test, predictors, target='default')

# Evaluate the model
from credit_risk_detection import evaluate_model_classification
evaluate_model_classification(y_test, preds)
```

CatBoost
```python
from credit_risk_detection import CatBoostModel

# Train the model
preds, clf = CatBoostModel(X_train, X_test, predictors, target='default')

# Evaluate the model
evaluate_model_classification(y_test, preds)
```

3.Visualizing model performance

ROC Curve
```python
from credit_risk_detection import CatBoostModel

# Train the model
preds, clf = CatBoostModel(X_train, X_test, predictors, target='default')

# Evaluate the model
evaluate_model_classification(y_test, preds)
```

Confusion Matrix
```python
from credit_risk_detection import CatBoostModel

# Train the model
preds, clf = CatBoostModel(X_train, X_test, predictors, target='default')

# Evaluate the model
evaluate_model_classification(y_test, preds)
```

##Authors
Wenbo Liu
Jiangao Han

##Acknowledgements
Special thanks to the open-source community and libraries that made this project possible.

##Explanation:

- **Title and Description**: A short introduction to the project, explaining its purpose and functionality.
- **Features**: A list of the key features the package offers.
- **Installation**: Detailed instructions on how to install the package, either directly or in a virtual environment.
- **Dependencies**: A list of required libraries that will be installed automatically.
- **Usage**: Sample code showing how to use the package for data preprocessing, model training, and visualization.
- **License**: Information about the license under which the package is distributed (MIT).
- **Contributing**: A section encouraging contributions from other developers.
- **Authors**: Listing the main contributors/authors of the project.
- **Acknowledgments**: A section to thank contributors or resources that helped with the project.
