import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def plot_default_distribution(data, target_column, plot_type='bar'):
    """
    Display a graph showing the proportion of non-default and default customers.
    
    Parameters:
    - data: The input DataFrame
    - target_column: The target column representing default status
    - plot_type: The type of plot, can be 'bar' (default) or 'pie'
    """
    # Count the number of default and non-default customers
    default_counts = data[target_column].value_counts()
    labels = default_counts.index.map({0: 'Non-default', 1: 'Default'})
    
    if plot_type == 'bar':
        # Display the proportion using a bar chart
        plt.figure(figsize=(6, 4))
        sns.barplot(x=labels, y=default_counts, palette='Set2')
        plt.title('Default vs Non-default Distribution')
        plt.xlabel('Customer Type')
        plt.ylabel('Count')
        plt.show()
    
    elif plot_type == 'pie':
        # Display the proportion using a pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(default_counts, labels=labels, autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'], startangle=90, wedgeprops={'edgecolor': 'black'})
        plt.title('Default vs Non-default Distribution')
        plt.show()

    else:
        raise ValueError("Invalid plot_type. Choose 'bar' or 'pie'.")

def plot_default_amount_distribution(data, amount_column, target_column, plot_type='hist', bins=None):
    """
    Display a graph of the amount distribution for default and non-default customers, allowing the user to choose bins.
    
    Parameters:
    - data: The input DataFrame
    - amount_column: The name of the column representing amounts
    - target_column: The target column representing default status
    - plot_type: The type of plot, can be 'hist' (default) or 'box'
    - bins: The number of bins for the histogram (only effective for 'hist')
    """
    if plot_type == 'hist':
        # Display the amount distribution for default and non-default customers using a histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=amount_column, hue=target_column, kde=True, palette="Set2", multiple="stack", bins=bins)
        plt.title(f'{amount_column} Distribution for Default vs Non-default')
        plt.xlabel(amount_column)
        plt.ylabel('Frequency')
        plt.show()
    
    elif plot_type == 'box':
        # Display the amount distribution for default and non-default customers using a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target_column, y=amount_column, data=data, palette="Set2")
        plt.title(f'{amount_column} Distribution for Default vs Non-default')
        plt.xlabel('Customer Type')
        plt.ylabel(amount_column)
        plt.xticks(ticks=[0, 1], labels=['Non-default', 'Default'])
        plt.show()

    else:
        raise ValueError("Invalid plot_type. Choose 'hist' or 'box'.")

def plot_feature_distribution(data, feature, target_column=None):
    """
    Plot the distribution of a single feature, distinguishing between default and non-default customers.
    
    Parameters:
    - data: The input DataFrame
    - feature: The name of the feature column to display
    - target_column: The target column (optional) for distinguishing different groups
    """
    plt.figure(figsize=(10, 6))
    
    if target_column:
        sns.histplot(data=data, x=feature, hue=target_column, kde=True, palette="Set2")
    else:
        sns.histplot(data=data, x=feature, kde=True)
    
    plt.title(f'{feature} distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def plot_correlation_heatmap(data, title='Correlation Matrix'):
    """
    Plot a heatmap of feature correlations.
    
    Parameters:
    - data: The input DataFrame
    - title: The title of the plot
    """
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title(title)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot a confusion matrix showing the model's prediction results.
    
    Parameters:
    - y_true: Actual labels
    - y_pred: Predicted labels
    - labels: Class labels (optional)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc_curve(y_true, y_score):
    """
    Plot the ROC curve and compute the AUC.
    
    Parameters:
    - y_true: Actual labels
    - y_score: Predicted scores or probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance, applicable to tree models (e.g., Random Forest, XGBoost).
    
    Parameters:
    - model: Trained model (should have `feature_importances_` attribute)
    - feature_names: List of feature names
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Ranking')
    plt.show()
