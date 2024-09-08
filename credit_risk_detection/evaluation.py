from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report as cr
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_classification(y_true, y_pred, y_prob=None):
    """
    Evaluate the main metrics for a classification model, including accuracy, precision, recall, F1 score, and AUC-ROC.
    
    Parameters:
    - y_true: Actual labels
    - y_pred: Predicted labels from the model
    - y_prob: Predicted probabilities for the positive class (used for AUC calculation)
    
    Returns:
    - A dictionary containing the main evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_prob)
    
    print("Model Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot the confusion matrix for a classification model.
    
    Parameters:
    - y_true: Actual labels
    - y_pred: Predicted labels from the model
    - labels: Class labels (optional)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def classification_report(y_true, y_pred):
    """
    Generate a classification report showing precision, recall, and F1 score.
    
    Parameters:
    - y_true: Actual labels
    - y_pred: Predicted labels from the model
    
    Returns:
    - A string-format classification report
    """
    report = cr(y_true, y_pred)
    print("Classification Report:")
    print(report)
    return report
