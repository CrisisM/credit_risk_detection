from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report as cr
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_classification(y_true, y_pred, y_prob=None):
    """
    评估分类模型的主要指标，包括准确率、精确率、召回率、F1 分数和AUC-ROC。
    
    参数:
    - y_true: 实际标签
    - y_pred: 模型预测的标签
    - y_prob: 模型预测的正类概率（用于计算AUC）
    
    返回:
    - 一个包含主要评估指标的字典
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
    绘制分类模型的混淆矩阵。
    
    参数:
    - y_true: 实际标签
    - y_pred: 模型预测的标签
    - labels: 类别标签（可选）
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
    生成分类模型的精确率、召回率、F1 分数报告。
    
    参数:
    - y_true: 实际标签
    - y_pred: 模型预测的标签
    
    返回:
    - 字符串格式的分类报告
    """
    report = cr(y_true, y_pred)
    print("Classification Report:")
    print(report)
    return report
