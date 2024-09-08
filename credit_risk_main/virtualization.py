import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def plot_default_distribution(data, target_column, plot_type='bar'):
    """
    显示非违约客户和违约客户的比例示意图。
    
    参数:
    - data: 输入的 DataFrame 数据
    - target_column: 目标列，表示违约与否的标签
    - plot_type: 图表类型，可选 'bar'（默认）或 'pie'
    """
    # 统计违约与非违约客户的数量
    default_counts = data[target_column].value_counts()
    labels = default_counts.index.map({0: 'Non-default', 1: 'Default'})
    
    if plot_type == 'bar':
        # 使用条形图显示比例
        plt.figure(figsize=(6, 4))
        sns.barplot(x=labels, y=default_counts, palette='Set2')
        plt.title('Default vs Non-default Distribution')
        plt.xlabel('Customer Type')
        plt.ylabel('Count')
        plt.show()
    
    elif plot_type == 'pie':
        # 使用饼图显示比例
        plt.figure(figsize=(6, 6))
        plt.pie(default_counts, labels=labels, autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'], startangle=90, wedgeprops={'edgecolor': 'black'})
        plt.title('Default vs Non-default Distribution')
        plt.show()

    else:
        raise ValueError("Invalid plot_type. Choose 'bar' or 'pie'.")

def plot_default_amount_distribution(data, amount_column, target_column, plot_type='hist', bins=None):
    """
    显示违约客户和非违约客户的金额分布示意图，并允许用户选择区间（bins）。
    
    参数:
    - data: 输入的 DataFrame 数据
    - amount_column: 金额列的名称
    - target_column: 目标列，表示违约与否
    - plot_type: 图表类型，可选 'hist'（默认）或 'box'
    - bins: 直方图的区间数（仅对 'hist' 图表有效）
    """
    if plot_type == 'hist':
        # 使用直方图显示违约和非违约客户的金额分布
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=amount_column, hue=target_column, kde=True, palette="Set2", multiple="stack", bins=bins)
        plt.title(f'{amount_column} Distribution for Default vs Non-default')
        plt.xlabel(amount_column)
        plt.ylabel('Frequency')
        plt.show()
    
    elif plot_type == 'box':
        # 使用箱线图显示违约和非违约客户的金额分布
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
    绘制单个特征的分布，区分违约和未违约客户。
    
    参数:
    - data: 输入的 DataFrame 数据
    - feature: 要展示的特征列名称
    - target_column: 目标列，若提供则按目标列进行分类展示
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
    绘制特征相关性热图。
    
    参数:
    - data: 输入的 DataFrame 数据
    - title: 图表标题
    """
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title(title)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    绘制混淆矩阵，展示模型的预测结果。
    
    参数:
    - y_true: 实际标签
    - y_pred: 预测标签
    - labels: 类别标签（可选）
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
    绘制ROC曲线并计算AUC。
    
    参数:
    - y_true: 实际标签
    - y_score: 预测得分或概率
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
    绘制特征重要性图，适用于树模型（如随机森林、XGBoost等）。
    
    参数:
    - model: 训练好的模型（应具有 `feature_importances_` 属性）
    - feature_names: 特征名称列表
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Ranking')
    plt.show()
