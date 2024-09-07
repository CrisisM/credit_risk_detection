import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler   # 特征缩放
from sklearn.metrics import accuracy_score         # 评估分类器性能
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
#加载数据集
all_transaction = pd.read_csv("UCI_Credit_Card.csv")
all_transaction.shape
#显示前五行
all_transaction.head()
all_transaction.info()
all_transaction.describe()
# 统计数据中的NaN
all_transaction.isnull().sum()
#确认数据平衡程度
payment_counts = all_transaction["default.payment.next.month"].value_counts()
default_counts_df = pd.DataFrame({'default.payment.next.month': payment_counts.index,'values': payment_counts.values})
plt.figure(figsize = (6,6))
plt.title('Default Credit Card Clients - target value - data unbalance\n (Not Default = 0, Default = 1)')
bar_plot = sns.barplot(x = 'default.payment.next.month', y="values", data=default_counts_df)
for p in bar_plot.patches:
    bar_plot.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom')
plt.show()
print(all_transaction['LIMIT_BAL'].describe())plt.figure(figsize = (14,6))
plt.title('Amount of credit limit - Density Plot')
plt.xlim(0, 1000000)  # 根据你的数据范围进行调整
sns.histplot(all_transaction['LIMIT_BAL'],kde=True,bins=200, color="blue")
plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="SEX", y="LIMIT_BAL", hue="SEX",data=all_transaction, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="SEX", y="LIMIT_BAL", hue="SEX",data=all_transaction, palette="PRGn",showfliers=False)
plt.show()
fig, (ax1,ax2) = plt.subplots(ncols = 2, figsize = (12,6))
sns.boxplot(ax = ax1, x="SEX", y="LIMIT_BAL", hue="SEX",data=all_transaction,palette="PRGn",showfliers=False)
sns.boxplot(ax = ax2, x="default.payment.next.month", y="LIMIT_BAL", hue="default.payment.next.month",data=all_transaction,palette="PRGn",showfliers=False)
plt.show()
var = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE']

plt.figure(figsize = (8,8))
plt.title('Amount of (Apr-Sept) \ncorrelation plot (Pearson)')
corr = all_transaction[var].corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)
plt.show()


## Define predictors and target values ##
target = 'default.payment.next.month'
predictors = [  'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
VALID_SIZE = 0.2
train_df, val_df = train_test_split(all_transaction, test_size=VALID_SIZE, random_state=2024, shuffle=True )

# 复制一份数据
train_df_bkp = train_df.copy()
val_df_bkp = val_df.copy()

## RandomForestClassifier ##
clf= RandomForestClassifier(n_jobs=-1, random_state=2024, criterion='gini', n_estimators= 100, verbose= False)
# 用测试集预测
preds = clf.predict(val_df[predictors])
# 可视化参数重要程度
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 
# 绘制confusion matrix
cm = cm = confusion_matrix(val_df[target], preds)
# 使用热力图来绘制混淆矩阵
labels = ['not default', 'default']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=labels, yticklabels=labels)

# 设置标题和坐标轴标签

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(classification_report(val_df[target], preds))
print('The ROC-AUC score obtained with RandomForrestClassifier is', roc_auc_score(val_df[target].values, preds))
#因为非数值型的特征无法直接被随机森林等分类器处理，采用one-hot encoding方法预处理这些特征。
cat_features = ['EDUCATION', 'SEX', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
train_f_df = pd.get_dummies(train_df_bkp, columns = cat_features)
val_f_df = pd.get_dummies(val_df_bkp, columns = cat_features)
#确保有相同的columns
train_fa_df, val_fa_df = train_f_df.align(val_f_df, join='outer', axis=1, fill_value=0)
#确认
print("Default of Credit Card Clients train data -  rows:",train_fa_df.shape[0]," columns:", train_fa_df.shape[1])
print("Default of Credit Card Clients val  data -  rows:",val_fa_df.shape[0]," columns:", val_fa_df.shape[1])
train_fa_df.head()
val_fa_df.head()
#define new predictors 
predictors_f = ['AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
       'BILL_AMT5', 'BILL_AMT6', 'EDUCATION_0', 'EDUCATION_1',
       'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'EDUCATION_5',
       'EDUCATION_6', 'LIMIT_BAL', 'MARRIAGE_0', 'MARRIAGE_1',
       'MARRIAGE_2', 'MARRIAGE_3', 'PAY_0_-1', 'PAY_0_-2', 'PAY_0_0',
       'PAY_0_1', 'PAY_0_2', 'PAY_0_3', 'PAY_0_4', 'PAY_0_5', 'PAY_0_6',
       'PAY_0_7', 'PAY_0_8', 'PAY_2_-1', 'PAY_2_-2', 'PAY_2_0', 'PAY_2_1',
       'PAY_2_2', 'PAY_2_3', 'PAY_2_4', 'PAY_2_5', 'PAY_2_6', 'PAY_2_7',
       'PAY_2_8', 'PAY_3_-1', 'PAY_3_-2', 'PAY_3_0', 'PAY_3_1', 'PAY_3_2',
       'PAY_3_3', 'PAY_3_4', 'PAY_3_5', 'PAY_3_6', 'PAY_3_7', 'PAY_3_8',
       'PAY_4_-1', 'PAY_4_-2', 'PAY_4_0', 'PAY_4_1', 'PAY_4_2', 'PAY_4_3',
       'PAY_4_4', 'PAY_4_5', 'PAY_4_6', 'PAY_4_7', 'PAY_4_8', 'PAY_5_-1',
       'PAY_5_-2', 'PAY_5_0', 'PAY_5_2', 'PAY_5_3', 'PAY_5_4', 'PAY_5_5',
       'PAY_5_6', 'PAY_5_7', 'PAY_5_8', 'PAY_6_-1', 'PAY_6_-2', 'PAY_6_0',
       'PAY_6_2', 'PAY_6_3', 'PAY_6_4', 'PAY_6_5', 'PAY_6_6', 'PAY_6_7',
       'PAY_6_8', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
       'PAY_AMT5', 'PAY_AMT6', 'SEX_1', 'SEX_2']

# 用独热处理后的数据再次进行训练
clf.fit(train_fa_df[predictors_f], train_fa_df[target].values)
# 用测试集预测
preds = clf.predict(val_fa_df[predictors_f])

# 可视化参数重要程度
tmp = pd.DataFrame({'Feature': predictors_f, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (16,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 
# 绘制confusion matrix
cm = confusion_matrix(val_f_df[target], preds)
# 使用热力图来绘制混淆矩阵
labels = ['not default', 'default']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=labels, yticklabels=labels)

# 设置标题和坐标轴标签

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(classification_report(val_df[target], preds))
print('The ROC-AUC score obtained with RandomForrestClassifier is', roc_auc_score(val_df[target].values, preds))

## AdaBoostClassifier ##
#拟合模型
clf.fit(train_df[predictors], train_df[target].values)
#预测
preds = clf.predict(val_df[predictors])
#同之前一样看特征的重要程度和混淆矩阵
# 可视化参数重要程度
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 
# 绘制confusion matrix
cm = cm = confusion_matrix(val_df[target], preds)
# 使用热力图来绘制混淆矩阵
labels = ['not default', 'default']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=labels, yticklabels=labels)

# 设置标题和坐标轴标签

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(classification_report(val_df[target], preds))
print('The ROC-AUC score obtained with AdaBoostClassifier is', roc_auc_score(val_df[target].values, preds))

## CatBoostClassifier ##
#初始化模型参数
clf = CatBoostClassifier(iterations=500,
                             learning_rate=0.02,
                             depth=12,
                             eval_metric='AUC',
                             random_seed = 2024,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=100)
#拟合模型
clf.fit(train_df[predictors], train_df[target].values,verbose=True)
#进行预测
preds = clf.predict(val_df[predictors])
#特征重要程度
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   

#混淆矩阵
cm = pd.crosstab(val_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()
print('The ROC-AUC score obtained with CatBoostClassifier is', roc_auc_score(val_df[target].values, preds))