from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

'LogisticRegression' # 逻辑回归
"SVC" # 支撑向量机
"KNN" # K近邻
'DT'# 决策树
'RFC' # 随机森林
'Bagging'# 集成学习bagging
'SGD' # 随机梯度
'GBC'# 集成学习Gradient
def model_gs(train_df, val_df, predictors, target, model_name):
    X_train = train_df[predictors]
    y_train = train_df[target].values
    X_val = val_df[predictors]
    y_val = val_df[target].values
    
    models_needing_scaling = ['LR', 'SVC']
    
    if model_name in models_needing_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    if model_name == 'LR':
        LR_param = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['liblinear'],
            'max_iter': [1000]
        }
        model = LogisticRegression()
        param_grid = LR_param
        
    elif model_name == 'KNN':
        KNN_param = {
            'n_neighbors': list(range(2, 5, 1)),
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        model = KNeighborsClassifier()
        param_grid = KNN_param
        
    elif model_name == 'SVC':
        SVC_param = {
            'C': [0.5, 0.7, 0.9, 1],
            'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
        }
        model = SVC(probability=True)
        param_grid = SVC_param
        
    elif model_name == 'DT':
        DT_param = {
            'criterion': ['gini', 'entropy'],
            'max_depth': list(range(2, 5, 1)),
            'min_samples_leaf': list(range(3, 7, 1))
        }
        model = DecisionTreeClassifier()
        param_grid = DT_param
        
    elif model_name == 'RFC':
        RFC_param = {
            'n_estimators': [100, 150, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': list(range(2, 5, 1)),
        }
        model = RandomForestClassifier()
        param_grid = RFC_param
        
    elif model_name == 'SGD':
        SGD_param = {
            'penalty': ['l2', 'l1'],
            'max_iter': [1000, 1500, 2000],
            'loss': ['log_loss', 'modified_huber']  # 支持 predict_proba 的损失函数
        }
        model = SGDClassifier()
        param_grid = SGD_param
    
    else:
        raise ValueError("Model not recognized. Please select from 'LR', 'KNN', 'SVC', 'DT', 'RFC', 'SGD'.")

    gs = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
    gs.fit(X_train, y_train)

    best_estimator = gs.best_estimator_
    
    # 使用 predict_proba 或 decision_function
    try:
        val_preds_proba = best_estimator.predict_proba(X_val)[:, 1]
    except AttributeError:
        val_preds_proba = best_estimator.decision_function(X_val)

    val_roc_auc = roc_auc_score(y_val, val_preds_proba)
    
    # print(f"Best parameters for {model_name}: {gs.best_params_}")
    # print(f"Validation ROC-AUC for {model_name}: {val_roc_auc:.4f}")
    
    return best_estimator, val_preds_proba

# 运行所有模型
def run_all_models(train_df, val_df, predictors, target):
    models = ['LR', 'KNN', 'DT', 'RFC', 'SGD']
    results = {}
    
    for model_name in models:
        print(f"Running model: {model_name}")
        best_model, val_preds_proba = model_gs(train_df, val_df, predictors, target, model_name)
        results[model_name] = best_model
        
    return results

# 调用时使用
# results = run_all_models(train_df, val_df, predictors, target)