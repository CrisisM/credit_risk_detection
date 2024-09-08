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
    
    # 需要标准化的模型
    models_needing_scaling = ['LR', 'SVC']
    
    # 对于 Logistic Regression 和 SVC 进行标准化
    if model_name in models_needing_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    if model_name == 'LR':
        # Logistic Regression
        LR_param = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['liblinear'],  # 适用于 l1 和 l2 的 solver
            'max_iter': [1000]  # 增加最大迭代次数
        }
        model = LogisticRegression()
        param_grid = LR_param
        
    elif model_name == 'KNN':
        # KNN
        KNN_param = {
            'n_neighbors': list(range(2, 5, 1)),
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        model = KNeighborsClassifier()
        param_grid = KNN_param
        
    elif model_name == 'SVC':
        # SVC
        SVC_param = {
            'C': [0.5, 0.7, 0.9, 1],
            'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
        }
        model = SVC(probability=True)  # 为了获得预测概率，设置 probability=True
        param_grid = SVC_param
        
    elif model_name == 'DT':
        # Decision Tree
        DT_param = {
            'criterion': ['gini', 'entropy'],
            'max_depth': list(range(2, 5, 1)),
            'min_samples_leaf': list(range(3, 7, 1))
        }
        model = DecisionTreeClassifier()
        param_grid = DT_param
        
    elif model_name == 'RFC':
        # Random Forest Classifier
        RFC_param = {
            'n_estimators': [100, 150, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': list(range(2, 5, 1)),
        }
        model = RandomForestClassifier()
        param_grid = RFC_param
        
    elif model_name == 'SGD':
        # Stochastic Gradient Descent
        SGD_param = {
            'penalty': ['l2', 'l1'],
            'max_iter': [1000, 1500, 2000]
        }
        model = SGDClassifier()
        param_grid = SGD_param
    
    else:
        raise ValueError("Model not recognized. Please select from 'LR', 'KNN', 'SVC', 'DT', 'RFC', 'SGD'.")

    # Apply GridSearchCV to find the best parameters, using roc_auc as scoring metric
    gs = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
    gs.fit(X_train, y_train)

    best_estimator = gs.best_estimator_  # Best estimator after grid search
    
    # Make predictions on the validation set (predict probabilities for ROC-AUC)
    val_preds_proba = best_estimator.predict_proba(X_val)[:, 1]
    
    # Evaluate ROC-AUC on the validation set
    val_roc_auc = roc_auc_score(y_val, val_preds_proba)
    
    print(f"Best parameters for {model_name}: {gs.best_params_}")
    print(f"Validation ROC-AUC for {model_name}: {val_roc_auc:.4f}")
    
    return best_estimator, val_preds_proba

# 运行所有模型
def run_all_models(train_df, val_df, predictors, target):
    models = ['LR', 'KNN', 'SVC', 'DT', 'RFC', 'SGD']
    results = {}
    
    for model_name in models:
        print(f"Running model: {model_name}")
        best_model, val_preds_proba = model_gs(train_df, val_df, predictors, target, model_name)
        results[model_name] = best_model
        
    return results

# 调用时使用
# results = run_all_models(train_df, val_df, predictors, target)