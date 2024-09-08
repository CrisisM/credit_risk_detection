# %%
from sklearn.ensemble import AdaBoostClassifier

def AdaBoostModel(train_df, val_df, predictors, target, random_state=2024, algorithm='SAMME.R', learning_rate=0.8, n_estimators=100):
    """
    封装的 AdaBoost 分类器训练和预测函数。

    参数:
    - train_df: 训练数据集的 DataFrame。
    - val_df: 验证数据集的 DataFrame。
    - predictors: 特征列的名称列表。
    - target: 目标列的名称。
    - random_state: 随机数种子，确保结果的可重复性。
    - algorithm: 使用的算法类型，'SAMME' 或 'SAMME.R'（默认是 'SAMME.R'）。
    - learning_rate: 每棵树对总结果的贡献程度。
    - n_estimators: 基础学习器（通常是决策树）的数量。

    返回:
    - preds: 验证集上的预测结果。
    - clf: 训练好的 AdaBoost 模型。
    """
    
    # 初始化 AdaBoost 模型
    clf = AdaBoostClassifier(random_state=random_state,
                             algorithm=algorithm,
                             learning_rate=learning_rate,
                             n_estimators=n_estimators)
    
    # 训练模型
    clf.fit(train_df[predictors], train_df[target].values)
    
    # 在验证集上进行预测
    preds = clf.predict(val_df[predictors])
    
    pass
    #return preds, clf


# %%
from catboost import CatBoostClassifier

def CatBoostModel(train_df, val_df, predictors, target, iterations=500, learning_rate=0.02, depth=12, eval_metric='AUC',
                  random_seed=2024, bagging_temperature=0.2, od_type='Iter', metric_period=50, od_wait=100, verbose=True):
    """
    封装的 CatBoost 分类器训练和预测函数。

    参数:
    - train_df: 训练数据集的 DataFrame。
    - val_df: 验证数据集的 DataFrame。
    - predictors: 特征列的名称列表。
    - target: 目标列的名称。
    - iterations: 训练迭代次数（决策树的数量）。
    - learning_rate: 学习率。
    - depth: 每棵决策树的最大深度。
    - eval_metric: 评估指标，默认是 'AUC'。
    - random_seed: 随机数种子，确保结果的可重复性。
    - bagging_temperature: 控制子采样的多样性。
    - od_type: 过拟合检测的类型，默认是 'Iter'。
    - metric_period: 评估指标输出的频率。
    - od_wait: 早停等待的轮数。
    - verbose: 控制训练过程中的详细输出。

    返回:
    - preds: 验证集上的预测结果。
    - clf: 训练好的 CatBoost 模型。
    """
    
    # 初始化 CatBoost 模型
    clf = CatBoostClassifier(iterations=iterations,
                             learning_rate=learning_rate,
                             depth=depth,
                             eval_metric=eval_metric,
                             random_seed=random_seed,
                             bagging_temperature=bagging_temperature,
                             od_type=od_type,
                             metric_period=metric_period,
                             od_wait=od_wait)
    
    # 训练模型
    clf.fit(train_df[predictors], train_df[target].values, verbose=verbose)
    
    # 在验证集上进行预测
    preds = clf.predict(val_df[predictors])
    
    return preds, clf


# %%
import xgboost as xgb

def XGBoostModel(train_df, val_df, predictors, target, num_boost_round=500, early_stopping_rounds=50, verbose_eval=100,
                 eta=0.039, max_depth=3, subsample=0.8, colsample_bytree=0.9, eval_metric='auc', random_state=2024):
    """
    封装的 XGBoost 分类器训练和预测函数。

    参数:
    - train_df: 训练数据集的 DataFrame。
    - val_df: 验证数据集的 DataFrame。
    - predictors: 特征列的名称列表。
    - target: 目标列的名称。
    - num_boost_round: 训练的总轮数（Boosting 的轮数）。
    - early_stopping_rounds: 早停的轮数，若在此轮数内模型性能未提升，则停止训练。
    - verbose_eval: 控制训练过程中的详细输出频率。
    - eta: 学习率，控制每棵树对最终结果的影响。
    - max_depth: 每棵树的最大深度。
    - subsample: 训练样本的子样本比例。
    - colsample_bytree: 每棵树的特征子样本比例。
    - eval_metric: 评估指标，通常为 'auc' 用于二分类任务。
    - random_state: 随机数种子，确保结果的可重复性。

    返回:
    - model: 训练好的 XGBoost 模型。
    """
    
    # 准备训练和验证数据集
    dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
    dvalid = xgb.DMatrix(val_df[predictors], val_df[target].values)
    
    # 监控训练过程中的评估指标
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    # 设置 XGBoost 参数
    params = {
        'objective': 'binary:logistic',
        'eta': eta,
        'silent': True,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'eval_metric': eval_metric,
        'random_state': random_state
    }
    
    # 训练模型
    model = xgb.train(params, 
                      dtrain, 
                      num_boost_round=num_boost_round, 
                      evals=watchlist, 
                      early_stopping_rounds=early_stopping_rounds, 
                      maximize=True, 
                      verbose_eval=verbose_eval)
    
    return model


# %%
import lightgbm as lgb
import gc

def LightGBMModel(train_df, val_df, predictors, target, categorical_features, params, num_boost_round=500, early_stopping_rounds=50, verbose_eval=100):
    """
    封装的 LightGBM 分类器训练和预测函数。

    参数:
    - train_df: 训练数据集的 DataFrame。
    - val_df: 验证数据集的 DataFrame。
    - predictors: 特征列的名称列表。
    - target: 目标列的名称。
    - categorical_features: 分类特征列的名称列表。
    - params: LightGBM 模型的参数字典。
    - num_boost_round: Boosting 的总轮数。
    - early_stopping_rounds: 早停的轮数。
    - verbose_eval: 控制训练过程中的详细输出频率。

    返回:
    - model: 训练好的 LightGBM 模型。
    - final_auc_train: 最终训练集上的 AUC 值。
    - final_auc_valid: 最终验证集上的 AUC 值。
    """

    # 准备训练和验证数据集
    dtrain = lgb.Dataset(train_df[predictors].values, 
                         label=train_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical_features)
    
    dvalid = lgb.Dataset(val_df[predictors].values,
                         label=val_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical_features)
    
    # 定义评估结果字典
    evals_results = {}
    
    # 定义回调函数
    from lightgbm import early_stopping, log_evaluation, record_evaluation

    callbacks = [
        early_stopping(stopping_rounds=early_stopping_rounds),
        log_evaluation(period=verbose_eval),
        record_evaluation(evals_results)
    ]
    
    # 训练模型
    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train', 'valid'], 
                      callbacks=callbacks, 
                      num_boost_round=num_boost_round)
    
    # 打印最终一轮的 AUC 值
    final_auc_train = evals_results['train']['auc'][-1]
    final_auc_valid = evals_results['valid']['auc'][-1]

    # print(f"Final AUC on train set: {final_auc_train:.4f}")
    # print(f"Final AUC on valid set: {final_auc_valid:.4f}")

    # 删除 dvalid 对象以释放内存
    del dvalid

    # 进行垃圾回收以释放内存
    gc.collect()

    return model, final_auc_train, final_auc_valid



