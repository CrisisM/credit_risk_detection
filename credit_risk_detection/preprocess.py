import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from collections import Counter

def preprocess_data(data, target_column, test_size=0.3, imbalance_threshold=0.1):
    """
    完整的数据预处理流程，包括：
    1. 分离特征和目标列
    2. 数值特征的标准化
    3. 类别特征的 One-Hot 编码
    4. 判断是否需要使用 SMOTE 进行平衡
    5. 拆分训练集和测试集
    
    参数:
    - data: 输入的 DataFrame 数据
    - target_column: 目标列（即表示违约与否的标签列的名称）
    - test_size: 测试集比例，默认为 0.3
    - imbalance_threshold: 类别不平衡的阈值，少数类比例低于此值则使用 SMOTE
    
    返回:
    - X_train: 训练集特征
    - X_test: 测试集特征
    - y_train: 训练集标签
    - y_test: 测试集标签
    """
    
    # Step 1: 分离特征和目标
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Step 2: 区分数值型特征和类别型特征
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Step 3: 数值特征处理 - 填补缺失值 + 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # 缺失值用均值填充
        ('scaler', StandardScaler())  # 标准化
    ])

    # Step 4: 类别特征处理 - 填补缺失值 + One-Hot 编码
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 缺失值用众数填充
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot 编码
    ])

    # Step 5: 构建完整的预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Step 6: 对整个数据集进行预处理
    X_preprocessed = preprocessor.fit_transform(X)

    # Step 7: 计算类别分布
    class_distribution = Counter(y)
    total_samples = len(y)
    minority_class_ratio = min(class_distribution.values()) / total_samples
    
    print(f"Class distribution: {class_distribution}")
    print(f"Minority class ratio: {minority_class_ratio:.2%}")

    # Step 8: 判断是否需要使用 SMOTE
    if minority_class_ratio < imbalance_threshold:
        print(f"Minority class ratio ({minority_class_ratio:.2%}) is below the threshold ({imbalance_threshold * 100:.2f}%), applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_preprocessed, y)
        print(f"Balanced class distribution: {Counter(y_balanced)}")
    else:
        print(f"Minority class ratio ({minority_class_ratio:.2%}) is above the threshold, skipping SMOTE.")
        X_balanced, y_balanced = X_preprocessed, y

    # Step 9: 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def separate_features_and_target(data, target_column):
    """
    将输入数据集分离为特征和目标列。
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def preprocess_numeric_features(X):
    """
    预处理数值型特征，包括缺失值填补和标准化。
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # 缺失值用均值填充
        ('scaler', StandardScaler())  # 标准化
    ])
    return numeric_transformer, numeric_features

def preprocess_categorical_features(X):
    """
    预处理类别型特征，包括缺失值填补和 One-Hot 编码。
    """
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 缺失值用众数填充
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot 编码
    ])
    return categorical_transformer, categorical_features

def build_preprocessor(numeric_transformer, categorical_transformer, numeric_features, categorical_features):
    """
    构建数据预处理的 ColumnTransformer，集成数值和类别特征的处理步骤。
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def apply_preprocessor(preprocessor, X):
    """
    对数据应用预处理步骤。
    """
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed

def split_data(X, y, test_size=0.3, random_state=42):
    """
    拆分数据集为训练集和测试集。
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 新增：随机过采样
def random_oversample(X, y):
    """
    使用随机过采样方法，平衡数据集中的类别比例。
    
    参数:
    - X: 特征数据
    - y: 标签数据
    
    返回:
    - X_resampled: 过采样后的特征数据
    - y_resampled: 过采样后的标签数据
    """
    from sklearn.utils import resample
    
    # 将数据组合起来，方便操作
    data = pd.concat([X, y], axis=1)
    
    # 拆分多数类和少数类
    majority_class = data[y == 0]
    minority_class = data[y == 1]
    
    # 过采样少数类
    minority_class_upsampled = resample(minority_class,
                                        replace=True,  # 允许重复抽样
                                        n_samples=len(majority_class),  # 样本数与多数类相同
                                        random_state=42)
    
    # 合并多数类和过采样后的少数类
    upsampled_data = pd.concat([majority_class, minority_class_upsampled])
    
    # 重新分离特征和标签
    X_resampled = upsampled_data.drop(columns=y.name)
    y_resampled = upsampled_data[y.name]
    
    print(f"Random Oversample - Class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

# 新增：SMOTE过采样
def smote_oversample(X, y):
    """
    使用SMOTE方法生成新的少数类样本，以平衡类别比例。
    
    参数:
    - X: 特征数据
    - y: 标签数据
    
    返回:
    - X_resampled: SMOTE后的特征数据
    - y_resampled: SMOTE后的标签数据
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"SMOTE Oversample - Class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled