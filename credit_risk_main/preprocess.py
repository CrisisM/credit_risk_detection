import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data, target_column, test_size=0.3):
    
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

    # Step 7: 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def separate_features_and_target(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def preprocess_numeric_features(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # 缺失值用均值填充
        ('scaler', StandardScaler())  # 标准化
    ])
    return numeric_transformer, numeric_features

def preprocess_categorical_features(X):
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 缺失值用众数填充
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot 编码
    ])
    return categorical_transformer, categorical_features

def build_preprocessor(numeric_transformer, categorical_transformer, numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def apply_preprocessor(preprocessor, X):
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)