import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from collections import Counter
from imblearn.over_sampling import SMOTE

def preprocess_data(data, target_column, test_size=0.3, imbalance_threshold=0.1):
    """
    Complete data preprocessing workflow including:
    1. Separating features and target
    2. Standardizing numerical features
    3. One-Hot encoding categorical features
    4. Checking if SMOTE is needed to balance the classes
    5. Splitting data into training and testing sets
    
    Parameters:
    - data: Input DataFrame
    - target_column: The target column (indicating default or not)
    - test_size: The proportion of the test set, default is 0.3
    - imbalance_threshold: Threshold for class imbalance, apply SMOTE if the minority class ratio is below this threshold
    
    Returns:
    - X_train: Training set features
    - X_test: Test set features
    - y_train: Training set labels
    - y_test: Test set labels
    """
    
    # Step 1: Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Step 2: Identify numerical and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Step 3: Preprocessing for numerical features - Imputation and Standardization
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())  # Standardize features
    ])

    # Step 4: Preprocessing for categorical features - Imputation and One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot encoding
    ])

    # Step 5: Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Step 6: Preprocess the entire dataset
    X_preprocessed = preprocessor.fit_transform(X)

    # Step 7: Calculate class distribution
    class_distribution = Counter(y)
    total_samples = len(y)
    minority_class_ratio = min(class_distribution.values()) / total_samples
    
    print(f"Class distribution: {class_distribution}")
    print(f"Minority class ratio: {minority_class_ratio:.2%}")

    # Step 8: Apply SMOTE if necessary
    if minority_class_ratio < imbalance_threshold:
        print(f"Minority class ratio ({minority_class_ratio:.2%}) is below the threshold ({imbalance_threshold * 100:.2f}%), applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_preprocessed, y)
        print(f"Balanced class distribution: {Counter(y_balanced)}")
    else:
        print(f"Minority class ratio ({minority_class_ratio:.2%}) is above the threshold, skipping SMOTE.")
        X_balanced, y_balanced = X_preprocessed, y

    # Step 9: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def separate_features_and_target(data, target_column):
    """
    Separate input data into features and target column.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def preprocess_numeric_features(X):
    """
    Preprocess numerical features, including missing value imputation and standardization.
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())  # Standardize features
    ])
    return numeric_transformer, numeric_features

def preprocess_categorical_features(X):
    """
    Preprocess categorical features, including missing value imputation and One-Hot encoding.
    """
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot encoding
    ])
    return categorical_transformer, categorical_features

def build_preprocessor(numeric_transformer, categorical_transformer, numeric_features, categorical_features):
    """
    Build a ColumnTransformer for data preprocessing that integrates steps for numerical and categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def apply_preprocessor(preprocessor, X):
    """
    Apply the preprocessing steps to the data.
    """
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# New: Random Oversampling
def random_oversample(X, y):
    """
    Use random oversampling to balance class proportions in the dataset.
    
    Parameters:
    - X: Feature data
    - y: Label data
    
    Returns:
    - X_resampled: Oversampled feature data
    - y_resampled: Oversampled label data
    """
    from sklearn.utils import resample
    
    # Combine the data for easier manipulation
    data = pd.concat([X, y], axis=1)
    
    # Separate majority and minority classes
    majority_class = data[y == 0]
    minority_class = data[y == 1]
    
    # Oversample the minority class
    minority_class_upsampled = resample(minority_class,
                                        replace=True,  # Allow resampling with replacement
                                        n_samples=len(majority_class),  # Match the majority class sample size
                                        random_state=42)
    
    # Combine majority class and oversampled minority class
    upsampled_data = pd.concat([majority_class, minority_class_upsampled])
    
    # Separate features and labels again
    X_resampled = upsampled_data.drop(columns=y.name)
    y_resampled = upsampled_data[y.name]
    
    print(f"Random Oversample - Class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

# New: SMOTE Oversampling
def smote_oversample(X, y):
    """
    Use SMOTE to generate new samples for the minority class to balance class proportions.
    
    Parameters:
    - X: Feature data
    - y: Label data
    
    Returns:
    - X_resampled: SMOTE-generated feature data
    - y_resampled: SMOTE-generated label data
    """
    # Adjust n_neighbors to fit small sample sizes
    smote = SMOTE(random_state=42, k_neighbors=min(1, len(X[y == 1]) - 1))  # Avoid n_neighbors > minority class samples
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"SMOTE Oversample - Class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled
