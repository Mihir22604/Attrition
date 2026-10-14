"""
AttritionIQ - Feature Engineering and Preprocessing Pipeline (Task 4)
Author: Senior Machine Learning Project Analyst & Technical Documentation Specialist
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Explicit Feature Taxonomy Specification
NUMERICAL_FEATURES: List[str] = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

ORDINAL_FEATURES: List[str] = [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance'
]

NOMINAL_CATEGORICAL_FEATURES: List[str] = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

TARGET_COLUMN: str = 'Attrition'

def build_column_transformer(scale_numericals: bool = True) -> ColumnTransformer:
    """
    Construct Scikit-Learn ColumnTransformer grouping transformations by feature type.
    Strictly preserves leakage-free design.
    """
    transformers = []
    
    # Numerical transformer: Standard scaling or passthrough
    if scale_numericals:
        transformers.append(('num', StandardScaler(), NUMERICAL_FEATURES))
    else:
        transformers.append(('num', 'passthrough', NUMERICAL_FEATURES))
        
    # Ordinal transformer: passthrough (retaining natural integer order 1..4, 1..5)
    transformers.append(('ord', 'passthrough', ORDINAL_FEATURES))
    
    # Nominal categorical transformer: OneHotEncoder with drop='first' to prevent dummy variable trap
    nominal_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    transformers.append(('nom', nominal_transformer, NOMINAL_CATEGORICAL_FEATURES))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor

def prepare_feature_data(
    input_csv: str = "data/processed/cleaned_hr_employee_attrition.csv",
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute end-to-end Task 4 pipeline:
    1. Separate features X and target y.
    2. Binary encode target (No: 0, Yes: 1).
    3. Stratified 80:20 train-test partition.
    4. Fit ColumnTransformer strictly on X_train.
    5. Transform X_train and X_test without target leakage.
    6. Extract expanded feature names.
    """
    df = pd.read_csv(input_csv)
    
    # 1. Target separation & encoding
    X = df.drop(columns=[TARGET_COLUMN])
    y_raw = df[TARGET_COLUMN]
    y = y_raw.map({'No': 0, 'Yes': 1}).astype(int)
    
    # 2. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Verify split distributions
    train_dist = y_train.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()
    
    # 3. Assemble and fit preprocessor strictly on X_train
    preprocessor = build_column_transformer(scale_numericals=True)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Extract feature names after One-Hot Encoding
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        # Fallback extraction if older sklearn
        ohe = preprocessor.named_transformers_['nom']
        ohe_cols = ohe.get_feature_names_out(NOMINAL_CATEGORICAL_FEATURES).tolist()
        feature_names = NUMERICAL_FEATURES + ORDINAL_FEATURES + ohe_cols
        
    # Convert to DataFrames for inspection and modeling
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    
    return {
        'X_raw': X,
        'y_raw': y,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_transformed': X_train_transformed,
        'X_test_transformed': X_test_transformed,
        'X_train_df': X_train_df,
        'X_test_df': X_test_df,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'train_counts': train_counts,
        'test_counts': test_counts,
        'train_dist': train_dist,
        'test_dist': test_dist,
        'n_features_in': len(X.columns),
        'n_features_out': len(feature_names)
    }

def export_preprocessed_artifacts(
    prep_data: Dict[str, Any],
    output_dir: str = "data/processed"
) -> None:
    """
    Persist transformed datasets and preprocessor pipeline object for modeling.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save transformed training & testing datasets
    train_export = prep_data['X_train_df'].copy()
    train_export['Attrition'] = prep_data['y_train']
    train_export.to_csv(os.path.join(output_dir, "train_features_preprocessed.csv"), index=False)
    
    test_export = prep_data['X_test_df'].copy()
    test_export['Attrition'] = prep_data['y_test']
    test_export.to_csv(os.path.join(output_dir, "test_features_preprocessed.csv"), index=False)
    
    # Save fitted preprocessor
    preprocessor_path = os.path.join(output_dir, "fitted_preprocessor.joblib")
    joblib.dump(prep_data['preprocessor'], preprocessor_path)
    
    print(f"Preprocessed artifacts successfully exported to: {output_dir}")
    print(f"Fitted Preprocessor serialized to: {preprocessor_path}")

if __name__ == '__main__':
    prep_data = prepare_feature_data()
    print("Task 4 Preprocessing Execution Complete!")
    print(f"X_train shape: {prep_data['X_train_df'].shape}")
    print(f"X_test shape:  {prep_data['X_test_df'].shape}")
    print(f"Train Class Counts: {prep_data['train_counts']} (Pos Rate: {prep_data['train_dist'][1]*100:.2f}%)")
    print(f"Test Class Counts:  {prep_data['test_counts']} (Pos Rate: {prep_data['test_dist'][1]*100:.2f}%)")
    print(f"Total Preprocessed Features: {prep_data['n_features_out']}")
    print(f"Preprocessed Feature Names Sample: {prep_data['feature_names'][:10]} ...")
    export_preprocessed_artifacts(prep_data)
