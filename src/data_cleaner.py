"""
AttritionIQ - Data Cleaner Module (Task 2)
Author: Senior ML Project Analyst & Technical Documentation Specialist
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

def run_data_quality_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform rigorous data quality verification across all columns.
    """
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    # Check for constant columns (zero variance)
    constant_columns = {}
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_columns[col] = {
                "unique_values": df[col].unique().tolist(),
                "variance": 0.0
            }
            
    # Check for unique identifier columns
    identifier_columns = [col for col in df.columns if df[col].nunique() == len(df)]
    
    # Verify target variable classes
    target_classes = df['Attrition'].value_counts().to_dict()
    
    # Numerical ranges and sanity check
    numeric_ranges = {}
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        numeric_ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
        
    return {
        "total_nulls": total_nulls,
        "null_counts": null_counts,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "identifier_columns": identifier_columns,
        "target_classes": target_classes,
        "numeric_ranges": numeric_ranges
    }

def clean_attrition_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Execute justified data cleaning operations on the raw DataFrame.
    
    Cleaning Operations:
    1. Prune zero-variance constant features: 'EmployeeCount', 'Over18', 'StandardHours'.
       Rationale: Constant features contain zero statistical variance and cannot contribute
       predictive power to any machine learning algorithm.
    2. Prune uninformative identifier: 'EmployeeNumber'.
       Rationale: Unique row index that carries no generalized domain signal and risks 
       spurious split memorization/overfitting in decision trees.
    """
    df_clean = df.copy()
    
    audit_log = []
    
    columns_to_prune = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    
    for col in columns_to_prune:
        if col in df_clean.columns:
            if col in ['EmployeeCount', 'Over18', 'StandardHours']:
                reason = "Zero-variance constant feature across all 1,470 records."
            else:
                reason = "Arbitrary unique identifier sequence that induces memorization and overfitting."
            
            df_clean.drop(columns=[col], inplace=True)
            audit_log.append({
                "column": col,
                "action": "DROPPED",
                "justification": reason
            })
            
    return df_clean, audit_log

def export_cleaned_dataset(df_clean: pd.DataFrame, output_path: str = "data/processed/cleaned_hr_employee_attrition.csv") -> None:
    """
    Save the cleaned dataset to CSV (standard comma-separated format).
    """
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully exported to: {output_path}")

if __name__ == "__main__":
    try:
        from src.data_loader import load_dataset
    except ImportError:
        from data_loader import load_dataset
    df = load_dataset()
    audit = run_data_quality_audit(df)
    print("Quality Audit Summary:")
    print(f"  Total Missing Values: {audit['total_nulls']}")
    print(f"  Total Duplicate Rows: {audit['duplicate_rows']}")
    print(f"  Constant Columns: {list(audit['constant_columns'].keys())}")
    print(f"  Identifier Columns: {audit['identifier_columns']}")
    print(f"  Target Classes: {audit['target_classes']}")
    
    df_clean, log = clean_attrition_data(df)
    print(f"\nCleaned DataFrame Shape: {df_clean.shape}")
    export_cleaned_dataset(df_clean)
