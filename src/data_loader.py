"""
AttritionIQ - Data Loader Module (Task 1)
Author: Senior ML Project Analyst & Technical Documentation Specialist
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

def load_dataset(filepath: str = "WA_Fn-UseC_-HR-Employee-Attrition.csv") -> pd.DataFrame:
    """
    Load the HR Employee Attrition dataset.
    
    CRITICAL IMPLEMENTATION NOTE:
    Although the file carries a '.csv' extension, it is tab-separated ('\\t').
    Explicit delimiter specification is mandatory to prevent single-column corruption.
    """
    try:
        # First attempt explicit tab delimiter
        df = pd.read_csv(filepath, sep='\t')
        if df.shape[1] == 1:
            # Fallback to python sniffer if tab didn't split columns
            df = pd.read_csv(filepath, sep=None, engine='python')
    except Exception as e:
        # Fallback to engine sniffer
        df = pd.read_csv(filepath, sep=None, engine='python')
        
    return df

def get_structural_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive structural metadata for the raw dataset.
    """
    rows, cols = df.shape
    
    # Categorize columns
    target_col = "Attrition"
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    identifier_cols = [c for c in df.columns if df[c].nunique() == rows]
    
    nominal_cols = [
        "BusinessTravel", "Department", "EducationField", 
        "Gender", "JobRole", "MaritalStatus", "OverTime"
    ]
    
    ordinal_cols = [
        "Education", "EnvironmentSatisfaction", "JobInvolvement", 
        "JobLevel", "JobSatisfaction", "PerformanceRating", 
        "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"
    ]
    
    continuous_discrete_cols = [
        "Age", "DailyRate", "DistanceFromHome", "HourlyRate", 
        "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", 
        "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear", 
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", 
        "YearsWithCurrManager"
    ]
    
    col_summary = []
    for col in df.columns:
        col_summary.append({
            "column_name": col,
            "dtype": str(df[col].dtype),
            "nunique": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
            "sample_values": df[col].dropna().unique()[:3].tolist()
        })
        
    return {
        "dimensions": (rows, cols),
        "target_col": target_col,
        "constant_cols": constant_cols,
        "identifier_cols": identifier_cols,
        "nominal_cols": nominal_cols,
        "ordinal_cols": ordinal_cols,
        "continuous_discrete_cols": continuous_discrete_cols,
        "column_summary": col_summary
    }

if __name__ == "__main__":
    df = load_dataset()
    summary = get_structural_summary(df)
    print(f"Loaded DataFrame successfully: Shape = {summary['dimensions']}")
    print(f"Target Column: {summary['target_col']}")
    print(f"Constant Columns: {summary['constant_cols']}")
    print(f"Identifier Columns: {summary['identifier_cols']}")
