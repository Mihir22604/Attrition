# TASK 2 DETAILED REPORT: Data Cleaning and Quality Verification

**Project Title:** AttritionIQ — Intelligent Employee Attrition Prediction System  
**Task ID:** Task 2 — Data Cleaning and Quality Verification  
**Status:** COMPLETED & VERIFIED  
**Artifact Folder:** `task_reports/`  
**Cleaned Dataset Path:** `data/processed/cleaned_hr_employee_attrition.csv`  
**Author / Role:** Senior Machine Learning Project Analyst & Technical Documentation Specialist  
**Execution Environment:** Python 3.12 / Pandas 2.2 / Scikit-Learn / NumPy  
**Input Data:** Raw Ingested DataFrame from Task 1 ($1,470$ rows $\times$ $35$ columns)  

---

## 1. Executive Summary and Task Objectives

The primary objective of **Task 2: Data Cleaning and Quality Verification** is to audit data integrity, identify and remove non-informative features, prevent algorithmic target leakage, validate target class representations, and produce a cleaned, production-grade dataset for downstream exploratory analysis and predictive modeling.

### Core Objectives Successfully Delivered:
1. **Missing Value Audit:** Audited all 35 columns for null, NaN, or blank entries (confirmed: **0 missing values**).
2. **Duplicate Record Audit:** Evaluated row-level duplication across all 1,470 observations (confirmed: **0 duplicates**).
3. **Constant / Zero-Variance Feature Pruning:** Identified and pruned 3 features (`EmployeeCount`, `Over18`, `StandardHours`) that possess zero variance ($\sigma^2 = 0$).
4. **Identifier Pruning (Leakage/Overfitting Prevention):** Identified and pruned 1 high-cardinality identifier (`EmployeeNumber`) to prevent decision tree memorization.
5. **Target Integrity Validation:** Verified that `Attrition` contains strictly two binary outcome classes (`Yes` and `No`) with zero anomalous tokens.
6. **Value Range & Anomaly Validation:** Verified empirical ranges for numerical, ordinal Likert, and categorical features.
7. **Cleaned Dataset Export:** Produced and exported the cleaned dataset of shape **$(1470, 31)$** to `data/processed/cleaned_hr_employee_attrition.csv`.

---

## 2. Rigorous Data Quality Audits

### 2.1 Missing Values Audit Across All 35 Attributes
A programmatic sweep across all 35 columns was executed using `df.isnull().sum()`. The audit returned:
- **Total Missing Values in Raw Dataset:** **$0$**
- **Completeness Rate:** **$100.00\%$**

```
Missing Value Verification Table (Summary):
- All 35 columns: 0 nulls / 1,470 valid non-null entries
- Imputation requirement: No statistical imputation (mean/median/mode) required for primary dataset.
```

### 2.2 Duplicate Record Audit
A row-level uniqueness audit was executed using `df.duplicated().sum()` across all 35 feature dimensions:
- **Duplicate Rows Identified:** **$0$**
- **Uniqueness Rate:** **$100.00\%$** ($1,470$ unique employee records)

### 2.3 Constant (Zero-Variance) Features Audit
Features that remain invariant across all observations contribute zero information entropy ($H(X) = 0$) and zero variance ($\sigma^2 = 0$). Three features were detected:

| Feature Name | Verified Dtype | Observed Unique Value | Unique Count | Percent Invariant |
| :--- | :---: | :---: | :---: | :---: |
| `EmployeeCount` | `int64` | `1` | 1 | $100.0\%$ |
| `Over18` | `object` | `'Y'` | 1 | $100.0\%$ |
| `StandardHours` | `int64` | `80` | 1 | $100.0\%$ |

### 2.4 High-Cardinality Identifier Audit
The column `EmployeeNumber` was audited for statistical utility:
- **Unique Cardinality:** $1,470$ unique integers (spanning $1$ to $2,068$).
- **Cardinality Ratio:** $\frac{1,470}{1,470} = 1.0$ ($100\%$ unique per row).
- **Analytical Assessment:** Arbitrary database surrogate primary key. Retaining it would allow tree-based algorithms to memorize individual employee IDs rather than learning generalizable workforce patterns.

### 2.5 Target Variable Validation
The target variable `Attrition` was audited for anomalous classes, whitespace errors, or nulls:
- **Unique Values:** Exactly `['Yes', 'No']`
- **Class `'No'`:** $1,233$ observations ($83.88\%$)
- **Class `'Yes'`:** $237$ observations ($16.12\%$)
- **Null / Unrecognized Classes:** $0$

### 2.6 Numerical Range and Sanity Validation
Every numerical and ordinal feature was validated against reasonable enterprise bounds:
- `Age`: Observed $18$ to $60$ years (all legal adult workforce age).
- `MonthlyIncome`: Observed $\$1,009$ to $\$19,999$ (no negative salaries).
- `TotalWorkingYears`: Observed $0$ to $40$ years (satisfies $\text{TotalWorkingYears} \le \text{Age} - 18$).
- `YearsAtCompany`: Observed $0$ to $40$ years (satisfies $\text{YearsAtCompany} \le \text{TotalWorkingYears}$).
- `WorkLifeBalance`, `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`: Strictly bounded within $[1, 4]$.
- `JobLevel`: Strictly bounded within $[1, 5]$.
- `Education`: Strictly bounded within $[1, 5]$.
- **Data Anomaly Noted:** `PerformanceRating` contains only values $3$ ("Excellent") and $4$ ("Outstanding"). Ratings $1$ ("Low") and $2$ ("Good") do not appear in this dataset.

---

## 3. Formal Data Cleaning Decisions and Justification Log

In accordance with assignment mandates, every cleaning operation is formally justified rather than applied arbitrarily:

| Column Name | Action Taken | Formal Mathematical & Domain Justification |
| :--- | :---: | :--- |
| `EmployeeCount` | **DROPPED** | **Zero Statistical Variance ($\sigma^2 = 0$):** Every record has the scalar value `1`. Provides zero discriminative information to any linear or tree-based machine learning model. Dropping reduces dimensional overhead. |
| `Over18` | **DROPPED** | **Zero Statistical Variance ($\sigma^2 = 0$):** Every record contains the constant string `'Y'`. Imparts zero entropy ($H(X)=0$) and zero predictive capacity. |
| `StandardHours` | **DROPPED** | **Zero Statistical Variance ($\sigma^2 = 0$):** Every employee is assigned `80` standard bi-weekly hours. Zero variance prevents any correlation with voluntary turnover. |
| `EmployeeNumber` | **DROPPED** | **Identifier Memorization & Overfitting Prevention:** As an arbitrary sequence key with $1,470$ unique values, retaining it risks decision tree leaf memorization, leading to severe train-test generalization collapse. |

---

## 4. Characteristics of the Cleaned Dataset

Following execution of the cleaning operations:
- **Raw Feature Count:** $35$ columns
- **Columns Pruned:** $4$ columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`)
- **Cleaned Feature Count:** **$31$ columns** (30 candidate predictors + 1 target variable)
- **Cleaned Record Count:** **$1,470$ rows** (zero rows dropped)
- **Cleaned Dimensions:** **$(1470, 31)$**
- **Output Storage Path:** [`data/processed/cleaned_hr_employee_attrition.csv`](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/data/processed/cleaned_hr_employee_attrition.csv)

### 4.1 Retained Column Inventory in Cleaned Dataset (31 Columns)
1. `Age` (Continuous Numerical)
2. `Attrition` (**Target Variable**)
3. `BusinessTravel` (Nominal Categorical)
4. `DailyRate` (Continuous Numerical)
5. `Department` (Nominal Categorical)
6. `DistanceFromHome` (Discrete Numerical)
7. `Education` (Ordered Ordinal)
8. `EducationField` (Nominal Categorical)
9. `EnvironmentSatisfaction` (Ordered Ordinal)
10. `Gender` (Nominal Categorical)
11. `HourlyRate` (Continuous Numerical)
12. `JobInvolvement` (Ordered Ordinal)
13. `JobLevel` (Ordered Ordinal)
14. `JobRole` (Nominal Categorical)
15. `JobSatisfaction` (Ordered Ordinal)
16. `MaritalStatus` (Nominal Categorical)
17. `MonthlyIncome` (Continuous Numerical)
18. `MonthlyRate` (Continuous Numerical)
19. `NumCompaniesWorked` (Discrete Numerical)
20. `OverTime` (Nominal Categorical)
21. `PercentSalaryHike` (Continuous Numerical)
22. `PerformanceRating` (Ordered Ordinal)
23. `RelationshipSatisfaction` (Ordered Ordinal)
24. `StockOptionLevel` (Ordered Ordinal)
25. `TotalWorkingYears` (Continuous Numerical)
26. `TrainingTimesLastYear` (Discrete Numerical)
27. `WorkLifeBalance` (Ordered Ordinal)
28. `YearsAtCompany` (Continuous Numerical)
29. `YearsInCurrentRole` (Continuous Numerical)
30. `YearsSinceLastPromotion` (Continuous Numerical)
31. `YearsWithCurrManager` (Continuous Numerical)

---

## 5. Verification and Quality Checks Completed

- [x] Zero null values verified across all columns ($0$ missing).
- [x] Zero duplicate rows verified across all $1,470$ instances.
- [x] Constant zero-variance features identified and verified (`EmployeeCount`, `Over18`, `StandardHours`).
- [x] Identifier column identified and verified (`EmployeeNumber`).
- [x] Target variable audited (`Attrition`: 1,233 'No', 237 'Yes').
- [x] Value ranges audited for biological, organizational, and Likert boundaries.
- [x] Formal justification log documented for all dropped columns.
- [x] Pruning executed to produce cleaned DataFrame of shape $(1470, 31)$.
- [x] Cleaned dataset exported to `data/processed/cleaned_hr_employee_attrition.csv`.
- [x] Module `src/data_cleaner.py` verified and operational.
- [x] Master notebook `notebooks/AttritionIQ_Workflow.ipynb` updated with Task 2 cells.

---

## 6. Handoff to Task 3 and Task 4

The exported cleaned dataset ([`data/processed/cleaned_hr_employee_attrition.csv`](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/data/processed/cleaned_hr_employee_attrition.csv)) provides a clean, validated foundation ready for:
1. **Task 3: Exploratory Data Analysis (EDA):** Generating the 8+ mandatory visualizations, computing cohort attrition rates, and formulating the 5 supported insights.
2. **Task 4: Feature Engineering & Preprocessing Pipeline:** Encoding nominals via `OneHotEncoder`, scaling continuous features, and establishing stratified 80:20 training/test splits.
