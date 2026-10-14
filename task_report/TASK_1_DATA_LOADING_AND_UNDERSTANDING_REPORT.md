# TASK 1 DETAILED REPORT: Data Loading and Understanding

**Project Title:** AttritionIQ — Intelligent Employee Attrition Prediction System  
**Task ID:** Task 1 — Data Loading and Dataset Understanding  
**Status:** COMPLETED & VERIFIED  
**Artifact Folder:** `task_reports/`  
**Author / Role:** Senior Machine Learning Project Analyst & Technical Documentation Specialist  
**Execution Environment:** Python 3.12 / Pandas 2.2 / Scikit-Learn / NumPy  
**Dataset Analyzed:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`  

---

## 1. Executive Summary and Task Objectives

The primary objective of **Task 1: Data Loading and Understanding** is to establish the computational foundation for the AttritionIQ system by ingesting the raw HR Employee Attrition dataset, resolving structural parsing nuances, inspecting dimensions and memory footprint, and cataloging the schema into appropriate analytical feature families.

### Core Objectives Successfully Delivered:
1. **Runtime Verification:** Successfully imported standard data science packages (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`).
2. **Ingestion & Delimiter Resolution:** Detected and addressed the physical tab-delimited (`\t`) structure of `WA_Fn-UseC_-HR-Employee-Attrition.csv`, preventing single-column tabular degradation.
3. **Dimensional Verification:** Confirmed dataset dimensions as exactly **1,470 records (rows)** and **35 attributes (columns)**.
4. **Inspection of Extremes:** Validated the first five (`head`) and last five (`tail`) observations.
5. **Schema & Cardinality Profiling:** Generated comprehensive datatypes, non-null counts, and uniqueness metrics across all 35 columns using `info()`, `describe()`, and `nunique()`.
6. **Feature Classification:** Formalized the taxonomy of all 35 columns into continuous numerical, discrete numerical, ordinal Likert ratings, nominal categories, constant zero-variance features, unique identifiers, and the target variable.

---

## 2. Technical Methodology and Ingestion Strategy

### 2.1 Critical Implementation Discovery: Tab-Separated Structure
Inspection of `WA_Fn-UseC_-HR-Employee-Attrition.csv` revealed that despite the `.csv` file extension, the file is physically delimited by **horizontal tab characters (`\t`)** rather than standard commas.

```
Ingestion Risk Assessment:
Standard pd.read_csv('...') ──► Assumes ',' ──► Fails: Yields Shape (1470, 1) [35 concatenated columns]
Correct Ingestion Strategy   ──► sep='\t'    ──► Success: Yields Shape (1470, 35) [All columns parsed]
```

### 2.2 Ingestion Code Implementation
The production loading module in `src/data_loader.py` implements an explicit tab-delimiter parse with an automated engine sniffer fallback:

```python
import pandas as pd

def load_dataset(filepath: str = "WA_Fn-UseC_-HR-Employee-Attrition.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, sep='\t')
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(filepath, sep=None, engine='python')
    return df
```

---

## 3. Empirical Dataset Verification and Schema Profile

### 3.1 Dataset Dimensions and Footprint
- **Total Records (Instances $N$):** $1,470$
- **Total Features (Attributes $D$):** $35$
- **Memory Consumption:** Approximately $402.1+$ KB
- **Missing Value Count:** $0$ ($100\%$ complete data across all cells)
- **Duplicate Row Count:** $0$ duplicate instances

### 3.2 Target Variable Baseline Profile
- **Target Feature Name:** `Attrition`
- **Variable Type:** Binary Nominal String (`'No'`, `'Yes'`)
- **Distribution:**
  - Class `'No'` (Retained Employees): **1,233 records (83.88%)**
  - Class `'Yes'` (Departed Employees): **237 records (16.12%)**
- **Baseline Enterprise Attrition Rate:**
  $$\text{Attrition Rate} = \frac{237}{1,470} \times 100 = 16.1224\% \approx 16.12\%$$
- **Class Imbalance Ratio:** Approximately **5.20 : 1** (majority to minority).

---

## 4. Comprehensive 35-Column Data Schema & Cardinality Audit

The following table provides the audited profile for all 35 columns extracted directly from the verified dataset:

| # | Column Name | Verified Dtype | Non-Null Count | Unique Count | Observed Values / Range | Analytical Classification |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 1 | `Age` | `int64` | 1470 | 43 | 18 to 60 (Median: 36.0) | Continuous Numerical |
| 2 | `Attrition` | `object` | 1470 | 2 | `['Yes', 'No']` | **Target Variable** |
| 3 | `BusinessTravel` | `object` | 1470 | 3 | Travel_Rarely, Travel_Frequently, Non-Travel | Nominal Categorical |
| 4 | `DailyRate` | `int64` | 1470 | 886 | 102 to 1499 (Median: 802.0) | Continuous Numerical |
| 5 | `Department` | `object` | 1470 | 3 | Sales, R&D, Human Resources | Nominal Categorical |
| 6 | `DistanceFromHome` | `int64` | 1470 | 29 | 1 to 29 (Median: 7.0) | Discrete Numerical |
| 7 | `Education` | `int64` | 1470 | 5 | 1 (Below College) to 5 (Doctor) | Ordered Ordinal |
| 8 | `EducationField` | `object` | 1470 | 6 | Life Sciences, Medical, Marketing, etc. | Nominal Categorical |
| 9 | `EmployeeCount` | `int64` | 1470 | 1 | Constant value: `1` | **Constant (Zero-Variance)** |
| 10 | `EmployeeNumber` | `int64` | 1470 | 1470 | 1 to 2068 | **Unique Identifier** |
| 11 | `EnvironmentSatisfaction` | `int64` | 1470 | 4 | 1 (Low) to 4 (Very High) | Ordered Ordinal |
| 12 | `Gender` | `object` | 1470 | 2 | `['Male', 'Female']` | Nominal Categorical |
| 13 | `HourlyRate` | `int64` | 1470 | 71 | 30 to 100 (Median: 66.0) | Continuous Numerical |
| 14 | `JobInvolvement` | `int64` | 1470 | 4 | 1 (Low) to 4 (Very High) | Ordered Ordinal |
| 15 | `JobLevel` | `int64` | 1470 | 5 | 1 (Entry) to 5 (Executive) | Ordered Ordinal |
| 16 | `JobRole` | `object` | 1470 | 9 | Sales Exec, Research Scientist, etc. | Nominal Categorical |
| 17 | `JobSatisfaction` | `int64` | 1470 | 4 | 1 (Low) to 4 (Very High) | Ordered Ordinal |
| 18 | `MaritalStatus` | `object` | 1470 | 3 | Single, Married, Divorced | Nominal Categorical |
| 19 | `MonthlyIncome` | `int64` | 1470 | 1349 | \$1,009 to \$19,999 (Median: \$4,919.0) | Continuous Numerical |
| 20 | `MonthlyRate` | `int64` | 1470 | 1427 | 2094 to 26999 (Median: 14235.5) | Continuous Numerical |
| 21 | `NumCompaniesWorked` | `int64` | 1470 | 10 | 0 to 9 (Median: 2.0) | Discrete Numerical |
| 22 | `Over18` | `object` | 1470 | 1 | Constant value: `'Y'` | **Constant (Zero-Variance)** |
| 23 | `OverTime` | `object` | 1470 | 2 | `['Yes', 'No']` | Nominal Categorical |
| 24 | `PercentSalaryHike` | `int64` | 1470 | 15 | 11% to 25% (Median: 14.0%) | Continuous Numerical |
| 25 | `PerformanceRating` | `int64` | 1470 | 2 | Ratings 3 (Excellent) & 4 (Outstanding) | Ordered Ordinal |
| 26 | `RelationshipSatisfaction` | `int64` | 1470 | 4 | 1 (Low) to 4 (Very High) | Ordered Ordinal |
| 27 | `StandardHours` | `int64` | 1470 | 1 | Constant value: `80` | **Constant (Zero-Variance)** |
| 28 | `StockOptionLevel` | `int64` | 1470 | 4 | 0 to 3 | Ordered Ordinal |
| 29 | `TotalWorkingYears` | `int64` | 1470 | 40 | 0 to 40 (Median: 10.0) | Continuous Numerical |
| 30 | `TrainingTimesLastYear` | `int64` | 1470 | 7 | 0 to 6 (Median: 3.0) | Discrete Numerical |
| 31 | `WorkLifeBalance` | `int64` | 1470 | 4 | 1 (Bad) to 4 (Best) | Ordered Ordinal |
| 32 | `YearsAtCompany` | `int64` | 1470 | 37 | 0 to 40 (Median: 5.0) | Continuous Numerical |
| 33 | `YearsInCurrentRole` | `int64` | 1470 | 19 | 0 to 18 (Median: 3.0) | Continuous Numerical |
| 34 | `YearsSinceLastPromotion`| `int64` | 1470 | 16 | 0 to 15 (Median: 1.0) | Continuous Numerical |
| 35 | `YearsWithCurrManager` | `int64` | 1470 | 18 | 0 to 17 (Median: 3.0) | Continuous Numerical |

---

## 5. Statistical Profile of Key Numerical Attributes

Descriptive summary statistics computed across primary quantitative features:

| Feature Name | Mean | Std Dev | Min | 25% (Q1) | Median | 75% (Q3) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Age` | 36.92 | 9.14 | 18 | 30.0 | 36.0 | 43.0 | 60 |
| `DailyRate` | 802.49 | 403.51 | 102 | 465.0 | 802.0 | 1157.0 | 1499 |
| `DistanceFromHome` | 9.19 | 8.11 | 1 | 2.0 | 7.0 | 14.0 | 29 |
| `HourlyRate` | 65.89 | 20.33 | 30 | 48.0 | 66.0 | 83.75 | 100 |
| `MonthlyIncome` | 6502.93 | 4707.96 | 1009 | 2911.0 | 4919.0 | 8379.0 | 19999 |
| `MonthlyRate` | 14313.10 | 7117.79 | 2094 | 8047.0 | 14235.5 | 20461.5 | 26999 |
| `NumCompaniesWorked` | 2.69 | 2.50 | 0 | 1.0 | 2.0 | 4.0 | 9 |
| `PercentSalaryHike` | 15.21 | 3.66 | 11 | 12.0 | 14.0 | 18.0 | 25 |
| `TotalWorkingYears` | 11.28 | 7.78 | 0 | 6.0 | 10.0 | 15.0 | 40 |
| `TrainingTimesLastYear` | 2.80 | 1.29 | 0 | 2.0 | 3.0 | 3.0 | 6 |
| `YearsAtCompany` | 7.01 | 6.13 | 0 | 3.0 | 5.0 | 9.0 | 40 |
| `YearsInCurrentRole` | 4.23 | 3.62 | 0 | 2.0 | 3.0 | 7.0 | 18 |
| `YearsSinceLastPromotion`| 2.19 | 3.22 | 0 | 0.0 | 1.0 | 3.0 | 15 |
| `YearsWithCurrManager` | 4.12 | 3.57 | 0 | 2.0 | 3.0 | 7.0 | 17 |

---

## 6. Categorical Attributes and Cardinality Profile

The dataset contains **9 nominal categorical variables** (including target and constant):

1. **`BusinessTravel`** (3 categories): `Travel_Rarely` (1,043, 70.95%), `Travel_Frequently` (277, 18.84%), `Non-Travel` (150, 10.20%).
2. **`Department`** (3 categories): `Research & Development` (961, 65.37%), `Sales` (446, 30.34%), `Human Resources` (63, 4.29%).
3. **`EducationField`** (6 categories): `Life Sciences` (606, 41.22%), `Medical` (464, 31.56%), `Marketing` (159, 10.82%), `Technical Degree` (132, 8.98%), `Other` (82, 5.58%), `Human Resources` (27, 1.84%).
4. **`Gender`** (2 categories): `Male` (882, 60.00%), `Female` (588, 40.00%).
5. **`JobRole`** (9 categories): `Sales Executive` (326, 22.18%), `Research Scientist` (292, 19.86%), `Laboratory Technician` (259, 17.62%), `Manufacturing Director` (145, 9.86%), `Healthcare Representative` (131, 8.91%), `Manager` (102, 6.94%), `Sales Representative` (83, 5.65%), `Research Director` (80, 5.44%), `Human Resources` (52, 3.54%).
6. **`MaritalStatus`** (3 categories): `Married` (673, 45.78%), `Single` (470, 31.97%), `Divorced` (327, 22.24%).
7. **`Over18`** (1 category - Constant): `'Y'` (1,470, 100.0%).
8. **`OverTime`** (2 categories): `No` (1,054, 71.70%), `Yes` (416, 28.30%).
9. **`Attrition`** (2 categories - Target): `No` (1,233, 83.88%), `Yes` (237, 16.12%).

---

## 7. Key Findings and Analytical Insights

1. **Class Imbalance Realities:**
   - Voluntary turnover occurs in only **16.12%** of cases. Standard accuracy is an unsuitable metric; models will easily score 83.9% by predicting 'No' on every sample. Evaluation must focus on Recall for `Attrition = Yes`, Precision, and ROC-AUC.
2. **Data Cleanliness Quality:**
   - Zero missing entries and zero duplicate records reflect high data curation quality. No imputation routines will be needed for this primary dataset.
3. **Zero Variance Pruning Flags:**
   - `EmployeeCount` (all 1), `Over18` (all 'Y'), and `StandardHours` (all 80) contribute zero entropy or statistical variance ($\sigma^2 = 0$). They must be pruned in Task 2.
4. **Memorization Risk in Identifier:**
   - `EmployeeNumber` represents an arbitrary row key ranging from 1 to 2,068. Retaining it would allow tree algorithms to split on individual employee numbers and overfit.

---

## 8. Verification and Quality Checks Completed

- [x] Python analytical libraries imported and runtime versions logged.
- [x] Physical tab delimiter identified and verified.
- [x] Ingestion verified with dimensions exactly $(1470, 35)$.
- [x] First 5 records (`head`) and last 5 records (`tail`) verified.
- [x] Full column datatypes and memory footprint generated via `info()`.
- [x] Descriptive statistics generated via `describe()`.
- [x] Cardinality computed via `nunique()`.
- [x] Feature catalog established across 7 analytical families.
- [x] Module `src/data_loader.py` implemented and verified.
- [x] Master notebook `notebooks/AttritionIQ_Workflow.ipynb` updated with Task 1 cells.

---

## 9. Handoff to Task 2

With Task 1 complete, the raw verified DataFrame of shape $(1470, 35)$ is handed over to **Task 2: Data Cleaning and Quality Verification** to execute feature pruning, data integrity validation, and export the cleaned dataset.
