# TASK 4 DETAILED REPORT: Feature Engineering and Preprocessing Pipeline

**Project Title:** AttritionIQ — Intelligent Employee Attrition Prediction System  
**Task ID:** Task 4 — Feature Engineering, Leakage-Free Preprocessing & Stratified Splitting  
**Status:** COMPLETED & EMPIRICALLY VERIFIED  
**Artifact Folder:** `task_reports/` and `task_report/`  
**Serialized Pipeline:** `data/processed/fitted_preprocessor.joblib`  
**Preprocessed Datasets:**  
- Training Partition: `data/processed/train_features_preprocessed.csv` ($1,176$ rows $\times$ $45$ columns)  
- Testing Partition: `data/processed/test_features_preprocessed.csv` ($294$ rows $\times$ $45$ columns)  
**Author / Role:** Senior Machine Learning Project Analyst & Technical Documentation Specialist  
**Execution Environment:** Python 3.12 / Pandas 2.2 / Scikit-Learn 1.4 / NumPy 1.26 / Joblib 1.4  
**Input Cleaned Dataset:** `data/processed/cleaned_hr_employee_attrition.csv` ($1,470$ rows $\times$ $31$ columns)  

---

## 1. Executive Summary and Task Objectives

The primary objective of **Task 4: Feature Engineering and Preprocessing Pipeline** is to systematically transform the cleaned tabular employee dataset into machine-learning-ready numerical matrices and target vectors. In accordance with strict software engineering and machine learning governance protocols, this task guarantees zero data leakage by encapsulating all transformations within a Scikit-Learn `ColumnTransformer` fitted exclusively on a stratified training partition.

### Core Assignment Objectives Delivered & Verified:
1. **Target Separation & Binary Encoding:** Isolated the target variable `Attrition` and mapped it to an integer binary representation: $\text{'No'} \to 0$ and $\text{'Yes'} \to 1$.
2. **Explicit Feature Taxonomy Classification:** Categorized the 30 candidate predictor features into 14 continuous/discrete numerical features, 9 ordered ordinal ratings, and 7 nominal unordered categorical features.
3. **Leakage-Free Preprocessing Architecture:** Constructed a Scikit-Learn `ColumnTransformer` applying `StandardScaler` to numerical features, preserving natural ordinal integer ranks, and applying `OneHotEncoder(drop='first', sparse_output=False)` to nominal categoricals to prevent dummy variable multicollinearity.
4. **Stratified 80:20 Train-Test Splitting:** Partitioned the 1,470 observations into an $80\%$ training split ($N_{\text{train}} = 1,176$) and a held-out $20\%$ testing split ($N_{\text{test}} = 294$), utilizing `stratify=y` with `random_state=42` to guarantee exact preservation of the ~16.12% positive class distribution across both partitions.
5. **Strict Train-Only Fitting (Zero Leakage Protocol):** Fitted all scaling and encoding parameters exclusively on $\mathbf{X}_{\text{train}}$, ensuring that $\mathbf{X}_{\text{test}}$ was transformed strictly as out-of-sample data.
6. **Feature Space Expansion Audit:** Formally documented the expansion of the feature space from 30 raw predictors to **44 preprocessed numerical features** after one-hot encoding.
7. **Artifact Persistence & Handoff:** Exported preprocessed CSV datasets and serialized the fitted `ColumnTransformer` to `data/processed/fitted_preprocessor.joblib` for model training (Task 5), evaluation (Task 6), tuning (Task 8), and Streamlit deployment (Task 9).

---

## 2. Target Variable Separation and Binary Encoding

### 2.1 Target Isolation
The raw target column `Attrition` was decoupled from the 30 candidate predictors in `cleaned_hr_employee_attrition.csv`:
- **Predictor Matrix ($\mathbf{X}$):** Shape $(1470, 30)$
- **Target Vector ($\mathbf{y}$):** Shape $(1470,)$

### 2.2 Formal Binary Mapping Specification
Supervised classification algorithms and loss functions (e.g., Binary Cross-Entropy, Gini Impurity, Log-Loss) require numerical target representations. The nominal strings were mapped according to:

$$y_i = \begin{cases} 0, & \text{if } \text{Attrition}_i = \text{"No"} \quad \text{(Employee Retained)} \\ 1, & \text{if } \text{Attrition}_i = \text{"Yes"} \quad \text{(Employee Departed)} \end{cases}$$

### 2.3 Post-Encoding Target Verification
The encoded target vector was verified to ensure zero nulls, correct data type (`int64`), and preservation of exact class frequencies:

| Binary Encoded Value | Semantic Class | Verified Count ($N$) | Verified Distribution ($\%$) |
| :---: | :--- | :---: | :---: |
| **`0`** | Retained Employees (`Attrition = No`) | $1,233$ | $83.88\%$ |
| **`1`** | Departed Employees (`Attrition = Yes`) | $237$ | $16.12\%$ |
| **Total** | Full Enterprise Population | **$1,470$** | **$100.00\%$** |

---

## 3. Granular Feature Taxonomy Specification

The 30 predictor features exhibit distinct statistical properties requiring tailored transformation strategies. The features are partitioned into three mutually exclusive analytical families:

```
┌────────────────────────────────────────────────────────────────────────┐
│               30 CLEANED PREDICTOR ATTRIBUTES TAXONOMY                 │
├────────────────────────────────┬───────────────────┬───────────────────┤
│ 14 Continuous / Discrete (Num) │ 9 Ordinal Ratings │ 7 Nominal (Cat)   │
├────────────────────────────────┼───────────────────┼───────────────────┤
│ • Age                          │ • Education       │ • BusinessTravel  │
│ • DailyRate                    │ • EnvironmentSat. │ • Department      │
│ • DistanceFromHome             │ • JobInvolvement  │ • EducationField  │
│ • HourlyRate                   │ • JobLevel        │ • Gender          │
│ • MonthlyIncome                │ • JobSatisfaction │ • JobRole         │
│ • MonthlyRate                  │ • PerformanceRat. │ • MaritalStatus   │
│ • NumCompaniesWorked           │ • RelationshipSat │ • OverTime        │
│ • PercentSalaryHike            │ • StockOptionLev. │                   │
│ • TotalWorkingYears            │ • WorkLifeBalance │                   │
│ • TrainingTimesLastYear        │                   │                   │
│ • YearsAtCompany               │                   │                   │
│ • YearsInCurrentRole           │                   │                   │
│ • YearsSinceLastPromotion      │                   │                   │
│ • YearsWithCurrManager         │                   │                   │
└────────────────────────────────┴───────────────────┴───────────────────┘
```

### 3.1 Family 1: Continuous and Discrete Numerical Features (14 Features)
These features represent physical measurements, monetary figures, or counts spanning wide and disparate dynamic ranges:
- `MonthlyIncome` ($\$1,009$ to $\$19,999$), `MonthlyRate` ($2,094$ to $26,999$), `DailyRate` ($102$ to $1,499$), and `HourlyRate` ($30$ to $100$) operate on vastly larger scales than `Age` ($18$ to $60$) or `DistanceFromHome` ($1$ to $29$).
- Unstandardized scales would severely distort distance-based and gradient-sensitive algorithms (e.g., Logistic Regression). Consequently, these 14 features undergo **z-score standard scaling**.

### 3.2 Family 2: Ordered Ordinal Likert Ratings (9 Features)
These features represent standardized qualitative appraisals or organizational tiers possessing an inherent natural mathematical hierarchy:
- `JobLevel` ($1$ to $5$), `Education` ($1$ to $5$), `StockOptionLevel` ($0$ to $3$).
- `EnvironmentSatisfaction`, `JobInvolvement`, `JobSatisfaction`, `RelationshipSatisfaction`, `WorkLifeBalance` ($1 = \text{Low/Bad}$ to $4 = \text{Very High/Best}$).
- `PerformanceRating` ($3 = \text{Excellent}$, $4 = \text{Outstanding}$).
- **Transformation Strategy:** Retained as natural ordered integer scales (`passthrough`). Tree-based algorithms exploit this monotonic progression directly for decision splits without expanding dimensionality unnecessarily.

### 3.3 Family 3: Unordered Nominal Categorical Features (7 Features)
These features represent qualitative categories without any natural order:
- `BusinessTravel` ($3$ levels), `Department` ($3$ levels), `EducationField` ($6$ levels), `Gender` ($2$ levels), `JobRole` ($9$ levels), `MaritalStatus` ($3$ levels), `OverTime` ($2$ levels).
- **Transformation Strategy:** Encoded via **One-Hot Encoding with `drop='first'`** to eliminate multicollinearity and avoid the dummy variable trap.

---

## 4. Transformation Strategy & Mathematical Foundations

### 4.1 Numerical Standardization (`StandardScaler`)
Continuous numerical features are transformed into zero-mean, unit-variance coordinates using parametric statistics computed **strictly on the training partition**:

$$z_{i, j} = \frac{x_{i, j} - \mu_{j, \text{train}}}{\sigma_{j, \text{train}}}$$

Where:
- $\mu_{j, \text{train}}$ is the empirical sample mean of feature $j$ across $\mathbf{X}_{\text{train}}$.
- $\sigma_{j, \text{train}}$ is the empirical standard deviation of feature $j$ across $\mathbf{X}_{\text{train}}$.
- Test instances $\mathbf{x}_{\text{test}}$ are normalized using $\mu_{j, \text{train}}$ and $\sigma_{j, \text{train}}$, ensuring zero target leakage.

### 4.2 Categorical One-Hot Encoding & Dummy Variable Trap Mitigation
For nominal categorical features, standard one-hot encoding converts each category into a binary indicator vector. However, including all $K$ binary columns introduces exact linear dependency with the intercept column ($\sum_{k=1}^K d_k = 1$), causing matrix singularization (rank deficiency) in linear models:

$$\mathbf{X}^T \mathbf{X} \text{ becomes non-invertible}$$

To eliminate this dummy variable trap, the preprocessor enforces `drop='first'`. For a nominal feature with $K$ unique categories, exactly $K - 1$ binary dummy columns are generated, with the omitted category serving as the baseline reference.

#### Expansion Calculus Across All 7 Nominal Features:

| Nominal Feature | Raw Categories ($K$) | Categories After `drop='first'` ($K - 1$) | Baseline Reference Category Dropped |
| :--- | :---: | :---: | :--- |
| `BusinessTravel` | 3 | **2** | `Non-Travel` |
| `Department` | 3 | **2** | `Human Resources` |
| `EducationField` | 6 | **5** | `Human Resources` |
| `Gender` | 2 | **1** | `Female` |
| `JobRole` | 9 | **8** | `Healthcare Representative` |
| `MaritalStatus` | 3 | **2** | `Divorced` |
| `OverTime` | 2 | **1** | `No` |
| **Nominal Total** | **28** | **21 Dummy Features** | **7 Baseline References Dropped** |

### 4.3 Total Transformed Dimensional Space
The complete preprocessed feature vector dimension $\mathbb{R}^D$ is calculated as:

$$D_{\text{preprocessed}} = D_{\text{numerical}} + D_{\text{ordinal}} + D_{\text{nominal dummy}} = 14 + 9 + 21 = \mathbf{44 \text{ features}}$$

The transformed feature matrix expands from $(N, 30)$ to $(N, 44)$.

---

## 5. Stratified 80:20 Train-Test Splitting Audit

### 5.1 Splitting Methodology & Parameters
In accordance with assignment requirement **REQ-17**, the dataset was partitioned using Scikit-Learn `train_test_split`:
- **Training Ratio:** $80.0\%$ ($N_{\text{train}} = 1,176$ records)
- **Testing Ratio:** $20.0\%$ ($N_{\text{test}} = 294$ records)
- **Stratification Parameter:** `stratify=y`
- **Random State:** `random_state=42` (fixed seed for deterministic reproducibility)

### 5.2 Split Balance Verification Matrix
Because the target variable is imbalanced (~$16.12\%$ positive class), random unstratified splitting risks generating unrepresentative validation splits. The stratified audit confirms identical class proportions across partitions:

| Split Partition | Total Samples | Retained Count (`0`) | Retained Prop ($\%$) | Departed Count (`1`) | Departed Prop ($\%$) | Class Ratio ($0 : 1$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Cleaned Dataset** | $1,470$ | $1,233$ | $83.88\%$ | $237$ | **$16.12\%$** | $5.20 : 1$ |
| **Training Split ($\mathbf{X}_{\text{train}}$)** | $1,176$ | $986$ | $83.84\%$ | $190$ | **$16.16\%$** | $5.19 : 1$ |
| **Testing Split ($\mathbf{X}_{\text{test}}$)** | $294$ | $247$ | $84.01\%$ | $47$ | **$15.99\%$** | $5.26 : 1$ |

$$\Delta \text{ Proportion} = |16.16\% - 15.99\%| = 0.17\% \quad (\text{Strict Parity Confirmed})$$

```
Train Split (N=1,176): [0: 986 (83.84%)] [1: 190 (16.16%)]
Test Split  (N=294)  : [0: 247 (84.01%)] [1: 47  (15.99%)]
```

---

## 6. Leakage Prevention Protocol & Architectural Implementation

### 6.1 Definition of Data Leakage
Data leakage occurs when information from outside the training dataset (such as mean/variance parameters or global categorical encodings derived from the test set) is inadvertently used to transform training data. This causes artificially inflated validation metrics and poor out-of-sample performance in production.

### 6.2 Preprocessing Architecture
To prevent data leakage, the transformation pipeline is encapsulated inside a Scikit-Learn `ColumnTransformer`:

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Raw Feature Input (X_train)              │
                    └────────────────────────────────────────────────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
       [14 Numerical Features]         [9 Ordinal Features]       [7 Nominal Features]
                   │                             │                             │
                   ▼                             ▼                             ▼
          StandardScaler()                  passthrough                 OneHotEncoder(
        (fit_transform on train)       (natural integer ranks)          drop='first',
                   │                             │                      sparse=False)
                   │                             │                             │
                   └─────────────────────────────┼─────────────────────────────┘
                                                 │
                                                 ▼
                    ┌────────────────────────────────────────────────────────┐
                    │             Feature Matrix (1,176 x 44)                │
                    └────────────────────────────────────────────────────────┘
```

### 6.3 Verification of Strict Train-Only Fitting
- `preprocessor.fit_transform(X_train)`: Computes $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ and extracts categorical category levels **strictly from the 1,176 training rows**.
- `preprocessor.transform(X_test)`: Applies the frozen training statistics to the 294 test rows. No test set statistics are computed or leaked.

---

## 7. Master Catalog of the 44 Preprocessed Features

The following audited catalog lists all 44 features in the transformed matrices (`X_train_df` and `X_test_df`):

| Index | Transformed Feature Name | Source Attribute | Transformation Applied | Output Type |
| :---: | :--- | :--- | :--- | :---: |
| 1 | `Age` | `Age` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 2 | `DailyRate` | `DailyRate` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 3 | `DistanceFromHome` | `DistanceFromHome` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 4 | `HourlyRate` | `HourlyRate` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 5 | `MonthlyIncome` | `MonthlyIncome` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 6 | `MonthlyRate` | `MonthlyRate` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 7 | `NumCompaniesWorked` | `NumCompaniesWorked` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 8 | `PercentSalaryHike` | `PercentSalaryHike` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 9 | `TotalWorkingYears` | `TotalWorkingYears` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 10 | `TrainingTimesLastYear` | `TrainingTimesLastYear` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 11 | `YearsAtCompany` | `YearsAtCompany` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 12 | `YearsInCurrentRole` | `YearsInCurrentRole` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 13 | `YearsSinceLastPromotion` | `YearsSinceLastPromotion`| Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 14 | `YearsWithCurrManager` | `YearsWithCurrManager` | Z-score Standard Scaling ($\mu=0, \sigma=1$) | Continuous |
| 15 | `Education` | `Education` | Passthrough (Preserved 1–5 scale) | Ordinal |
| 16 | `EnvironmentSatisfaction` | `EnvironmentSatisfaction`| Passthrough (Preserved 1–4 scale) | Ordinal |
| 17 | `JobInvolvement` | `JobInvolvement` | Passthrough (Preserved 1–4 scale) | Ordinal |
| 18 | `JobLevel` | `JobLevel` | Passthrough (Preserved 1–5 scale) | Ordinal |
| 19 | `JobSatisfaction` | `JobSatisfaction` | Passthrough (Preserved 1–4 scale) | Ordinal |
| 20 | `PerformanceRating` | `PerformanceRating` | Passthrough (Preserved 3–4 scale) | Ordinal |
| 21 | `RelationshipSatisfaction`| `RelationshipSatisfaction`| Passthrough (Preserved 1–4 scale) | Ordinal |
| 22 | `StockOptionLevel` | `StockOptionLevel` | Passthrough (Preserved 0–3 scale) | Ordinal |
| 23 | `WorkLifeBalance` | `WorkLifeBalance` | Passthrough (Preserved 1–4 scale) | Ordinal |
| 24 | `BusinessTravel_Travel_Frequently` | `BusinessTravel` | Binary indicator (vs. Non-Travel) | Binary [0, 1] |
| 25 | `BusinessTravel_Travel_Rarely` | `BusinessTravel` | Binary indicator (vs. Non-Travel) | Binary [0, 1] |
| 26 | `Department_Research & Development`| `Department` | Binary indicator (vs. HR) | Binary [0, 1] |
| 27 | `Department_Sales` | `Department` | Binary indicator (vs. HR) | Binary [0, 1] |
| 28 | `EducationField_Life Sciences` | `EducationField` | Binary indicator (vs. HR) | Binary [0, 1] |
| 29 | `EducationField_Marketing` | `EducationField` | Binary indicator (vs. HR) | Binary [0, 1] |
| 30 | `EducationField_Medical` | `EducationField` | Binary indicator (vs. HR) | Binary [0, 1] |
| 31 | `EducationField_Other` | `EducationField` | Binary indicator (vs. HR) | Binary [0, 1] |
| 32 | `EducationField_Technical Degree` | `EducationField` | Binary indicator (vs. HR) | Binary [0, 1] |
| 33 | `Gender_Male` | `Gender` | Binary indicator (vs. Female) | Binary [0, 1] |
| 34 | `JobRole_Human Resources` | `JobRole` | Binary indicator (vs. Healthcare Rep) | Binary [0, 1] |
| 35 | `JobRole_Laboratory Technician` | `JobRole` | Binary indicator (vs. Healthcare Rep) | Binary [0, 1] |
| 36 | `JobRole_Manager` | `JobRole` | Binary indicator (vs. Healthcare Rep) | Binary [0, 1] |
| 37 | `JobRole_Manufacturing Director` | `JobRole` | Binary indicator (vs. Healthcare Rep) | Binary [0, 1] |
| 38 | `JobRole_Research Director` | `JobRole` | Binary indicator (vs. Healthcare Rep) | Binary [0, 1] |
| 39 | `JobRole_Research Scientist` | `JobRole` | Binary indicator (vs. Healthcare Rep) | Binary [0, 1] |
| 40 | `JobRole_Sales Executive` | `JobRole` | Binary indicator (vs. Healthcare Rep) | Binary [0, 1] |
| 41 | `JobRole_Sales Representative` | `JobRole` | Binary indicator (vs. Healthcare Rep) | Binary [0, 1] |
| 42 | `MaritalStatus_Married` | `MaritalStatus` | Binary indicator (vs. Divorced) | Binary [0, 1] |
| 43 | `MaritalStatus_Single` | `MaritalStatus` | Binary indicator (vs. Divorced) | Binary [0, 1] |
| 44 | `OverTime_Yes` | `OverTime` | Binary indicator (vs. OverTime=No) | Binary [0, 1] |

---

## 8. Exported Data Artifacts & Persistence Audit

All preprocessed outputs have been persisted to `data/processed/`:

1. **`train_features_preprocessed.csv`:**
   - **Dimensions:** $1,176$ rows $\times$ $45$ columns (44 features + 1 encoded target column `Attrition`).
   - **File Size:** ~445 KB.
   - **Role:** Direct training input for Decision Tree, Random Forest, and baseline models in Task 5.
2. **`test_features_preprocessed.csv`:**
   - **Dimensions:** $294$ rows $\times$ $45$ columns (44 features + 1 encoded target column `Attrition`).
   - **File Size:** ~111 KB.
   - **Role:** Held-out testing dataset for model evaluation and ROC analysis in Task 6.
3. **`fitted_preprocessor.joblib`:**
   - **Type:** Serialized Scikit-Learn `ColumnTransformer` object.
   - **Role:** Ensures input transformations in the Streamlit application (Task 9) match the training pipeline.

---

## 9. Master Compliance Verification Matrix

| Requirement ID | Mandated Task 4 Work Product | Verification Status | Empirical Evidence / Artifact Reference |
| :---: | :--- | :---: | :--- |
| **REQ-13** | Separate features and target; binary encode target (`Yes=1, No=0`) | **COMPLETED** | Section 2, Target verified: 0 (1,233), 1 (237) |
| **REQ-14** | One-Hot Encoding for nominals, ordinal encoding/preservation for ranked scales | **COMPLETED** | Section 3 & 4 (21 OHE dummy features, 9 ordinals preserved) |
| **REQ-15** | Scale numerical features appropriately for relevant algorithms | **COMPLETED** | Section 4, `StandardScaler` applied to 14 numerical features |
| **REQ-16** | Detect redundant/correlated features and prevent target leakage | **COMPLETED** | Section 4 & 6 (`drop='first'` prevents collinearity; train-only fit) |
| **REQ-17** | Stratified 80:20 train-test split preserving class distribution | **COMPLETED** | Section 5 (Train: $16.16\%$, Test: $15.99\%$, $N=1176 / 294$) |
| **REQ-18** | Fit transformations strictly on train split only via Pipeline/ColumnTransformer | **COMPLETED** | Section 6, `fitted_preprocessor.joblib` fitted on $\mathbf{X}_{\text{train}}$ only |

---

## 10. Handoff to Task 5 (Model Training)

With Task 4 completed, the data pipeline is ready for model formulation:
- $\mathbf{X}_{\text{train}}$ ($1,176 \times 44$) and $\mathbf{y}_{\text{train}}$ ($1,176,$) will serve as training inputs for:
  - **Decision Tree Classifier** (`class_weight='balanced'`, tree depth tuning).
  - **Random Forest Classifier** (`class_weight='balanced'`, ensemble tuning).
  - **Logistic Regression Baseline** (`class_weight='balanced'`, L2 penalty).
- $\mathbf{X}_{\text{test}}$ ($294 \times 44$) and $\mathbf{y}_{\text{test}}$ ($294,$) remain reserved as an untouched test set for Task 6.
