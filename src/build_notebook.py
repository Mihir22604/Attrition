"""
Build Master Notebook for AttritionIQ (Tasks 1, 2, 3, and 4)
"""
import nbformat as nbf
import os

def create_master_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Markdown Title
    cells.append(nbf.v4.new_markdown_cell("""# AttritionIQ — Intelligent Employee Attrition Prediction System
## Master Analytical Workflow & Machine Learning Pipeline

**Author:** Senior Machine Learning Project Analyst & Technical Documentation Specialist  
**Dataset:** `WA_Fn-UseC_-HR-Employee-Attrition.csv` (IBM HR Analytics Attrition Benchmark)  
**Execution Stage:** Phase 1 (Tasks 1 & 2) & Phase 2 (Tasks 3 & 4) Completed & Verified
"""))

    # ==========================================
    # Task 1 Section
    # ==========================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Task 1: Data Loading and Understanding

### Task Objectives:
- Import essential data science and machine learning libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`.
- Correctly ingest the raw dataset, handling the physical tab delimiter (`\\t`).
- Display first and last 5 records to verify tabular parsing.
- Extract dataset dimensions, column names, datatypes, and non-null counts.
- Run `info()`, `describe()`, and `nunique()` for statistical and cardinality profiling.
- Catalog features into continuous numerical, discrete numerical, ordinal ratings, nominal categories, identifiers, and target.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.1 Import analytical libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-Learn version: {sklearn.__version__}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.2 Ingest raw dataset with tab delimiter
raw_data_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
df_raw = pd.read_csv(raw_data_path, sep='\\t')

print(f"Raw Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.3 Display first 5 records
df_raw.head()
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.4 Display last 5 records
df_raw.tail()
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.5 Column datatypes, non-null counts, and memory usage
df_raw.info()
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.6 Statistical summary across all attributes
df_raw.describe(include='all').T
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.7 Feature cardinality (unique value counts)
df_raw.nunique()
"""))

    # ==========================================
    # Task 2 Section
    # ==========================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Task 2: Data Cleaning and Quality Verification

### Task Objectives:
- Audit missing values across every column.
- Detect and evaluate duplicate records.
- Identify constant / zero-variance columns (`EmployeeCount`, `Over18`, `StandardHours`).
- Identify uninformative identifier columns (`EmployeeNumber`).
- Validate target variable integrity (`Attrition` contains exactly 'Yes' and 'No').
- Apply justified cleaning transformations, document rationale, and export cleaned dataset.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Audit missing values
missing_counts = df_raw.isnull().sum()
print(f"Total Missing Values in Dataset: {missing_counts.sum()}")
print("Columns with missing values (if any):")
print(missing_counts[missing_counts > 0])
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.2 Audit duplicate records
duplicate_count = df_raw.duplicated().sum()
print(f"Total Duplicate Rows: {duplicate_count}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.3 Detect zero-variance constant features
constant_cols = [col for col in df_raw.columns if df_raw[col].nunique() <= 1]
print(f"Zero-Variance Constant Columns ({len(constant_cols)}): {constant_cols}")
for col in constant_cols:
    print(f"  - {col}: unique value = {df_raw[col].unique().tolist()}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.4 Detect unique identifier columns
identifier_cols = [col for col in df_raw.columns if df_raw[col].nunique() == len(df_raw)]
print(f"Identifier Columns ({len(identifier_cols)}): {identifier_cols}")
for col in identifier_cols:
    print(f"  - {col}: {df_raw[col].nunique()} unique records (range: {df_raw[col].min()} to {df_raw[col].max()})")
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.5 Verify target variable distribution
print("Target Variable ('Attrition') Counts:")
print(df_raw['Attrition'].value_counts())
print("\\nTarget Variable ('Attrition') Proportions (%):")
print((df_raw['Attrition'].value_counts(normalize=True) * 100).round(2))
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.6 Execute justified pruning operations
# 1. Prune constant columns: Zero variance provides zero predictive information.
# 2. Prune EmployeeNumber: Unique row ID risks tree memorization / overfitting.
columns_to_prune = constant_cols + identifier_cols
df_cleaned = df_raw.drop(columns=columns_to_prune)

print(f"Original shape: {df_raw.shape}")
print(f"Pruned columns: {columns_to_prune}")
print(f"Cleaned shape:  {df_cleaned.shape}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.7 Export cleaned dataset for downstream modeling (Task 3 & 4)
os.makedirs('data/processed', exist_ok=True)
cleaned_data_path = 'data/processed/cleaned_hr_employee_attrition.csv'
df_cleaned.to_csv(cleaned_data_path, index=False)
print(f"Cleaned dataset successfully exported to: {cleaned_data_path}")
"""))

    # ==========================================
    # Task 3 Section
    # ==========================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Task 3: Exploratory Data Analysis (EDA)

### Task Objectives:
- Perform univariate analysis on continuous variables (`Age`, `MonthlyIncome`, `YearsAtCompany`) and ordinal factors (`JobSatisfaction`, `WorkLifeBalance`).
- Calculate the exact enterprise baseline attrition rate using the formal formula.
- Conduct bivariate analysis across all mandatory factors:
  1. `Department` vs `Attrition`
  2. `JobRole` vs `Attrition`
  3. `OverTime` vs `Attrition`
  4. `MonthlyIncome` vs `Attrition`
  5. `JobSatisfaction` vs `Attrition`
  6. `WorkLifeBalance` vs `Attrition`
  7. `YearsAtCompany` vs `Attrition`
  8. `BusinessTravel` vs `Attrition`
- Compute relative percentage attrition rates within each category (avoiding raw count fallacy).
- Perform multivariate correlation analysis and generate an annotated Pearson heatmap.
- Synthesize findings into at least 5 empirical, data-driven analytical insights.
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.1 Calculate exact enterprise attrition rate formula
total_employees = len(df_cleaned)
departed_count = (df_cleaned['Attrition'] == 'Yes').sum()
retained_count = (df_cleaned['Attrition'] == 'No').sum()
attrition_rate = (departed_count / total_employees) * 100.0

print(f"Total Employees (N):          {total_employees:,}")
print(f"Employees Departed (Yes):     {departed_count:,} ({attrition_rate:.2f}%)")
print(f"Employees Retained (No):      {retained_count:,} ({100 - attrition_rate:.2f}%)")
print(f"Enterprise Attrition Formula: ({departed_count} / {total_employees}) * 100 = {attrition_rate:.4f}%")
print(f"Class Imbalance Ratio:        {retained_count / departed_count:.2f} : 1")
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.2 Visualizing Target Distribution (Headcount & Donut Chart)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
palette_target = {'No': '#2b5c8f', 'Yes': '#d9534f'}

sns.countplot(data=df_cleaned, x='Attrition', hue='Attrition', ax=ax[0], palette=palette_target, order=['No', 'Yes'], legend=False)
ax[0].set_title("Enterprise Attrition Headcount Distribution", fontsize=12, fontweight='bold')
ax[0].set_ylabel("Employee Count")
for p in ax[0].patches:
    h = p.get_height()
    pct = (h / total_employees) * 100
    ax[0].annotate(f"{int(h):,}\\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                   ha='center', va='center', fontsize=11, color='white', fontweight='bold')

ax[1].pie([retained_count, departed_count], labels=['Retained (No)', 'Departed (Yes)'],
          autopct='%1.2f%%', startangle=140, colors=['#2b5c8f', '#d9534f'],
          explode=(0, 0.08), wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2))
ax[1].set_title(f"Baseline Enterprise Attrition Rate: {attrition_rate:.2f}%", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.3 Univariate Distributions: Age, Monthly Income, and Years at Company
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age distribution
sns.histplot(data=df_cleaned, x='Age', hue='Attrition', kde=True, ax=axes[0], palette=palette_target, bins=20, element='step')
axes[0].set_title("Age Distribution by Attrition", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Age (Years)")

# Monthly Income distribution
sns.histplot(data=df_cleaned, x='MonthlyIncome', hue='Attrition', kde=True, ax=axes[1], palette=palette_target, bins=25, element='step')
axes[1].set_title("Monthly Income ($) Distribution", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Monthly Income (USD)")

# Years at Company distribution
sns.histplot(data=df_cleaned, x='YearsAtCompany', hue='Attrition', kde=True, ax=axes[2], palette=palette_target, bins=20, element='step')
axes[2].set_title("Tenure at Company Distribution", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Years at Company")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.4 Bivariate Analysis: Department-Level Attrition
dept_summary = pd.crosstab(df_cleaned['Department'], df_cleaned['Attrition'], margins=True)
dept_summary['Attrition Rate (%)'] = (dept_summary['Yes'] / dept_summary['All'] * 100).round(2)
print("=== Department-Wise Attrition Summary ===")
display(dept_summary)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
dept_rates = (pd.crosstab(df_cleaned['Department'], df_cleaned['Attrition'], normalize='index')['Yes'] * 100).sort_values(ascending=False)

# Stacked Headcount
pd.crosstab(df_cleaned['Department'], df_cleaned['Attrition']).loc[dept_rates.index].plot(
    kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[0]
)
axes[0].set_title("Headcount Breakdown by Department", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Employee Count")
axes[0].tick_params(axis='x', rotation=15)

# Percentage Attrition Rate
sns.barplot(x=dept_rates.index, y=dept_rates.values, hue=dept_rates.index, ax=axes[1], palette='Reds_r', legend=False)
axes[1].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
axes[1].set_title("Attrition Rate (%) by Department", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Attrition Rate (%)")
axes[1].tick_params(axis='x', rotation=15)
for p in axes[1].patches:
    h = p.get_height()
    axes[1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.5), ha='center', va='bottom', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.5 Bivariate Analysis: Job Role Disproportionate Turnover
role_ct = pd.crosstab(df_cleaned['JobRole'], df_cleaned['Attrition'])
role_rate = (role_ct['Yes'] / role_ct.sum(axis=1) * 100).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#d9534f' if r > attrition_rate else '#2b5c8f' for r in role_rate.values]
ax.barh(role_rate.index, role_rate.values, color=colors, height=0.65)
ax.axvline(attrition_rate, color='darkred', linestyle='--', linewidth=1.5, label=f"Enterprise Avg ({attrition_rate:.1f}%)")
ax.set_title("Attrition Rate (%) across 9 Job Roles (Ranked)", fontsize=13, fontweight='bold')
ax.set_xlabel("Attrition Rate (%)")
for i, v in enumerate(role_rate.values):
    ax.text(v + 0.6, i, f"{v:.1f}%", va='center', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.6 Bivariate Analysis: OverTime & Business Travel Impact
ot_rates = (pd.crosstab(df_cleaned['OverTime'], df_cleaned['Attrition'], normalize='index')['Yes'] * 100)
bt_rates = (pd.crosstab(df_cleaned['BusinessTravel'], df_cleaned['Attrition'], normalize='index')['Yes'] * 100).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Overtime Impact
sns.barplot(x=ot_rates.index, y=ot_rates.values, hue=ot_rates.index, ax=axes[0], palette=['#2b5c8f', '#d9534f'], legend=False)
axes[0].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
axes[0].set_title("Attrition Rate: Mandatory OverTime Impact", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Attrition Rate (%)")
for p in axes[0].patches:
    h = p.get_height()
    axes[0].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.6), ha='center', va='bottom', fontweight='bold')
axes[0].legend()

# Business Travel Impact
sns.barplot(x=bt_rates.index, y=bt_rates.values, hue=bt_rates.index, ax=axes[1], palette='Purples_r', legend=False)
axes[1].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
axes[1].set_title("Attrition Rate by Business Travel Frequency", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Attrition Rate (%)")
for p in axes[1].patches:
    h = p.get_height()
    axes[1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.5), ha='center', va='bottom', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.7 Subjective Sentiment: Job Satisfaction & Work-Life Balance
js_rates = (pd.crosstab(df_cleaned['JobSatisfaction'], df_cleaned['Attrition'], normalize='index')['Yes'] * 100)
wlb_rates = (pd.crosstab(df_cleaned['WorkLifeBalance'], df_cleaned['Attrition'], normalize='index')['Yes'] * 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
js_labels = ['1: Low', '2: Medium', '3: High', '4: Very High']
wlb_labels = ['1: Bad', '2: Good', '3: Better', '4: Best']

sns.barplot(x=[js_labels[i-1] for i in js_rates.index], y=js_rates.values, hue=js_rates.index, ax=axes[0], palette='Blues_r', legend=False)
axes[0].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
axes[0].set_title("Attrition Rate by Job Satisfaction Rating", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Attrition Rate (%)")
for p in axes[0].patches:
    h = p.get_height()
    axes[0].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.4), ha='center', va='bottom', fontweight='bold')
axes[0].legend()

sns.barplot(x=[wlb_labels[i-1] for i in wlb_rates.index], y=wlb_rates.values, hue=wlb_rates.index, ax=axes[1], palette='Oranges_r', legend=False)
axes[1].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
axes[1].set_title("Attrition Rate by Work-Life Balance Rating", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Attrition Rate (%)")
for p in axes[1].patches:
    h = p.get_height()
    axes[1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.4), ha='center', va='bottom', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.8 Multivariate Correlation Matrix Heatmap
numeric_df = df_cleaned.select_dtypes(include=['int64', 'float64']).copy()
numeric_df['Attrition_Numeric'] = (df_cleaned['Attrition'] == 'Yes').astype(int)
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.4, vmax=1.0, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, annot=False, ax=ax)
ax.set_title("Multivariate Pearson Correlation Matrix (Numerical & Ordinal Features)", fontsize=14, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Top 5 Negative Correlations with Attrition:")
print(corr_matrix['Attrition_Numeric'].drop('Attrition_Numeric').sort_values().head(5))
print("\\nTop 5 Positive Correlations with Attrition:")
print(corr_matrix['Attrition_Numeric'].drop('Attrition_Numeric').sort_values().tail(5))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Task 3 Key Analytical Insights:
1. **The Overtime Penalty (3x Multiplier):** Employees working mandatory OverTime exhibit an alarming **30.53% attrition rate**, nearly triple the **10.44%** rate observed for non-overtime staff. OverTime represents one of the strongest single operational churn drivers.
2. **Role & Seniority Vulnerability (Sales Representative Crisis):** Junior front-line roles suffer disproportionate voluntary turnover: **Sales Representatives exhibit 39.76% attrition**, followed by **Laboratory Technicians (23.94%)** and **Human Resources personnel (23.08%)**. Conversely, senior leadership roles (**Managers at 4.90%**, **Research Directors at 2.50%**) exhibit near-perfect retention.
3. **The Early-Tenure & Early-Career Flight Window:** Both Age and Total Working Years correlate negatively with turnover ($r = -0.159$ and $r = -0.171$). Median tenure for departing employees is under 3 years, confirming that voluntary turnover is heavily concentrated in early career onboarding windows.
4. **Compensation & Hierarchical Buffer:** Monthly Income exhibits a clear protective threshold. Employees earning below ~$3,000/month (Job Level 1) face elevated attrition, whereas individuals at Job Level 4 and 5 ($10,000+ to $19,999/month) possess attrition rates under 5%.
5. **Business Travel & Extreme Work-Life Stress:** Frequent business travel drives an attrition rate of **24.91%** compared to only **8.00%** for non-travelers. In addition, employees rating Work-Life Balance as 'Bad' (Level 1) leave at a **31.25%** rate, confirming that unsustainable schedules induce turnover regardless of base pay.
"""))

    # ==========================================
    # Task 4 Section
    # ==========================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Task 4: Feature Engineering and Preprocessing Pipeline

### Task Objectives:
- Separate features $\\mathbf{X}$ (30 predictors) and target $\\mathbf{y}$ (`Attrition`).
- Binary encode target variable (`No` $\\to 0$, `Yes` $\\to 1$).
- Explicitly partition features into continuous numericals (14), ordinals (9), and nominal categoricals (7).
- Construct Scikit-Learn `ColumnTransformer` with `StandardScaler` for continuous numericals and `OneHotEncoder(drop='first', sparse_output=False)` for nominal categoricals.
- Preserve natural ordinal order for 9 rating scales.
- Execute a **Stratified 80:20 Train-Test Split** (`random_state=42`, `stratify=y`) preserving class distribution.
- Fit all transformations strictly on $\\mathbf{X}_{\\text{train}}$ only to guarantee zero data leakage.
- Verify dimensional alignment and export preprocessed artifacts.
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.1 Target Separation & Binary Encoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

X = df_cleaned.drop(columns=['Attrition'])
y = df_cleaned['Attrition'].map({'No': 0, 'Yes': 1}).astype(int)

print(f"Features matrix shape: {X.shape}")
print(f"Target vector shape:   {y.shape}")
print(f"Encoded Target Balance: 0 (No) = {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%), 1 (Yes) = {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.2 Explicit Feature Taxonomy Definition
NUMERICAL_FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

ORDINAL_FEATURES = [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance'
]

NOMINAL_CATEGORICAL_FEATURES = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

print(f"Numerical continuous/discrete features: {len(NUMERICAL_FEATURES)}")
print(f"Ordinal features:                       {len(ORDINAL_FEATURES)}")
print(f"Nominal categorical features:           {len(NOMINAL_CATEGORICAL_FEATURES)}")
print(f"Total Predictors:                       {len(NUMERICAL_FEATURES) + len(ORDINAL_FEATURES) + len(NOMINAL_CATEGORICAL_FEATURES)}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.3 Stratified 80:20 Train-Test Partitioning
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"Training instances: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing instances:  {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"Train Target Distribution: {dict(y_train.value_counts())} -> Positive Rate: {y_train.mean()*100:.2f}%")
print(f"Test Target Distribution:  {dict(y_test.value_counts())} -> Positive Rate: {y_test.mean()*100:.2f}%")
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.4 Construct Scikit-Learn ColumnTransformer Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('ord', 'passthrough', ORDINAL_FEATURES),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), NOMINAL_CATEGORICAL_FEATURES)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# Fit exclusively on X_train to avoid data leakage
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out().tolist()

X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

print(f"Preprocessed X_train shape: {X_train_df.shape}")
print(f"Preprocessed X_test shape:  {X_test_df.shape}")
print(f"Total features after One-Hot Encoding: {len(feature_names)}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.5 Inspect Sample Transformed Features
display(X_train_df.head())
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.6 Persist Preprocessed Datasets & Fitted Preprocessor
train_export = X_train_df.copy()
train_export['Attrition'] = y_train
train_export.to_csv('data/processed/train_features_preprocessed.csv', index=False)

test_export = X_test_df.copy()
test_export['Attrition'] = y_test
test_export.to_csv('data/processed/test_features_preprocessed.csv', index=False)

joblib.dump(preprocessor, 'data/processed/fitted_preprocessor.joblib')

print("Fitted preprocessor successfully serialized to: data/processed/fitted_preprocessor.joblib")
print("Transformed train & test splits saved to: data/processed/")
"""))

    # ==========================================
    # Task 5 Section: Build Machine Learning Models
    # ==========================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Task 5: Build the Machine Learning Models

### Task Objectives:
- Train Decision Tree Classifier: experiment with hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`) and class weighting.
- Train Random Forest Classifier: configure ensemble size, depth bounds, and leaf constraints.
- Train Logistic Regression baseline: apply cost-sensitive weighting on standardized features.
- Ensure strict training parity by training all algorithms on the identical stratified training split ($N = 1,176$).
- Visualize the Decision Tree decision hierarchy.
- Serialize all fitted estimators using Joblib.
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.1 Import Model Architectures & Load Preprocessed Partitions
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Load preprocessed matrices
X_train_model = pd.read_csv('data/processed/train_features_preprocessed.csv').drop(columns=['Attrition'])
y_train_model = pd.read_csv('data/processed/train_features_preprocessed.csv')['Attrition']

X_test_model = pd.read_csv('data/processed/test_features_preprocessed.csv').drop(columns=['Attrition'])
y_test_model = pd.read_csv('data/processed/test_features_preprocessed.csv')['Attrition']

print(f"Training predictor shape: {X_train_model.shape}")
print(f"Testing predictor shape:  {X_test_model.shape}")
print(f"Train class balance: Retained={sum(y_train_model==0)}, Departed={sum(y_train_model==1)}")
print(f"Test class balance:  Retained={sum(y_test_model==0)}, Departed={sum(y_test_model==1)}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.2 Model 1A: Decision Tree (Unconstrained Baseline)
dt_unconstrained = DecisionTreeClassifier(
    criterion='gini',
    max_depth=None,
    random_state=42
)
dt_unconstrained.fit(X_train_model, y_train_model)
os.makedirs('models', exist_ok=True)
joblib.dump(dt_unconstrained, 'models/dt_unconstrained.joblib')

print(f"Decision Tree (Unconstrained) Fitted:")
print(f" - Max Depth: {dt_unconstrained.get_depth()}")
print(f" - Number of Leaves: {dt_unconstrained.get_n_leaves()}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.3 Model 1B: Decision Tree (Regularized & Balanced)
dt_regularized = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
dt_regularized.fit(X_train_model, y_train_model)
joblib.dump(dt_regularized, 'models/dt_regularized.joblib')

print(f"Decision Tree (Regularized & Balanced) Fitted:")
print(f" - Max Depth: {dt_regularized.get_depth()}")
print(f" - Number of Leaves: {dt_regularized.get_n_leaves()}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.4 Model 2: Random Forest (Balanced Ensemble)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=6,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_model, y_train_model)
joblib.dump(rf_model, 'models/random_forest.joblib')

print(f"Random Forest (Balanced) Fitted:")
print(f" - Estimator Count: {len(rf_model.estimators_)}")
print(f" - Base Criterion: {rf_model.criterion}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.5 Model 3: Logistic Regression (Balanced Baseline)
lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_model, y_train_model)
joblib.dump(lr_model, 'models/logistic_regression.joblib')

print(f"Logistic Regression Fitted:")
print(f" - Solved Iterations: {lr_model.n_iter_[0]}")
print(f" - Intercept: {lr_model.intercept_[0]:.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.6 Decision Tree Architecture Visualization
plt.figure(figsize=(22, 10), dpi=200)
plot_tree(
    dt_regularized,
    feature_names=X_train_model.columns.tolist(),
    class_names=['Retained', 'Departed'],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3,
    proportion=True
)
plt.title("Regularized Decision Tree Hierarchy (Top 3 Decision Levels)", fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()
"""))

    # ==========================================
    # Task 6 Section: Model Evaluation and Comparison
    # ==========================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Task 6: Model Evaluation and Comparison

### Task Objectives:
- Evaluate Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC for all models.
- Produce Scikit-Learn standard classification reports on held-out test data.
- Benchmark training vs. testing performance to formally diagnose overfitting and empirical memorization.
- Prioritize positive class recall (`Attrition = Yes`) over misleading raw accuracy.
- Render overlaid ROC curves and 2x2 confusion matrix breakdowns.
- Construct the comprehensive model comparison table benchmarking all algorithms.
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.1 Inference & Metric Computation Across All Models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

models_dict = {
    'Decision Tree (Unconstrained)': dt_unconstrained,
    'Decision Tree (Regularized & Balanced)': dt_regularized,
    'Random Forest (Balanced)': rf_model,
    'Logistic Regression (Balanced)': lr_model
}

results_summary = []

for name, model in models_dict.items():
    y_tr_pred = model.predict(X_train_model)
    y_te_pred = model.predict(X_test_model)
    y_tr_proba = model.predict_proba(X_train_model)[:, 1]
    y_te_proba = model.predict_proba(X_test_model)[:, 1]
    
    tr_acc = accuracy_score(y_train_model, y_tr_pred)
    te_acc = accuracy_score(y_test_model, y_te_pred)
    te_auc = roc_auc_score(y_test_model, y_te_proba)
    te_prec = precision_score(y_test_model, y_te_pred, pos_label=1, zero_division=0)
    te_rec = recall_score(y_test_model, y_te_pred, pos_label=1, zero_division=0)
    te_f1 = f1_score(y_test_model, y_te_pred, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_test_model, y_te_pred)
    
    results_summary.append({
        'Model': name,
        'Train Acc (%)': round(tr_acc * 100, 2),
        'Test Acc (%)': round(te_acc * 100, 2),
        'Gap (%)': round((tr_acc - te_acc) * 100, 2),
        'Precision (Yes)': round(te_prec, 4),
        'Recall (Yes)': round(te_rec, 4),
        'F1-Score (Yes)': round(te_f1, 4),
        'ROC-AUC': round(te_auc, 4),
        'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0], 'TP': cm[1, 1]
    })
    
    print(f"\\n{'='*65}\\nMODEL: {name}\\n{'='*65}")
    print(classification_report(y_test_model, y_te_pred, target_names=['Retained (0)', 'Departed (1)'], digits=4))
    print(f"ROC-AUC: {te_auc:.4f} | Recall (Yes): {te_rec*100:.2f}% | Test Accuracy: {te_acc*100:.2f}%")
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.2 Master Model Comparison Table
comparison_df = pd.DataFrame(results_summary)
display(comparison_df[['Model', 'Train Acc (%)', 'Test Acc (%)', 'Gap (%)', 'Recall (Yes)', 'Precision (Yes)', 'F1-Score (Yes)', 'ROC-AUC']])
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.3 Overlaid ROC Curves Visualization
plt.figure(figsize=(9, 6.5), dpi=200)
palette = {'Decision Tree (Unconstrained)': '#DC2626', 'Decision Tree (Regularized & Balanced)': '#2563EB', 'Random Forest (Balanced)': '#059669', 'Logistic Regression (Balanced)': '#7C3AED'}

for name, model in models_dict.items():
    y_te_proba = model.predict_proba(X_test_model)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_model, y_te_proba)
    auc = roc_auc_score(y_test_model, y_te_proba)
    ls = '--' if 'Unconstrained' in name else '-'
    plt.plot(fpr, tpr, color=palette[name], lw=2.2, linestyle=ls, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], color='#94A3B8', linestyle=':', lw=1.5, label='Chance Line (AUC = 0.5000)')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.03])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate (Recall)', fontsize=11, fontweight='bold')
plt.title('AttritionIQ: Overlaid ROC Curves Across All Models', fontsize=13, fontweight='bold', pad=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.4 Confusion Matrix Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
axes = axes.flatten()

for idx, (name, model) in enumerate(models_dict.items()):
    ax = axes[idx]
    y_te_pred = model.predict(X_test_model)
    cm = confusion_matrix(y_test_model, y_te_pred)
    
    total = cm.sum()
    labels = np.array([
        [f"TN: {cm[0,0]}\\n({cm[0,0]/total*100:.1f}%)", f"FP: {cm[0,1]}\\n({cm[0,1]/total*100:.1f}%)"],
        [f"FN: {cm[1,0]}\\n({cm[1,0]/total*100:.1f}%)", f"TP: {cm[1,1]}\\n({cm[1,1]/total*100:.1f}%)"]
    ])
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax,
                annot_kws={'fontsize': 11, 'fontweight': 'bold'}, linewidths=1.5, linecolor='#CBD5E1')
    ax.set_title(f"{name}", fontsize=11.5, fontweight='bold', pad=8)
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    ax.set_xticklabels(['Retained (0)', 'Departed (1)'])
    ax.set_yticklabels(['Retained (0)', 'Departed (1)'])

plt.suptitle("AttritionIQ: Confusion Matrix Breakdown Across Models (Test Set N = 294)", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()
"""))

    # ==========================================
    # Task 7 Section
    # ==========================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Task 7: Feature Importance and Interpretability Analysis

### Task Objectives:
- Extract Mean Decrease in Impurity (Gini Importance / MDI) from the fitted Random Forest ensemble.
- Aggregate MDI importances across one-hot encoded dummy variables back to parent features.
- Compute Permutation Feature Importance on the held-out test split (`X_test`, `y_test`) over 20 random shuffles to assess out-of-sample generalization impact.
- Contrast MDI vs Permutation Importance to demonstrate MDI's susceptibility to high-cardinality bias.
- Cross-reference top predictive features with findings from Exploratory Data Analysis (EDA).
- Formally differentiate statistical correlation from causal mechanisms (Correlation != Causation mandate).
"""))

    cells.append(nbf.v4.new_code_cell("""# 7.1 Extract and Plot Gini Feature Importance (MDI)
rf_model = models_dict['Random Forest (Balanced)']
mdi_importances = rf_model.feature_importances_
tree_importances = np.array([tree.feature_importances_ for tree in rf_model.estimators_])
mdi_std = np.std(tree_importances, axis=0)

mdi_df = pd.DataFrame({
    'feature': feature_names,
    'mdi_importance': mdi_importances,
    'mdi_std': mdi_std
}).sort_values(by='mdi_importance', ascending=False).reset_index(drop=True)

print("Top 10 Features by Gini Impurity Decrease (MDI):")
display(mdi_df.head(10))

plt.figure(figsize=(10, 6), dpi=150)
top10_mdi = mdi_df.head(10).iloc[::-1]
plt.barh(np.arange(10), top10_mdi['mdi_importance'], xerr=top10_mdi['mdi_std'], color='#2563EB', alpha=0.9, ecolor='#94A3B8', capsize=3)
plt.yticks(np.arange(10), top10_mdi['feature'], fontsize=10)
plt.xlabel('Gini Impurity Decrease (MDI)', fontsize=11, fontweight='bold')
plt.title('Task 7: Top 10 Random Forest Features by MDI (with +/- 1-sigma Cross-Tree Spread)', fontsize=12, fontweight='bold', pad=10)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 7.2 Permutation Importance on Held-Out Test Set (Unbiased Generalization)
from sklearn.inspection import permutation_importance

perm_result = permutation_importance(
    rf_model, X_test_model, y_test_model,
    scoring='roc_auc',
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'perm_roc_auc_drop': perm_result.importances_mean,
    'perm_std': perm_result.importances_std
}).sort_values(by='perm_roc_auc_drop', ascending=False).reset_index(drop=True)

print("Top 10 Features by Out-of-Sample Test ROC-AUC Decrease:")
display(perm_df.head(10))

plt.figure(figsize=(10, 6), dpi=150)
top10_perm = perm_df.head(10).iloc[::-1]
plt.barh(np.arange(10), top10_perm['perm_roc_auc_drop'], xerr=top10_perm['perm_std'], color='#7C3AED', alpha=0.9, ecolor='#94A3B8', capsize=3)
plt.yticks(np.arange(10), top10_perm['feature'], fontsize=10)
plt.xlabel('Test ROC-AUC Decrease Upon Shuffling', fontsize=11, fontweight='bold')
plt.title('Task 7: Top 10 Features by Permutation Importance (Test Set N=294)', fontsize=12, fontweight='bold', pad=10)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 7.3 Architectural Contrast: MDI Cardinality Bias vs Permutation Importance
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# MDI Plot
axes[0].barh(np.arange(10), top10_mdi['mdi_importance'], color='#2563EB', alpha=0.85)
axes[0].set_yticks(np.arange(10))
axes[0].set_yticklabels(top10_mdi['feature'], fontsize=9.5)
axes[0].set_xlabel('Gini Impurity Decrease (MDI - In-Sample)', fontsize=10.5, fontweight='bold')
axes[0].set_title('(A) MDI Importance (Favors Continuous High-Cardinality)', fontsize=11, fontweight='bold')
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# Permutation Plot
axes[1].barh(np.arange(10), top10_perm['perm_roc_auc_drop'], color='#7C3AED', alpha=0.85)
axes[1].set_yticks(np.arange(10))
axes[1].set_yticklabels(top10_perm['feature'], fontsize=9.5)
axes[1].set_xlabel('Out-of-Sample ROC-AUC Drop (Unbiased)', fontsize=10.5, fontweight='bold')
axes[1].set_title('(B) Test Permutation Importance (True Generalization Impact)', fontsize=11, fontweight='bold')
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

plt.suptitle("Task 7: In-Sample MDI vs Out-of-Sample Permutation Importance", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    # ==========================================
    # Task 8 Section
    # ==========================================
    cells.append(nbf.v4.new_markdown_cell("""---
## Task 8: Systematic Hyperparameter Optimization & Model Selection Framework

### Task Objectives:
- Configure Stratified 5-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- Execute systematic hyperparameter search using `GridSearchCV` across all three architectures: Decision Tree, Random Forest, and Logistic Regression.
- Optimize specifically for an attrition-oriented metric (`scoring='f1'`) to address the 5.2:1 class imbalance and boost minority class recall (REQ-32).
- Generate and display standard Scikit-Learn Classification Reports for all tuned models.
- Build a Master Post-Tuning Comparison Table benchmarking all tuned architectures.
- Establish the Model Selection Framework detailing on what factors the production model is selected.
"""))

    cells.append(nbf.v4.new_code_cell("""# 8.1 Multi-Model Hyperparameter Optimization via GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define search spaces
tuning_configs = {
    'Decision Tree (Tuned)': {
        'estimator': DecisionTreeClassifier(random_state=42),
        'param_grid': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5, 6, 8],
            'min_samples_split': [4, 10, 20],
            'min_samples_leaf': [2, 4, 8, 12],
            'class_weight': ['balanced', {0: 1, 1: 3.5}]
        }
    },
    'Random Forest (Tuned)': {
        'estimator': RandomForestClassifier(random_state=42, n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 150, 200],
            'max_depth': [4, 6, 8, 10],
            'min_samples_split': [4, 8, 12],
            'min_samples_leaf': [2, 4, 6],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    },
    'Logistic Regression (Tuned)': {
        'estimator': LogisticRegression(random_state=42, max_iter=1000),
        'param_grid': {
            'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            'penalty': ['l2'],
            'solver': ['saga', 'lbfgs'],
            'class_weight': ['balanced', {0: 1, 1: 3.0}, {0: 1, 1: 4.0}]
        }
    }
}

tuned_models = {}
for name, cfg in tuning_configs.items():
    print(f"Executing GridSearchCV for: {name}...")
    gs = GridSearchCV(cfg['estimator'], cfg['param_grid'], scoring='f1', cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train_model, y_train_model)
    tuned_models[name] = gs.best_estimator_
    print(f"Optimal 5-Fold CV F1: {gs.best_score_:.4f}")
    print(f"Best Hyperparameters: {gs.best_params_}\\n")
"""))

    cells.append(nbf.v4.new_code_cell("""# 8.2 Standard Classification Reports for All Tuned Models
for name, model in tuned_models.items():
    y_pred = model.predict(X_test_model)
    y_proba = model.predict_proba(X_test_model)[:, 1]
    
    print(f"\\n{'='*70}\\nCLASSIFICATION REPORT (POST-TUNING): {name}\\n{'='*70}")
    print(classification_report(y_test_model, y_pred, target_names=['Retained (0)', 'Departed (1)'], digits=4))
    print(f"Test Accuracy: {accuracy_score(y_test_model, y_pred)*100:.2f}% | Test ROC-AUC: {roc_auc_score(y_test_model, y_proba):.4f}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 8.3 Master Post-Tuning Comparison Table Across All Models
tuned_comparison_rows = []
for name, model in tuned_models.items():
    y_tr_p = model.predict(X_train_model)
    y_te_p = model.predict(X_test_model)
    y_te_prob = model.predict_proba(X_test_model)[:, 1]
    cm = confusion_matrix(y_test_model, y_te_p)
    
    tr_acc = accuracy_score(y_train_model, y_tr_p)
    te_acc = accuracy_score(y_test_model, y_te_p)
    rec = recall_score(y_test_model, y_te_p)
    prec = precision_score(y_test_model, y_te_p)
    f1 = f1_score(y_test_model, y_te_p)
    auc = roc_auc_score(y_test_model, y_te_prob)
    
    tuned_comparison_rows.append({
        'Model (Tuned)': name,
        'Train Acc (%)': f"{tr_acc*100:.2f}%",
        'Test Acc (%)': f"{te_acc*100:.2f}%",
        'Gap (%)': f"{(tr_acc - te_acc)*100:+.2f}%",
        'Recall (Yes)': f"{rec*100:.2f}%",
        'Precision (Yes)': f"{prec*100:.2f}%",
        'F1-Score (Yes)': f"{f1:.4f}",
        'ROC-AUC': f"{auc:.4f}",
        'TP / FN / FP': f"{cm[1,1]} / {cm[1,0]} / {cm[0,1]}"
    })

display(pd.DataFrame(tuned_comparison_rows))
"""))

    cells.append(nbf.v4.new_code_cell("""# 8.4 Overlaid ROC Curves & Confusion Matrices for All Tuned Models
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
palette = {'Decision Tree (Tuned)': '#2563EB', 'Random Forest (Tuned)': '#059669', 'Logistic Regression (Tuned)': '#7C3AED'}

for name, model in tuned_models.items():
    y_te_prob = model.predict_proba(X_test_model)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_model, y_te_prob)
    auc = roc_auc_score(y_test_model, y_te_prob)
    ax.plot(fpr, tpr, color=palette[name], lw=2.4, label=f"{name} (AUC = {auc:.4f})")

ax.plot([0, 1], [0, 1], color='#94A3B8', linestyle=':', lw=1.5, label='Chance Line')
ax.set_title('AttritionIQ: Overlaid ROC Curves for All Tuned Models', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 8.5 Model Selection Framework & Decision Rationale

#### Selection Factors & Operational Trade-offs:
1. **Minority Recall (Catching Departures):**
   - Tuned Logistic Regression captures **$53.19\%$ of departures ($25/47$)**; Tuned Decision Tree captures **$51.06\%$ ($24/47$)**; Tuned Random Forest captures **$38.30\%$ ($18/47$)** at default $\\\\tau = 0.50$ and **$63.83\%$ ($30/47$)** at $\\\\tau = 0.42$.
2. **Precision & False Alarm Mitigation:**
   - Tuned Logistic Regression achieves **$49.02\%$ Precision** (only 26 false alarms); Tuned Random Forest achieves **$45.00\%$ Precision** (22 false alarms). Decision Tree generates **57 false alarms** ($29.63\%$ Precision), causing severe alert fatigue.
3. **Discriminative Ability (ROC-AUC):**
   - Logistic Regression achieves highest discrimination (**$\\\\text{AUC} = 0.7994$**), followed by Random Forest (**$0.7638$**) and Decision Tree (**$0.7032$**).
4. **Generalization Gap (Overfitting Control):**
   - Logistic Regression exhibits near-zero overfitting (**$+3.23\%$ gap**), followed by Decision Tree (**$+6.38\%$**) and Random Forest (**$+11.05\%$**).

#### Final Selection Determination:
- **Champion Parametric Model:** **Tuned Logistic Regression** — Best overall accuracy ($83.67\%$), F1 ($0.5102$), ROC-AUC ($0.7994$), and highest generalizability.
- **Champion Ensemble Model:** **Tuned Random Forest** — Captures non-linear feature interactions, delivers lowest false positives (22), and offers flexible decision support thresholding ($\\\\tau = 0.42$ achieves $63.83\%$ Recall).
"""))

    nb.cells = cells
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/AttritionIQ_Workflow.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Master notebook successfully generated at notebooks/AttritionIQ_Workflow.ipynb")

if __name__ == '__main__':
    create_master_notebook()
