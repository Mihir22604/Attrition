# TASK 8 DETAILED REPORT: Systematic Model Optimization, Cross-Architecture Tuning & Production Selection Framework

**Project Title:** AttritionIQ — Intelligent Employee Attrition Prediction System  
**Task ID:** Task 8 — Systematic Hyperparameter Optimization, Multi-Model Tuning, Post-Tuning Classification Reports & Model Selection Framework  
**Status:** COMPLETED & EMPIRICALLY VERIFIED  
**Artifact Folder:** `reports/figures/`, `reports/classification_reports/`, `models/`  
**Author / Role:** Senior Machine Learning Project Analyst & Technical Documentation Specialist  
**Execution Environment:** Python 3.12 (venv) / Scikit-Learn 1.9.1 / NumPy 1.26 / Pandas 2.2 / Matplotlib 3.8 / Seaborn 0.13  
**Consolidated Workflow Script:** [task7_task8_runner.py](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/src/task7_task8_runner.py)  
**Evaluated Architectures in Tuning Workflow:**  
1. **Random Forest** — Non-linear bagging ensemble of decorrelated decision trees  
2. **Logistic Regression** — Parametric linear model with L2 regularization and cost-sensitive loss weighting  
*(Note: Decision Tree was evaluated in baseline Tasks 5 & 6 and is deliberately excluded from the Task 8 tuning and production workflow).*  
**Held-Out Evaluation Split:**  
- $\mathbf{X}_{\text{test}}$: `data/processed/test_features_preprocessed.csv` ($N_{\text{test}} = 294$ observations $\times$ $44$ features)  
- $\mathbf{y}_{\text{test}}$: Out-of-sample Ground Truth ($247$ Retained `0`, $47$ Departed `1`, $15.99\%$ positive prevalence)  

---

## 1. Executive Summary & Optimization Strategy

The primary objective of **Task 8: Model Optimization** is to systematically tune the hyperparameters across the primary candidate architectures (**Random Forest** and **Logistic Regression**), benchmark performance before and after tuning on strictly untouched held-out data, and establish an objective, factor-based framework explaining **why Tuned Logistic Regression is selected as the final deployed production model** while **Random Forest is retained for feature interpretability and comparative analysis**.

### 1.1 Baseline Deficiencies Addressed
During baseline evaluation (Task 6), un-tuned models exhibited distinct operational challenges:
- **Random Forest (Baseline):** Although demonstrating high accuracy ($82.31\%$), baseline tree bagging captured only $21$ of $47$ voluntary departures ($44.68\%$ recall), leaving significant retention risk unflagged.
- **Logistic Regression (Baseline):** Delivered strong sensitivity ($65.96\%$ recall, capturing $31 / 47$ departures), but generated $56$ false alarms ($35.63\%$ precision) due to uncalibrated unit regularization ($C=1.0$).

### 1.2 Core Methodological Safeguards & Data Leakage Prevention
To guarantee scientific validity and prevent optimistic bias:
1. **Stratified Cross-Validation:** Hyperparameter optimization utilized 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) applied strictly to the training split ($N = 1,176$).
2. **Targeted Optimization Metric:** In strict compliance with project requirements (REQ-32), the search objective maximized **F1-Score on the minority positive class (`Attrition = Yes`)**, balancing departure capture against alert fatigue.
3. **Leakage-Safe Threshold Exploration:** Operational decision threshold exploration was conducted using **out-of-fold cross-validation probabilities on the training set**, ensuring the held-out test split remained strictly untouched.
4. **Untouched Test Evaluation:** Final before-and-after evaluations were computed on the untouched $20\%$ test partition ($N = 294$).

---

## 2. Model Tuning Details & Best Hyperparameters

### 2.1 Model 1: Random Forest
- **Tuning Search Space:**
  - `n_estimators`: `[100, 150, 200]`
  - `max_depth`: `[4, 6, 8, 10]`
  - `min_samples_split`: `[4, 8, 12]`
  - `min_samples_leaf`: `[2, 4, 6]`
  - `class_weight`: `['balanced', 'balanced_subsample', {0: 1, 1: 3.5}]`
- **Optimal Hyperparameters Selected via 5-Fold CV:**
  - `n_estimators`: `100`
  - `max_depth`: `10`
  - `min_samples_split`: `8`
  - `min_samples_leaf`: `2`
  - `class_weight`: `'balanced'`
- **Best 5-Fold Cross-Validation F1-Score:** `0.5045`
- **Serialized Estimator Artifact:** [random_forest_tuned.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/random_forest_tuned.joblib)

### 2.2 Model 2: Logistic Regression (Production Champion)
- **Tuning Search Space:**
  - `C`: `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]`
  - `penalty`: `['l1', 'l2']`
  - `solver`: `['liblinear', 'saga']`
  - `class_weight`: `['balanced', {0: 1, 1: 3.0}, {0: 1, 1: 4.0}, {0: 1, 1: 5.0}]`
- **Optimal Hyperparameters Selected via 5-Fold CV:**
  - `C`: `0.05` (Stronger L2 shrinkage suppressing collinear noise)
  - `penalty`: `'l2'`
  - `solver`: `'saga'`
  - `class_weight`: `{0: 1, 1: 3.0}` (Calibrated $3:1$ cost-sensitive penalty)
- **Best 5-Fold Cross-Validation F1-Score:** `0.5896`
- **Serialized Estimator Artifact:** [logistic_regression_tuned.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/logistic_regression_tuned.joblib)
- **Unified Production Pipeline Artifact:** [attrition_model_pipeline.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/attrition_model_pipeline.joblib)

---

## 3. Comprehensive Before vs After Evaluation Benchmark

All models were evaluated on the exact held-out test split ($N = 294$, containing $247$ Retained and $47$ Departed employees).

### 3.1 Performance Comparison Table

| Evaluation Entry | Model Architecture | Tuning Status | Accuracy | Precision (`Yes`) | Recall (`Yes`) | F1-Score (`Yes`) | ROC-AUC | TP | FN | FP | TN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entry 1** | Random Forest | **Before Tuning** | 82.31% | 44.68% | 44.68% | 0.4468 | 0.7660 | 21 | 26 | 26 | 221 |
| **Entry 2** | Random Forest | **After Tuning** | 81.63% | 42.86% | 44.68% | 0.4375 | 0.7706 | 21 | 26 | 28 | 219 |
| **Entry 3** | Logistic Regression | **Before Tuning** | 75.51% | 35.63% | **65.96%** | 0.4627 | **0.8035** | **31** | 16 | 56 | 191 |
| **Entry 4** | Logistic Regression | **After Tuning (Champion)** | **83.67%** | **49.02%** | **53.19%** | **0.5102** | **0.7994** | **25** | 22 | **26** | **221** |

### 3.2 Performance Diagnostic Insights
1. **Logistic Regression Massive Accuracy & Precision Boost:** Post-tuning L2 shrinkage ($C=0.05$) combined with $\{0: 1, 1: 3.0\}$ cost-weighting reduced false alarms from $56$ down to $26$ (a **$53.6\%$ reduction in false positives**), lifting Precision from $35.63\%$ to $49.02\%$ and overall Test Accuracy from $75.51\%$ to **$83.67\%$**.
2. **Superior F1-Score Equilibrium:** Tuned Logistic Regression achieved the highest overall minority F1-score (**$0.5102$**), striking an optimal trade-off between capturing voluntary turnover and preventing HR alert fatigue.
3. **Discriminative Capacity:** Both baseline and tuned Logistic Regression demonstrated the highest ROC-AUC values ($0.8035$ and $0.7994$), proving superior ranking capability across probability thresholds compared to tree ensembles.

---

## 4. Required Task 8 Deliverables & Artifact Manifest

### 4.1 Combined Classification Report
The combined classification report benchmarks all four evaluation states in a single consolidated artifact:
- **Combined Report Image:** [task8_combined_classification_report.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_combined_classification_report.png)
- **Combined Report JSON:** [task8_combined_classification_report.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/task8_combined_classification_report.json)

![Combined Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_combined_classification_report.png)
*Figure 8.1: AttritionIQ Task 8 Combined Standard Classification Report for Both Architectures Before and After Hyperparameter Tuning.*

---

### 4.2 Four Separate Confusion Matrices
Four dedicated confusion matrices illustrate actual vs. predicted classifications with explicit TN, FP, FN, and TP counts and percentages:

1. **Random Forest — Before Tuning:**  
   Artifact: [task8_random_forest_confusion_matrix_before.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_random_forest_confusion_matrix_before.png)  
   *Metrics: TN=221 (75.2%), FP=26 (8.8%), FN=26 (8.8%), TP=21 (7.1%) | Recall: 44.68% | Precision: 44.68%*
2. **Random Forest — After Tuning:**  
   Artifact: [task8_random_forest_confusion_matrix_after.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_random_forest_confusion_matrix_after.png)  
   *Metrics: TN=219 (74.5%), FP=28 (9.5%), FN=26 (8.8%), TP=21 (7.1%) | Recall: 44.68% | Precision: 42.86%*
3. **Logistic Regression — Before Tuning:**  
   Artifact: [task8_logistic_regression_confusion_matrix_before.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_logistic_regression_confusion_matrix_before.png)  
   *Metrics: TN=191 (65.0%), FP=56 (19.0%), FN=16 (5.4%), TP=31 (10.5%) | Recall: 65.96% | Precision: 35.63%*
4. **Logistic Regression — After Tuning (Champion):**  
   Artifact: [task8_logistic_regression_confusion_matrix_after.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_logistic_regression_confusion_matrix_after.png)  
   *Metrics: TN=221 (75.2%), FP=26 (8.8%), FN=22 (7.5%), TP=25 (8.5%) | Recall: 53.19% | Precision: 49.02%*

---

### 4.3 Four Separate ROC Curves
Four individual ROC plots illustrate sensitivity (TPR) vs. 1-specificity (FPR) across decision thresholds against the diagonal chance baseline:

1. **Random Forest — Before Tuning:**  
   Artifact: [task8_random_forest_roc_before.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_random_forest_roc_before.png) | **$\text{ROC-AUC} = 0.7660$**
2. **Random Forest — After Tuning:**  
   Artifact: [task8_random_forest_roc_after.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_random_forest_roc_after.png) | **$\text{ROC-AUC} = 0.7706$**
3. **Logistic Regression — Before Tuning:**  
   Artifact: [task8_logistic_regression_roc_before.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_logistic_regression_roc_before.png) | **$\text{ROC-AUC} = 0.8035$**
4. **Logistic Regression — After Tuning (Champion):**  
   Artifact: [task8_logistic_regression_roc_after.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_logistic_regression_roc_after.png) | **$\text{ROC-AUC} = 0.7994$**

---

## 5. Model Selection Framework: Why Logistic Regression is the Production Champion

In production workforce analytics, model selection balances predictive accuracy, alert fatigue, interpretability, and computational stability. The selection framework formally determines:

### 5.1 Why Logistic Regression is Selected for Production
1. **Top Metric Performance:** Tuned Logistic Regression delivers the **highest Test Accuracy ($83.67\%$)**, **highest Minority F1-Score ($0.5102$)**, **highest Precision ($49.02\%$)**, and **highest ROC-AUC ($0.7994$)** among all tuned models.
2. **Low Alert Fatigue:** By reducing false alarms from $56$ to $26$, HR business partners are not overwhelmed by spurious notifications.
3. **Probability Calibration & Monotonicity:** Logistic regression outputs smooth, well-calibrated posterior probabilities $\sigma(w^T x + b) \in [0, 1]$, providing HR leaders with an interpretable continuum of departure risk rather than step-function artifacts.
4. **Auditable Linearity for Responsible AI:** Log-odds weights provide direct transparency, ensuring that risk factors (e.g., compulsory overtime, equity tier, commute distance) can be audited and communicated clearly to stakeholders.

### 5.2 Why Random Forest Remains in the Project
1. **Core Engine for Task 7 Feature Importance:** Random Forest provides Mean Decrease in Impurity (MDI) across 100 decorrelated trees and powers Permutation Feature Importance (PFI) on the test partition.
2. **Non-Linear Benchmark:** Random Forest remains in `models/random_forest_tuned.joblib` as a comparative benchmark to monitor complex interactions (e.g., OverTime $\times$ Compensation tier).

### 5.3 Exclusion of Decision Tree
Decision Tree was evaluated in Tasks 5 & 6 and demonstrated severe vulnerability to memorization ($100\%$ training accuracy vs $76.5\%$ testing accuracy) and low precision ($29.6\%$, generating 57 false alarms). It was deliberately excluded from Task 8 hyperparameter tuning and deployment workflows.

---

## 6. Execution Instructions for the Consolidated Runner

To re-execute the complete consolidated workflow and regenerate all Task 7 and Task 8 outputs:

```powershell
# Run consolidated Task 7 & Task 8 workflow in virtual environment
.\venv\Scripts\python src/task7_task8_runner.py
```

### Generated Artifact Deliverables Summary
- **Task 7 Interpretability Outputs:**
  - `reports/figures/task7_rf_feature_importance_mdi.png`
  - `reports/figures/task7_parent_feature_importance_mdi.png`
  - `reports/figures/task7_permutation_importance_test.png`
  - `reports/figures/task7_mdi_vs_permutation_comparison.png`
  - `reports/task7_feature_importance_rankings.csv`
  - `reports/task7_feature_importance_summary.json`
- **Task 8 Optimization & Benchmark Outputs:**
  - `reports/figures/task8_combined_classification_report.png`
  - `reports/classification_reports/task8_combined_classification_report.json`
  - `reports/figures/task8_random_forest_confusion_matrix_before.png`
  - `reports/figures/task8_random_forest_confusion_matrix_after.png`
  - `reports/figures/task8_logistic_regression_confusion_matrix_before.png`
  - `reports/figures/task8_logistic_regression_confusion_matrix_after.png`
  - `reports/figures/task8_random_forest_roc_before.png`
  - `reports/figures/task8_random_forest_roc_after.png`
  - `reports/figures/task8_logistic_regression_roc_before.png`
  - `reports/figures/task8_logistic_regression_roc_after.png`
  - `reports/task8_hyperparameter_tuning_summary.json`
- **Production Model Artifacts:**
  - `models/attrition_model_pipeline.joblib` (Champion Logistic Regression Pipeline)
  - `models/attrition_logistic_pipeline.joblib`
  - `models/logistic_regression_tuned.joblib`
  - `models/random_forest_tuned.joblib`
