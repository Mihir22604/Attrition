# TASK 8 DETAILED REPORT: Comprehensive Multi-Model Hyperparameter Tuning, Classification Reports & Selection Analysis

**Project Title:** AttritionIQ — Intelligent Employee Attrition Prediction System  
**Task ID:** Task 8 — Systematic Hyperparameter Optimization, Multi-Model Tuning, Post-Tuning Classification Reports & Model Selection Framework  
**Status:** COMPLETED & EMPIRICALLY VERIFIED  
**Artifact Folder:** `reports/figures/`, `reports/classification_reports/`, `models/`  
**Author / Role:** Senior Machine Learning Project Analyst & Technical Documentation Specialist  
**Execution Environment:** Python 3.12 / Scikit-Learn 1.6.1 / NumPy 1.26 / Pandas 2.2 / Matplotlib 3.8 / Seaborn 0.13  
**Evaluated Architectures:**  
1. **Decision Tree (Tuned & Pruned)** — White-box rule-based tree estimator  
2. **Random Forest (Tuned & Optimized)** — Non-linear bagging ensemble of 200 decorrelated trees  
3. **Logistic Regression (Tuned & Calibrated)** — Parametric linear model with L2 regularization and weighted loss  
**Held-Out Evaluation Split:**  
- $\mathbf{X}_{\text{test}}$: `data/processed/test_features_preprocessed.csv` ($N_{\text{test}} = 294$ observations $\times$ $44$ features)  
- $\mathbf{y}_{\text{test}}$: Out-of-sample Ground Truth ($247$ Retained `0`, $47$ Departed `1`, $15.99\%$ positive prevalence)  

---

## 1. Executive Summary & Multi-Model Tuning Strategy

The primary objective of **Task 8: Model Optimization** is to systematically tune the hyperparameters across all candidate architectures evaluated in Task 5 & 6, generate standard post-tuning Scikit-Learn classification reports for each model, and establish an objective, factor-based framework explaining **how and why the final model is selected for production deployment**.

### 1.1 The Baseline Problem Identified in Task 6
During baseline evaluation (Task 6), un-tuned models exhibited distinct structural drawbacks:
- **Decision Tree (Unconstrained):** Suffered from severe memorization ($100\%$ train accuracy vs $76.53\%$ test accuracy, a $+23.47\%$ generalization gap), with low test precision ($31.03\%$).
- **Random Forest (Default):** Exhibited low minority class sensitivity, catching only **$10$ out of $47$ departing employees** ($21.28\%$ Recall), missing nearly four-fifths of voluntary resignations.
- **Logistic Regression (Default):** Showed promising balance (Recall $65.96\%$), but produced $56$ false alarms ($FP$) due to crude $1.0$ penalty scaling.

### 1.2 Core Optimization Mandates Executed
To resolve these deficiencies, systematic hyperparameter optimization was executed:
1. **Cross-Validation Design:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) applied across all models, ensuring identical class distribution (~$16.1\%$ positive prevalence) across all folds.
2. **Targeted Optimization Metric:** Rather than optimizing for misleading raw accuracy, hyperparameter tuning explicitly optimized for **F1-Score on the minority class (`Attrition = Yes`)**, balancing departure capture (Recall) against alert fatigue (Precision).
3. **Leakage-Free Execution:** Transformations and scaling parameters were frozen to training splits, and final evaluation was conducted on the held-out test split ($N = 294$) untouched during search.

---

## 2. Model-by-Model Tuning Details & Search Configurations

Below is the complete audit of what hyperparameter tuning was conducted for each candidate architecture:

### 2.1 Model 1: Decision Tree (Tuned & Pruned)
- **Tuning Search Space:**
  - `criterion`: `['gini', 'entropy']` (Impurity split criteria)
  - `max_depth`: `[3, 4, 5, 6, 8]` (Tree hierarchy restrictions)
  - `min_samples_split`: `[4, 10, 20]` (Minimum node split requirement)
  - `min_samples_leaf`: `[2, 4, 8, 12]` (Terminal leaf smoothing)
  - `class_weight`: `['balanced', {0: 1, 1: 3.5}, {0: 1, 1: 5.0}]` (Cost-sensitive error penalty)
- **Optimal Hyperparameters Selected via 5-Fold CV:**
  - `criterion`: `'gini'`
  - `max_depth`: `4`
  - `min_samples_leaf`: `12`
  - `min_samples_split`: `4`
  - `class_weight`: `{0: 1, 1: 3.5}`
- **Best Cross-Validation F1-Score:** `0.4441`
- **Serialized Model Artifact:** [dt_tuned.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/dt_tuned.joblib)

### 2.2 Model 2: Random Forest (Tuned & Optimized)
- **Tuning Search Space:**
  - `n_estimators`: `[100, 150, 200]` (Ensemble tree count)
  - `max_depth`: `[4, 6, 8, 10]` (Individual tree depth regularization)
  - `min_samples_split`: `[4, 8, 12]` (Internal node split criteria)
  - `min_samples_leaf`: `[2, 4, 6]` (Terminal leaf sample threshold)
  - `class_weight`: `['balanced', 'balanced_subsample', {0: 1, 1: 3.5}]` (Cost-sensitive weighting)
- **Optimal Hyperparameters Selected via 5-Fold CV:**
  - `n_estimators`: `200`
  - `max_depth`: `6`
  - `min_samples_leaf`: `4`
  - `min_samples_split`: `4`
  - `class_weight`: `'balanced'`
- **Best Cross-Validation F1-Score:** `0.4730`
- **Serialized Model Artifact:** [random_forest_tuned.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/random_forest_tuned.joblib)

### 2.3 Model 3: Logistic Regression (Tuned & Calibrated)
- **Tuning Search Space:**
  - `C`: `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]` (Inverse regularization strength)
  - `penalty`: `['l1', 'l2']` (Lasso feature selection vs Ridge shrinkage)
  - `solver`: `['liblinear', 'saga']` (Convex optimization algorithms)
  - `class_weight`: `['balanced', {0: 1, 1: 3.0}, {0: 1, 1: 4.0}, {0: 1, 1: 5.0}]` (Loss weighting)
- **Optimal Hyperparameters Selected via 5-Fold CV:**
  - `C`: `0.05` (Stronger L2 shrinkage to suppress collinear noise)
  - `penalty`: `'l2'`
  - `solver`: `'saga'`
  - `class_weight`: `{0: 1, 1: 3.0}` (Tailored $3:1$ penalty ratio)
- **Best Cross-Validation F1-Score:** `0.5896`
- **Serialized Model Artifact:** [logistic_regression_tuned.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/logistic_regression_tuned.joblib)

---

## 3. Standard Scikit-Learn Classification Reports (All Tuned Models)

Below are the standard Scikit-Learn classification reports for all three tuned architectures evaluated on the held-out test split ($N = 294$, containing 247 Retained and 47 Departed employees):

---

### 3.1 Model 1: Decision Tree (Tuned & Pruned)

- **Artifact JSON:** [classification_report_dt_tuned.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_dt_tuned.json)
- **Artifact Image:** [classification_report_dt_tuned.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_dt_tuned.png)

#### Standard Classification Report Table:
| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (0)** | 0.8920 | 0.7692 | 0.8261 | 247 |
| **Departed (1)** | **0.2963** | **0.5106** | **0.3750** | **47** |
| **accuracy** | — | — | **0.7279** | **294** |
| **macro avg** | 0.5942 | 0.6399 | 0.6005 | 294 |
| **weighted avg** | 0.7968 | 0.7279 | 0.7540 | 294 |

**Performance Diagnostic:**
- **Test Accuracy:** $72.79\%$ ($214 / 294$ correct)
- **Minority Recall (`Attrition = Yes`):** $51.06\%$ ($24 / 47$ departures captured; $23$ missed)
- **Minority Precision:** $29.63\%$ ($57$ false positives)
- **ROC-AUC Score:** $0.7032$
- **Overfitting Gap:** $+6.38\%$ (Pruned depth successfully eliminated the $+23.47\%$ memorization gap)

![Decision Tree Tuned Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_dt_tuned.png)
*Figure 8.1: Standard Classification Report for Decision Tree (Tuned & Pruned).*

---

### 3.2 Model 2: Random Forest (Tuned & Optimized)

- **Artifact JSON:** [classification_report_random_forest_tuned.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_random_forest_tuned.json)
- **Artifact Image:** [classification_report_random_forest_tuned.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_random_forest_tuned.png)

#### Standard Classification Report Table:
| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (0)** | 0.8858 | 0.9109 | 0.8982 | 247 |
| **Departed (1)** | **0.4500** | **0.3830** | **0.4138** | **47** |
| **accuracy** | — | — | **0.8265** | **294** |
| **macro avg** | 0.6679 | 0.6470 | 0.6560 | 294 |
| **weighted avg** | 0.8162 | 0.8265 | 0.8208 | 294 |

**Performance Diagnostic:**
- **Test Accuracy:** $82.65\%$ ($243 / 294$ correct)
- **Minority Recall (`Attrition = Yes`):** $38.30\%$ ($18 / 47$ departures captured; $+80.0\%$ boost over baseline $21.28\%$)
- **Minority Precision:** $45.00\%$ ($22$ false positives — lowest false alarm rate among tree models)
- **Minority F1-Score:** $0.4138$ ($+51.0\%$ relative boost over baseline $0.2740$)
- **ROC-AUC Score:** $0.7638$
- **Overfitting Gap:** $+11.05\%$ (Down from $+15.65\%$ baseline)

![Random Forest Tuned Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_random_forest_tuned.png)
*Figure 8.2: Standard Classification Report for Random Forest (Tuned & Optimized).*

---

### 3.3 Model 3: Logistic Regression (Tuned & Calibrated)

- **Artifact JSON:** [classification_report_logistic_regression_tuned.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_logistic_regression_tuned.json)
- **Artifact Image:** [classification_report_logistic_regression_tuned.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_logistic_regression_tuned.png)

#### Standard Classification Report Table:
| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (0)** | 0.9095 | 0.8947 | 0.9020 | 247 |
| **Departed (1)** | **0.4902** | **0.5319** | **0.5102** | **47** |
| **accuracy** | — | — | **0.8367** | **294** |
| **macro avg** | 0.6998 | 0.7133 | 0.7061 | 294 |
| **weighted avg** | 0.8424 | 0.8367 | 0.8394 | 294 |

**Performance Diagnostic:**
- **Test Accuracy:** **$83.67\%$** ($246 / 294$ correct — Highest Overall Accuracy)
- **Minority Recall (`Attrition = Yes`):** **$53.19\%$** ($25 / 47$ departures captured)
- **Minority Precision:** **$49.02\%$** (Highest Precision Across All Models; only $26$ false alarms)
- **Minority F1-Score:** **$0.5102$** (Highest F1-Score Across All Architectures)
- **ROC-AUC Score:** **$0.7994$** (Highest Discriminative Capacity)
- **Overfitting Gap:** **$+3.23\%$** (Near-Optimal Generalization Stability)

![Logistic Regression Tuned Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_logistic_regression_tuned.png)
*Figure 8.3: Standard Classification Report for Logistic Regression (Tuned & Calibrated).*

---

### 3.4 Combined Multi-Model Benchmark (Post-Tuning)
Below is the combined side-by-side visualization benchmarking all three tuned models on the identical test split:

- **Combined JSON Report:** [combined_classification_report_all_tuned_models.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/combined_classification_report_all_tuned_models.json)
- **Combined Publication Image:** [combined_classification_report_all_tuned_models.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/combined_classification_report_all_tuned_models.png)

![Combined Tuned Classification Reports](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/combined_classification_report_all_tuned_models.png)
*Figure 8.4: Combined Scikit-Learn Classification Reports for All Three Models After Hyperparameter Tuning.*

---

## 4. Master Cross-Architecture Comparison Table (After Tuning)

The table below contrasts the post-tuning performance across all three architectures on the held-out test dataset ($N_{\text{test}} = 294$, containing $247$ Retained and $47$ Departed employees):

| Evaluation Metric | Decision Tree (Tuned & Pruned) | Random Forest (Tuned & Optimized) | Logistic Regression (Tuned & Calibrated) | Optimal Architecture for Metric |
| :--- | :---: | :---: | :---: | :--- |
| **Training Accuracy** | $79.17\%$ | $93.71\%$ | $86.90\%$ | Controlled Complexity across all |
| **Testing Accuracy** | $72.79\%$ | $82.65\%$ | **$83.67\%$** | **Logistic Regression** ($+10.88\%$ vs DT) |
| **Generalization Gap ($\Delta_{\text{Acc}}$)** | $+6.38\%$ | $+11.05\%$ | **$+3.23\%$** | **Logistic Regression** (Most Generalizable) |
| **Precision (`Attrition = Yes`)** | $29.63\%$ | $45.00\%$ | **$49.02\%$** | **Logistic Regression** (Lowest False Alarms) |
| **Recall (`Attrition = Yes`)** | $51.06\%$ | $38.30\%$ *(or $63.8\%$ at $\tau=0.42$)* | **$53.19\%$** | **Logistic Regression** *(Default $\tau$)* |
| **F1-Score (`Attrition = Yes`)** | $0.3750$ | $0.4138$ *(or $0.4959$ at $\tau=0.42$)* | **$0.5102$** | **Logistic Regression** ($+23.3\%$ vs RF default) |
| **ROC-AUC Score** | $0.7032$ | $0.7638$ | **$0.7994$** | **Logistic Regression** (Highest Discrimination) |
| **True Positives (TP)** | 24 | 18 *(or $30$ at $\tau=0.42$)* | **25** | **Logistic Regression** *(Default $\tau$)* |
| **False Negatives (FN)** | 23 | 29 *(or $17$ at $\tau=0.42$)* | **22** | **Logistic Regression** *(Default $\tau$)* |
| **False Positives (FP)** | 57 | **22** *(or $44$ at $\tau=0.42$)* | 26 | **Random Forest** (Lowest False Alarms) |
| **True Negatives (TN)** | 190 | **225** | 221 | **Random Forest** (Highest Specificity) |

---

## 5. Diagnostic Comparison Visualizations

### 5.1 Overlaid ROC Curves Across All Tuned Models
The overlaid ROC curves demonstrate that **Tuned Logistic Regression ($\text{AUC} = 0.7994$)** achieves superior sensitivity-specificity trade-offs across nearly all decision thresholds, closely followed by **Tuned Random Forest ($\text{AUC} = 0.7638$)**, while **Tuned Decision Tree ($\text{AUC} = 0.7032$)** exhibits piecewise step-function limitations.

![Overlaid ROC Curves for Tuned Models](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_all_tuned_models_roc_curves.png)
*Figure 8.5: Overlaid ROC Curves for All Three Tuned Architectures on Held-Out Test Set.*

---

### 5.2 Confusion Matrix Breakdown Across Tuned Models
The 3-panel confusion matrix breakdown highlights the precise operational impact:
- **Decision Tree:** High recall ($24$ TPs), but excessive false alarms ($57$ FPs, $29.6\%$ precision).
- **Random Forest:** Outstanding precision ($45.0\%$, only $22$ FPs) and highest True Negatives ($225$), with balanced departure detection ($18$ TPs).
- **Logistic Regression:** Superb operational equilibrium: $25$ TPs, only $22$ FNs, $26$ FPs, and $49.0\%$ precision.

![All Tuned Models Confusion Matrices](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_all_tuned_models_confusion_matrices.png)
*Figure 8.6: 3-Panel Confusion Matrix Comparison Across All Three Tuned Models.*

---

## 6. Model Selection Framework: How and On What Factors the Model is Selected

In enterprise data science, selecting a winning machine learning model must never be based on arbitrary preference or naive accuracy alone. The selection framework for AttritionIQ evaluates candidate architectures across **five explicit technical and operational factors**:

```
                       MODEL SELECTION EVALUATION MATRIX
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Factor 1: Minority Class Sensitivity (Recall on Attrition = Yes)                 │
│ Factor 2: Operational False Positive Tolerance (Alert Fatigue vs Precision)      │
│ Factor 3: Discriminative Power & Probability Calibration (ROC-AUC)               │
│ Factor 4: Generalization Stability & Overfitting Resistance (Delta Accuracy)     │
│ Factor 5: Model Interpretability & Operational Deployment Viability              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Factor-by-Factor Evaluation Analysis

#### Factor 1: Minority Class Sensitivity (Recall)
- **Business Rationale:** In voluntary attrition prediction, missing an employee who departs ($FN$) carries severe financial penalties (\$30,000–\$100,000+ in replacement costs). Higher recall is paramount.
- **Model Comparison:** Tuned Logistic Regression captures **$53.19\%$ of departures ($25 / 47$)** at default threshold. Tuned Decision Tree captures **$51.06\%$ ($24 / 47$)**. Tuned Random Forest captures **$38.30\%$ ($18 / 47$)** at default threshold, but can be calibrated via threshold slider ($\tau = 0.42$) to achieve **$63.83\%$ ($30 / 47$)**.
- **Assessment:** **Logistic Regression** wins at default cutoff; **Random Forest** wins when calibrated thresholding is permitted.

#### Factor 2: Precision and False Alarm Mitigation
- **Business Rationale:** Excessive false positives ($FP$) induce "alert fatigue" among HR business partners and risk intrusive or awkward employee conversations.
- **Model Comparison:** Tuned Logistic Regression achieves **$49.02\%$ Precision** ($26$ FPs). Tuned Random Forest achieves **$45.00\%$ Precision** ($22$ FPs). Decision Tree achieves only **$29.63\%$ Precision** ($57$ FPs, generating more false alarms than actual cases).
- **Assessment:** **Logistic Regression** and **Random Forest** are both operationally sound; **Decision Tree is rejected** due to unacceptable false positive overhead.

#### Factor 3: Discriminative Power (ROC-AUC)
- **Business Rationale:** ROC-AUC evaluates the model's ability to rank departing employees above retained employees across all classification thresholds.
- **Model Comparison:** 
  - Tuned Logistic Regression: **$0.7994$**
  - Tuned Random Forest: **$0.7638$**
  - Tuned Decision Tree: **$0.7032$**
- **Assessment:** **Logistic Regression** provides the highest discriminative power.

#### Factor 4: Generalization Stability and Overfitting Gap
- **Business Rationale:** A model with a large gap between training and testing accuracy memorizes historical noise and will degrade in production.
- **Model Comparison:**
  - Tuned Logistic Regression: **$+3.23\%$** ($86.9\%$ train vs $83.7\%$ test) — Exceptional stability.
  - Tuned Decision Tree: **$+6.38\%$** ($79.2\%$ train vs $72.8\%$ test).
  - Tuned Random Forest: **$+11.05\%$** ($93.7\%$ train vs $82.7\%$ test).
- **Assessment:** **Logistic Regression** demonstrates the highest generalizability.

#### Factor 5: Interpretability and Operational Suitability
- **Business Rationale:** HR partners must be able to explain *why* an employee received an elevated departure risk score.
- **Model Comparison:**
  - **Decision Tree:** Perfect visual interpretability, but poor performance ($F1 = 0.3750$).
  - **Logistic Regression:** Log-odds coefficients provide direct linear interpretability and smooth, well-calibrated posterior probabilities.
  - **Random Forest:** Ensemble bagging captures non-linear interactions (e.g., OverTime $\times$ Low Income) and provides MDI and Permutation importance.
- **Assessment:** **Logistic Regression** and **Random Forest** both provide robust interpretability.

---

### 6.2 The Final Model Selection Determination

Based on the multi-factor evaluation, the formal architecture selection is established as follows:

1. **CHAMPION PARAMETRIC MODEL: Tuned Logistic Regression (`logistic_regression_tuned.joblib`)**
   - **Why Selected:** Achieves the **highest overall Test Accuracy ($83.67\%$)**, **highest Minority Recall ($53.19\%$)**, **highest Precision ($49.02\%$)**, **highest F1-Score ($0.5102$)**, **highest ROC-AUC ($0.7994$)**, and **lowest generalization gap ($+3.23\%$)**. It serves as the most accurate, reliable, and well-calibrated baseline for continuous probability inference.

2. **CHAMPION ENSEMBLE MODEL: Tuned Random Forest (`random_forest_tuned.joblib`)**
   - **Why Selected:** Selected as the **primary non-linear tree ensemble**. It exhibits outstanding specificity ($91.1\%$), lowest false alarm count ($22$ FPs), strong precision ($45.0\%$), and superior non-linear feature interaction capabilities. Furthermore, when paired with the interactive probability threshold slider ($\tau = 0.42$) in Streamlit, it achieves **$63.83\%$ Recall ($30 / 47$ departures detected)**.

3. **ELIMINATED MODEL: Decision Tree**
   - **Why Rejected:** Despite regularized depth ($max\_depth=4$), Decision Tree suffers from low precision ($29.63\%$), generating $57$ false positives on $47$ actual cases. It is unsuitable as the sole production engine.

---

## 7. Deliverables and Artifact Manifest

The complete set of post-tuning classification reports and artifacts produced is cataloged below:

| Deliverable ID | Artifact File / Path | File Type | Description |
| :--- | :--- | :---: | :--- |
| **DEL-8.1** | [classification_report_dt_tuned.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_dt_tuned.json) | JSON | Standard Scikit-Learn classification report for Tuned Decision Tree. |
| **DEL-8.2** | [classification_report_random_forest_tuned.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_random_forest_tuned.json) | JSON | Standard Scikit-Learn classification report for Tuned Random Forest. |
| **DEL-8.3** | [classification_report_logistic_regression_tuned.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_logistic_regression_tuned.json) | JSON | Standard Scikit-Learn classification report for Tuned Logistic Regression. |
| **DEL-8.4** | [combined_classification_report_all_tuned_models.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/combined_classification_report_all_tuned_models.json) | JSON | Combined cross-architecture classification reports and metric comparison. |
| **DEL-8.5** | [classification_report_dt_tuned.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_dt_tuned.png) | PNG Image | Publication-grade standard report image for Tuned Decision Tree. |
| **DEL-8.6** | [classification_report_random_forest_tuned.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_random_forest_tuned.png) | PNG Image | Publication-grade standard report image for Tuned Random Forest. |
| **DEL-8.7** | [classification_report_logistic_regression_tuned.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_logistic_regression_tuned.png) | PNG Image | Publication-grade standard report image for Tuned Logistic Regression. |
| **DEL-8.8** | [combined_classification_report_all_tuned_models.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/combined_classification_report_all_tuned_models.png) | PNG Image | Combined 3-panel publication comparison of all tuned model reports. |
| **DEL-8.9** | [task8_all_tuned_models_roc_curves.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_all_tuned_models_roc_curves.png) | PNG Image | Overlaid ROC curves comparing all three tuned models against chance baseline. |
| **DEL-8.10** | [task8_all_tuned_models_confusion_matrices.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/task8_all_tuned_models_confusion_matrices.png) | PNG Image | 3-panel confusion matrix breakdown comparing all three tuned models. |
| **DEL-8.11** | [random_forest_tuned.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/random_forest_tuned.joblib) | Joblib Model | Fitted tuned Random Forest estimator. |
| **DEL-8.12** | [logistic_regression_tuned.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/logistic_regression_tuned.joblib) | Joblib Model | Fitted tuned Logistic Regression estimator. |
| **DEL-8.13** | [dt_tuned.joblib](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/models/dt_tuned.joblib) | Joblib Model | Fitted tuned Decision Tree estimator. |
