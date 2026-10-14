# TASK 6 DETAILED REPORT: Model Evaluation, Comparison and ROC Analysis

**Project Title:** AttritionIQ — Intelligent Employee Attrition Prediction System  
**Task ID:** Task 6 — Model Evaluation, Cross-Architecture Comparison & Generalization Assessment  
**Status:** COMPLETED & EMPIRICALLY VERIFIED  
**Artifact Folder:** `reports/figures/`, `reports/classification_reports/`, `models/`  
**Author / Role:** Senior Machine Learning Project Analyst & Technical Documentation Specialist  
**Execution Environment:** Python 3.12 / Scikit-Learn 1.6.1 / NumPy 1.26 / Pandas 2.2 / Matplotlib 3.8 / Seaborn 0.13  
**Held-Out Evaluation Split:**  
- $\mathbf{X}_{\text{test}}$: `data/processed/test_features_preprocessed.csv` ($N_{\text{test}} = 294$ observations $\times$ $44$ features)  
- $\mathbf{y}_{\text{test}}$: Out-of-sample Ground Truth ($247$ Retained `0`, $47$ Departed `1`, $15.99\%$ positive prevalence)  

---

## 1. Executive Summary and Evaluation Framework

The primary objective of **Task 6: Model Evaluation and Comparison** is to execute an empirical performance evaluation across all classification architectures developed in Task 5:
1. **Decision Tree (Unconstrained Baseline)**
2. **Decision Tree (Regularized & Balanced)**
3. **Random Forest (Balanced Ensemble)**
4. **Logistic Regression (Balanced Linear Baseline)**

In accordance with strict machine learning governance, evaluation was conducted exclusively on the held-out test partition ($N_{\text{test}} = 294$) containing 247 actual retained employees and 47 actual departed employees. All continuous preprocessing transformations (standardization, categorical encoding) were frozen and fitted strictly on the training partition in Task 4, guaranteeing zero test-set data leakage.

### Critical Assignment Evaluation Principles:
- **The Class Imbalance Mandate:** In an enterprise workforce where ~84% of employees remain, **raw accuracy is a dangerously misleading metric**. A trivial majority-class classifier achieves 84.01% accuracy while possessing 0% recall on departing personnel. 
- **The Primacy of Recall for Attrition (`Class 1`):** In human capital analytics, the organizational penalty of an unexpected departure (False Negative: recruiting fees, productivity loss, institutional memory drain) significantly exceeds the operational friction of an unnecessary retention check-in (False Positive). Models must be judged on their sensitivity (Recall) and discriminative capability (ROC-AUC), not raw accuracy alone.
- **Overfitting Diagnostics:** Evaluating training versus test metrics provides an empirical audit of algorithmic memorization and generalizability.

---

## 2. Standard Scikit-Learn Classification Reports (Individual Models)

Below are the Scikit-Learn standard classification reports for all four models evaluated on the held-out test dataset ($N = 294$). Each report is documented in standard format across Precision, Recall, F1-Score, and Support, accompanied by its serialized JSON artifact path and publication-grade image.

---

### 2.1 Model 1A: Decision Tree (Unconstrained Baseline)

- **Artifact JSON:** [classification_report_dt_unconstrained.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_dt_unconstrained.json)
- **Artifact Image:** [classification_report_dt_unconstrained.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_dt_unconstrained.png)

#### Standard Classification Report Table:
| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (0)** | 0.8771 | 0.8381 | 0.8571 | 247 |
| **Departed (1)** | 0.3103 | 0.3830 | 0.3429 | 47 |
| **accuracy** | — | — | **0.7653** | **294** |
| **macro avg** | 0.5937 | 0.6105 | 0.6000 | 294 |
| **weighted avg** | 0.7865 | 0.7653 | 0.7749 | 294 |

**Model Performance Summary:**
- **Test Accuracy:** $76.53\%$ ($225 / 294$ correct)
- **Training Accuracy:** $100.00\%$ ($1,176 / 1,176$ correct)
- **Generalization Gap:** $+23.47\%$ (Severe Overfitting / Memorization)
- **ROC-AUC Score:** $0.6105$
- **Minority Class Recall (`Attrition = Yes`):** $38.30\%$ ($18 / 47$ departed employees captured; $29$ missed)
- **Minority Class Precision:** $31.03\%$

![Decision Tree Unconstrained Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_dt_unconstrained.png)
*Figure 6.1: Standard Classification Report for Decision Tree (Unconstrained Baseline).*

---

### 2.2 Model 1B: Decision Tree (Regularized & Balanced)

- **Artifact JSON:** [classification_report_dt_regularized.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_dt_regularized.json)
- **Artifact Image:** [classification_report_dt_regularized.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_dt_regularized.png)

#### Standard Classification Report Table:
| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (0)** | 0.8947 | 0.7571 | 0.8202 | 247 |
| **Departed (1)** | 0.2941 | 0.5319 | 0.3788 | 47 |
| **accuracy** | — | — | **0.7211** | **294** |
| **macro avg** | 0.5944 | 0.6445 | 0.5995 | 294 |
| **weighted avg** | 0.7987 | 0.7211 | 0.7496 | 294 |

**Model Performance Summary:**
- **Test Accuracy:** $72.11\%$ ($212 / 294$ correct)
- **Training Accuracy:** $78.49\%$
- **Generalization Gap:** $+6.38\%$ (Well Regularized)
- **ROC-AUC Score:** $0.6885$
- **Minority Class Recall (`Attrition = Yes`):** $53.19\%$ ($25 / 47$ departed employees captured; $+14.89\%$ boost over unconstrained tree)
- **Minority Class Precision:** $29.41\%$

![Decision Tree Regularized Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_dt_regularized.png)
*Figure 6.2: Standard Classification Report for Decision Tree (Regularized & Balanced).*

---

### 2.3 Model 2: Random Forest (Balanced Ensemble)

- **Artifact JSON:** [classification_report_random_forest.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_random_forest.json)
- **Artifact Image:** [classification_report_random_forest.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_random_forest.png)

#### Standard Classification Report Table:
| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (0)** | 0.8619 | 0.9352 | 0.8971 | 247 |
| **Departed (1)** | 0.3846 | 0.2128 | 0.2740 | 47 |
| **accuracy** | — | — | **0.8197** | **294** |
| **macro avg** | 0.6233 | 0.5740 | 0.5855 | 294 |
| **weighted avg** | 0.7856 | 0.8197 | 0.7975 | 294 |

**Model Performance Summary:**
- **Test Accuracy:** $81.97\%$ ($241 / 294$ correct — Highest Overall Accuracy)
- **Training Accuracy:** $97.62\%$
- **Generalization Gap:** $+15.65\%$
- **ROC-AUC Score:** $0.7787$
- **Minority Class Recall (`Attrition = Yes`):** $21.28\%$ ($10 / 47$ departed employees captured; $37$ missed)
- **Minority Class Precision:** $38.46\%$ (Highest Precision among tree models)

![Random Forest Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_random_forest.png)
*Figure 6.3: Standard Classification Report for Random Forest (Balanced Ensemble).*

---

### 2.4 Model 3: Logistic Regression (Balanced Baseline)

- **Artifact JSON:** [classification_report_logistic_regression.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/classification_report_logistic_regression.json)
- **Artifact Image:** [classification_report_logistic_regression.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_logistic_regression.png)

#### Standard Classification Report Table:
| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (0)** | 0.9227 | 0.7733 | 0.8414 | 247 |
| **Departed (1)** | 0.3563 | 0.6596 | 0.4627 | 47 |
| **accuracy** | — | — | **0.7551** | **294** |
| **macro avg** | 0.6395 | 0.7164 | 0.6520 | 294 |
| **weighted avg** | 0.8322 | 0.7551 | 0.7809 | 294 |

**Model Performance Summary:**
- **Test Accuracy:** $75.51\%$ ($222 / 294$ correct)
- **Training Accuracy:** $77.98\%$
- **Generalization Gap:** $+2.47\%$ (Exceptional Generalization Stability)
- **ROC-AUC Score:** $0.8034$ (Highest Overall Discriminative Capacity)
- **Minority Class Recall (`Attrition = Yes`):** $65.96\%$ ($31 / 47$ departed employees captured — Highest Recall)
- **Minority Class Precision:** $35.63\%$
- **Minority Class F1-Score:** $0.4627$ (Highest Positive Class Harmonic Balance)

![Logistic Regression Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/classification_report_logistic_regression.png)
*Figure 6.4: Standard Classification Report for Logistic Regression (Balanced Baseline).*

---

## 3. Combined Classification Report for All Models (Standard Format)

- **Artifact JSON:** [combined_classification_report_all_models.json](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/classification_reports/combined_classification_report_all_models.json)
- **Artifact Image:** [combined_classification_report_all_models.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/combined_classification_report_all_models.png)

This section presents the **combined standard classification report** synthesizing all models evaluated within the AttritionIQ system.

### Combined Standard Classification Matrix:

| Model Architecture | Class / Metric | Precision | Recall | F1-Score | Support | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Unconstrained)** | Retained (0)<br>Departed (1)<br>**accuracy**<br>macro avg<br>weighted avg | 0.8771<br>0.3103<br>—<br>0.5937<br>0.7865 | 0.8381<br>0.3830<br>—<br>0.6105<br>0.7653 | 0.8571<br>0.3429<br>**0.7653**<br>0.6000<br>0.7749 | 247<br>47<br>**294**<br>294<br>294 | **0.6105** |
| **Decision Tree (Regularized & Balanced)** | Retained (0)<br>Departed (1)<br>**accuracy**<br>macro avg<br>weighted avg | 0.8947<br>0.2941<br>—<br>0.5944<br>0.7987 | 0.7571<br>0.5319<br>—<br>0.6445<br>0.7211 | 0.8202<br>0.3788<br>**0.7211**<br>0.5995<br>0.7496 | 247<br>47<br>**294**<br>294<br>294 | **0.6885** |
| **Random Forest (Balanced Ensemble)** | Retained (0)<br>Departed (1)<br>**accuracy**<br>macro avg<br>weighted avg | 0.8619<br>0.3846<br>—<br>0.6233<br>0.7856 | 0.9352<br>0.2128<br>—<br>0.5740<br>0.8197 | 0.8971<br>0.2740<br>**0.8197**<br>0.5855<br>0.7975 | 247<br>47<br>**294**<br>294<br>294 | **0.7787** |
| **Logistic Regression (Balanced Baseline)** | Retained (0)<br>Departed (1)<br>**accuracy**<br>macro avg<br>weighted avg | 0.9227<br>0.3563<br>—<br>0.6395<br>0.8322 | 0.7733<br>0.6596<br>—<br>0.7164<br>0.7551 | 0.8414<br>0.4627<br>**0.7551**<br>0.6520<br>0.7809 | 247<br>47<br>**294**<br>294<br>294 | **0.8034** |

![Combined Standard Classification Report](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/combined_classification_report_all_models.png)
*Figure 6.5: Combined standard classification report rendering all four model architectures in standard tabular layout.*

---

## 4. Master Model Performance Comparison Table

The comprehensive model benchmarking table below integrates all evaluation metrics mandated by the assignment specification (`REQ-23` through `REQ-27`):

| Model Name | Train Accuracy | Test Accuracy | Overfitting Gap ($\Delta$) | Recall (`Attrition=Yes`) | Precision (`Attrition=Yes`) | F1-Score (`Attrition=Yes`) | ROC-AUC Score | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Unconstrained)** | $100.00\%$ | $76.53\%$ | $+23.47\%$ | $38.30\%$ | $31.03\%$ | $0.3429$ | $0.6105$ | 18 | 29 | 40 | 207 |
| **Decision Tree (Regularized)** | $78.49\%$ | $72.11\%$ | $+6.38\%$ | $53.19\%$ | $29.41\%$ | $0.3788$ | $0.6885$ | 25 | 22 | 60 | 187 |
| **Random Forest (Balanced)** | $97.62\%$ | $81.97\%$ | $+15.65\%$ | $21.28\%$ | $38.46\%$ | $0.2740$ | $0.7787$ | 10 | 37 | 16 | 231 |
| **Logistic Regression (Balanced)** | $77.98\%$ | $75.51\%$ | **$+2.47\%$** | **$65.96\%$** | $35.63\%$ | **$0.4627$** | **$0.8034$** | **31** | **16** | 56 | 191 |

---

## 5. Confusion Matrix Analysis and Error Decomposition

- **Artifact Image:** [confusion_matrices_all_models.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/confusion_matrices_all_models.png)

The 2x2 confusion matrix breaks down actual versus predicted outcomes on the 294 test instances:

```
                                  PREDICTED
                         Retained (0)    Departed (1)
               ┌───────────────────────┬───────────────────────┐
  Retained (0) │  True Negative (TN)   │  False Positive (FP)  │
ACTUAL         ├───────────────────────┼───────────────────────┤
  Departed (1) │  False Negative (FN)  │  True Positive (TP)   │
               └───────────────────────┴───────────────────────┘
```

![Confusion Matrix Breakdown](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/confusion_matrices_all_models.png)
*Figure 6.6: Confusion matrix heatmaps for all four evaluated models with absolute counts and percentages.*

### Detailed Error Analysis:
1. **The Cost of False Negatives (FN):**
   - **Random Forest:** Incurred **37 False Negatives** ($78.72\%$ of all departed employees were missed). Because the uncalibrated probability threshold ($\tau = 0.5$) combined with ensemble averaging suppressed minority probabilities, the forest frequently predicted the majority class.
   - **Logistic Regression:** Achieved the lowest False Negative count at **16 FN** ($65.96\%$ recall). It successfully flagged 31 out of 47 departing employees for proactive retention discussions.
2. **The Burden of False Positives (FP):**
   - **Decision Tree Regularized (60 FP):** Flagged 60 retained employees as departure risks ($24.3\%$ false alarm rate among retained staff).
   - **Logistic Regression (56 FP):** Produced 56 false positives. In organizational practice, this translates to conducting ~56 check-in or engagement surveys with employees who were not planning to leave. This is acceptable given the low financial cost of retention conversations compared to replacing high-performing talent.

---

## 6. Overfitting and Generalization Gap Diagnosis

- **Artifact Image:** [train_vs_test_overfitting_comparison.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/train_vs_test_overfitting_comparison.png)

Comparing training set performance against out-of-sample test performance is essential for diagnosing model variance:

![Overfitting Diagnosis Comparison](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/train_vs_test_overfitting_comparison.png)
*Figure 6.7: Empirical generalization gaps for overall accuracy and minority-class F1-score across all four models.*

### Empirical Findings:
1. **Decision Tree (Unconstrained):** Demonstrates **severe memorization**. Train Accuracy reached $100.00\%$ ($Train\ F1 = 1.0000$), while Test Accuracy collapsed to $76.53\%$ ($Test\ F1 = 0.3429$), creating an accuracy gap of **$+23.47\%$** and an F1 gap of **$+65.71\%$**. This confirms that unconstrained trees cannot generalize without pruning.
2. **Decision Tree (Regularized):** Pruning (`max_depth=4`, `min_samples_leaf=5`) reduced the generalization gap to **$+6.38\%$**, demonstrating effective control of model variance.
3. **Random Forest:** Bagging mitigated variance compared to the unconstrained tree, but an accuracy gap of **$+15.65\%$** ($97.62\%$ train vs $81.97\%$ test) confirms that deeper ensemble trees still overfit without explicit leaf sample constraints.
4. **Logistic Regression:** Exhibited the smallest generalization gap (**$+2.47\%$** in accuracy and $+6.77\%$ in F1), proving that regularized linear decision boundaries provide consistent out-of-sample stability.

---

## 7. Receiver Operating Characteristic (ROC) and Area Under Curve (AUC) Analysis

- **Artifact Image:** [overlaid_roc_curves.png](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/overlaid_roc_curves.png)

The Receiver Operating Characteristic (ROC) curve evaluates classification performance across all possible discrimination thresholds $\tau \in [0, 1]$ by plotting the True Positive Rate (Sensitivity) against the False Positive Rate ($1 - \text{Specificity}$).

![Overlaid ROC Curves](file:///c:/Users/mihir/Desktop/AttritionIQ%20%E2%80%94%20Intelligent%20Employee%20Attrition%20Prediction%20System/reports/figures/overlaid_roc_curves.png)
*Figure 6.8: Overlaid ROC Curves benchmarking all models against the random-chance reference line.*

### ROC Analysis:
- **Logistic Regression (AUC = 0.8034):** Achieved the highest overall area under the curve, confirming strong discriminative capacity across probability thresholds.
- **Random Forest (AUC = 0.7787):** Demonstrates strong ranking ability, indicating that with threshold tuning (Task 8), its operational recall can be substantially improved.
- **Decision Tree Regularized (AUC = 0.6885):** Discrete partitions restrict the curve to a step-wise profile, but it outperforms the unconstrained tree.
- **Decision Tree Unconstrained (AUC = 0.6105):** Poor out-of-sample ranking capacity, only marginally better than random guessing ($0.5000$).

---

## 8. Strategic Business Insights & Algorithmic Trade-offs

From an enterprise human resources perspective, the empirical findings illustrate key trade-offs:

1. **The Fallacy of High Accuracy:** Random Forest achieved the highest test accuracy ($81.97\%$), but missed **$78.7\%$ of actual departures** (Recall = $21.28\%$). Relying on this model in production would leave leadership blind to nearly four out of five resignations.
2. **The Value of Balanced Baselines:** Logistic Regression with balanced weighting achieved the highest recall ($65.96\%$) and highest ROC-AUC ($0.8034$). It successfully detected 31 departing employees while maintaining an acceptable False Positive rate ($22.6\%$).
3. **Threshold Calibration Need:** The ensemble forest contains valuable ranking signals (AUC = $0.7787$), but requires threshold calibration (adjusting $\tau$ from $0.50$ down to $\approx 0.30$) or hyperparameter optimization (Task 8) to convert ranking ability into higher positive recall.

---

## 9. Assignment Compliance Matrix

This matrix verifies that all requirements for **Task 6** from `Machine Learning Assignment.docx` have been satisfied:

| Requirement ID | Prescribed Specification | Implementation Status | Empirical Verification Evidence |
| :--- | :--- | :---: | :--- |
| **REQ-23** | Evaluate Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC | **COMPLETED** | All 6 metrics computed for all 4 models and documented in Section 4. Standard classification reports generated in text, JSON, and PNG formats. |
| **REQ-24** | Compare Train vs. Test metrics to assess overfitting/underfitting | **COMPLETED** | Generalization gaps calculated and visualized in Figure 6.7. Unconstrained tree confirmed severe overfitting ($+23.47\%$ gap). |
| **REQ-25** | Highlight recall for Attrition=Yes and the false-positive trade-off | **COMPLETED** | Detailed operational analysis of minority class recall vs FP burden provided in Sections 2, 4, and 5. |
| **REQ-26** | Plot overlaid ROC curves on a single chart | **COMPLETED** | Overlaid ROC curve generated and saved to `reports/figures/overlaid_roc_curves.png` (Figure 6.8). |
| **REQ-27** | Construct comprehensive model comparison table (Decision Tree vs Random Forest) | **COMPLETED** | Complete comparative table delivered in Section 4 benchmarking all algorithms across 11 empirical metrics. |

---

## 10. Downstream Handoff

The empirical findings from Task 6 establish that while Logistic Regression serves as a strong linear benchmark, the tree ensemble retains strong ranking potential (AUC = $0.7787$). 
- In **Task 7**, we will analyze feature importances (MDI vs. Permutation Importance) to interpret which organizational drivers govern these predictions.
- In **Task 8**, we will execute stratified hyperparameter optimization (`GridSearchCV`/`RandomizedSearchCV`) to optimize the Random Forest threshold and tree hyperparameters to achieve higher recall on departing personnel.
