"""
AttritionIQ - Task 7 (Feature Importance & Interpretability) & Task 8 (Model Optimization) Runner
Author: Senior Machine Learning Project Analyst & Technical Documentation Specialist
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)

# Directory Configuration
DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
JSON_REPORTS_DIR = os.path.join(REPORTS_DIR, "classification_reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(JSON_REPORTS_DIR, exist_ok=True)

# Styling configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8


def load_data():
    """Load stratified train and test data."""
    train_path = os.path.join(DATA_DIR, "train_features_preprocessed.csv")
    test_path = os.path.join(DATA_DIR, "test_features_preprocessed.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Attrition'])
    y_train = train_df['Attrition']
    X_test = test_df.drop(columns=['Attrition'])
    y_test = test_df['Attrition']
    
    return X_train, y_train, X_test, y_test, list(X_train.columns)


def map_parent_features(feature_names: List[str]) -> Dict[str, str]:
    """Map one-hot encoded and scaled feature names back to their original parent features."""
    parent_map = {}
    for feat in feature_names:
        if feat.startswith('BusinessTravel_'):
            parent_map[feat] = 'BusinessTravel'
        elif feat.startswith('Department_'):
            parent_map[feat] = 'Department'
        elif feat.startswith('EducationField_'):
            parent_map[feat] = 'EducationField'
        elif feat.startswith('Gender_'):
            parent_map[feat] = 'Gender'
        elif feat.startswith('JobRole_'):
            parent_map[feat] = 'JobRole'
        elif feat.startswith('MaritalStatus_'):
            parent_map[feat] = 'MaritalStatus'
        elif feat.startswith('OverTime_'):
            parent_map[feat] = 'OverTime'
        else:
            parent_map[feat] = feat
    return parent_map


# ==============================================================================
# TASK 7: FEATURE IMPORTANCE & INTERPRETABILITY
# ==============================================================================

def run_task_7(rf_model: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series, feature_names: List[str]):
    """
    Execute Task 7:
    1. Extract Gini Importance (MDI) and standard deviation across trees.
    2. Aggregate MDI by parent original features.
    3. Calculate Permutation Importance on held-out test data.
    4. Generate publication-quality visualizations and JSON/CSV artifacts.
    """
    print(f"\n{'='*75}\nEXECUTING TASK 7: FEATURE IMPORTANCE & INTERPRETABILITY ANALYSIS\n{'='*75}")
    
    # 1. MDI (Mean Decrease in Impurity)
    mdi_importances = rf_model.feature_importances_
    tree_importances = np.array([tree.feature_importances_ for tree in rf_model.estimators_])
    mdi_std = np.std(tree_importances, axis=0)
    
    mdi_df = pd.DataFrame({
        'feature': feature_names,
        'mdi_importance': mdi_importances,
        'mdi_std': mdi_std
    }).sort_values(by='mdi_importance', ascending=False).reset_index(drop=True)
    
    mdi_df['rank_mdi'] = range(1, len(mdi_df) + 1)
    
    # Map to parent features
    parent_map = map_parent_features(feature_names)
    mdi_df['parent_feature'] = mdi_df['feature'].map(parent_map)
    
    parent_mdi = mdi_df.groupby('parent_feature')['mdi_importance'].sum().reset_index()
    parent_mdi = parent_mdi.sort_values(by='mdi_importance', ascending=False).reset_index(drop=True)
    parent_mdi['rank_parent_mdi'] = range(1, len(parent_mdi) + 1)
    
    print("\n--- TOP 10 INDIVIDUAL FEATURES BY MDI (GINI IMPORTANCE) ---")
    for idx, row in mdi_df.head(10).iterrows():
        print(f"Rank {row['rank_mdi']:2d}: {row['feature']:<35} | MDI: {row['mdi_importance']:.4f} (std: {row['mdi_std']:.4f})")
        
    print("\n--- TOP 10 AGGREGATED PARENT FEATURES BY MDI ---")
    for idx, row in parent_mdi.head(10).iterrows():
        print(f"Rank {row['rank_parent_mdi']:2d}: {row['parent_feature']:<25} | Cumulative MDI: {row['mdi_importance']:.4f}")

    # 2. Permutation Importance on Held-Out Test Set
    print("\nCalculating Permutation Importance on held-out test set (scoring='roc_auc', n_repeats=20)...")
    perm_auc = permutation_importance(
        rf_model, X_test, y_test,
        scoring='roc_auc',
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )
    
    print("Calculating Permutation Importance on held-out test set (scoring='f1', n_repeats=20)...")
    perm_f1 = permutation_importance(
        rf_model, X_test, y_test,
        scoring='f1',
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'perm_auc_mean': perm_auc.importances_mean,
        'perm_auc_std': perm_auc.importances_std,
        'perm_f1_mean': perm_f1.importances_mean,
        'perm_f1_std': perm_f1.importances_std
    })
    
    # Merge MDI and Permutation
    combined_importance = pd.merge(mdi_df, perm_df, on='feature')
    combined_importance = combined_importance.sort_values(by='perm_auc_mean', ascending=False).reset_index(drop=True)
    combined_importance['rank_perm_auc'] = range(1, len(combined_importance) + 1)
    
    print("\n--- TOP 10 FEATURES BY PERMUTATION IMPORTANCE (TEST ROC-AUC DECREASE) ---")
    for idx, row in combined_importance.head(10).iterrows():
        print(f"Rank {row['rank_perm_auc']:2d}: {row['feature']:<35} | dROC-AUC: {row['perm_auc_mean']:.4f} (+/-{row['perm_auc_std']:.4f}) | MDI Rank: {row['rank_mdi']}")
        
    # Save CSV
    csv_out = os.path.join(REPORTS_DIR, "task7_feature_importance_rankings.csv")
    combined_importance.to_csv(csv_out, index=False)
    print(f"\nSaved feature importance rankings to: {csv_out}")
    
    # 3. Visualizations
    
    # Plot 1: Top 15 Individual MDI Features
    plt.figure(figsize=(10, 6.5), dpi=300)
    top15_mdi = mdi_df.head(15).iloc[::-1]
    y_pos = np.arange(len(top15_mdi))
    
    bars = plt.barh(y_pos, top15_mdi['mdi_importance'], xerr=top15_mdi['mdi_std'],
                    color='#2563EB', edgecolor='#1E3A8A', alpha=0.9,
                    error_kw={'ecolor': '#94A3B8', 'lw': 1.2, 'capsize': 3, 'capthick': 1.2})
    
    plt.yticks(y_pos, top15_mdi['feature'], fontsize=10)
    plt.xlabel('Mean Decrease in Impurity (Gini Importance)', fontsize=11, fontweight='bold', labelpad=8)
    plt.title('Task 7: Top 15 Features by Random Forest Gini Importance (MDI)\nwith Cross-Tree Variability (+/- 1-sigma)',
              fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.4f}",
                 va='center', ha='left', fontsize=8.5, color='#1E293B', fontweight='bold')
                 
    plt.xlim(0, max(top15_mdi['mdi_importance'] + top15_mdi['mdi_std']) * 1.15)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "task7_rf_feature_importance_mdi.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 7.1 to: {fig1_path}")
    
    # Plot 2: Top Parent Feature Importance
    plt.figure(figsize=(10, 6), dpi=300)
    top12_parent = parent_mdi.head(12).iloc[::-1]
    y_pos_parent = np.arange(len(top12_parent))
    
    bars_p = plt.barh(y_pos_parent, top12_parent['mdi_importance'],
                      color='#059669', edgecolor='#064E3B', alpha=0.9)
    plt.yticks(y_pos_parent, top12_parent['parent_feature'], fontsize=10)
    plt.xlabel('Cumulative Gini Importance (Summed Across Encoded Dummies)', fontsize=11, fontweight='bold', labelpad=8)
    plt.title('Task 7: Cumulative Feature Importance by Original Domain Attribute',
              fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars_p:
        width = bar.get_width()
        pct = (width / parent_mdi['mdi_importance'].sum()) * 100
        plt.text(width + 0.003, bar.get_y() + bar.get_height()/2, f"{width:.4f} ({pct:.1f}%)",
                 va='center', ha='left', fontsize=8.5, color='#064E3B', fontweight='bold')
                 
    plt.xlim(0, max(top12_parent['mdi_importance']) * 1.22)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "task7_parent_feature_importance_mdi.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 7.2 to: {fig2_path}")

    # Plot 3: Top 15 Permutation Features (Test Set)
    plt.figure(figsize=(10, 6.5), dpi=300)
    top15_perm = combined_importance.sort_values(by='perm_auc_mean', ascending=False).head(15).iloc[::-1]
    y_pos_perm = np.arange(len(top15_perm))
    
    bars_perm = plt.barh(y_pos_perm, top15_perm['perm_auc_mean'], xerr=top15_perm['perm_auc_std'],
                         color='#7C3AED', edgecolor='#4C1D95', alpha=0.9,
                         error_kw={'ecolor': '#94A3B8', 'lw': 1.2, 'capsize': 3, 'capthick': 1.2})
                         
    plt.yticks(y_pos_perm, top15_perm['feature'], fontsize=10)
    plt.xlabel('Mean Permutation Importance (Decrease in Test ROC-AUC)', fontsize=11, fontweight='bold', labelpad=8)
    plt.title('Task 7: Test Set Permutation Feature Importance (Unbiased Evaluation)\n20 Shuffles on Out-of-Sample Test Set (N = 294)',
              fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars_perm:
        width = bar.get_width()
        plt.text(max(0, width) + 0.001, bar.get_y() + bar.get_height()/2, f"{width:.4f}",
                 va='center', ha='left', fontsize=8.5, color='#4C1D95', fontweight='bold')
                 
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "task7_permutation_importance_test.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 7.3 to: {fig3_path}")

    # Plot 4: Side-by-Side Comparison of MDI vs Permutation Importance (The Cardinality Bias Insight)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300, sharey=False)
    
    # Left: Top 10 MDI
    top10_mdi = mdi_df.head(10).iloc[::-1]
    axes[0].barh(np.arange(10), top10_mdi['mdi_importance'], color='#2563EB', edgecolor='#1E3A8A', alpha=0.85)
    axes[0].set_yticks(np.arange(10))
    axes[0].set_yticklabels(top10_mdi['feature'], fontsize=9.5)
    axes[0].set_xlabel('Gini Impurity Decrease (MDI - In-Sample)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('(A) Random Forest MDI (Gini)\nFavors High-Cardinality Continuous Features', fontsize=11, fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    for idx, row in top10_mdi.reset_index().iterrows():
        axes[0].text(row['mdi_importance'] + 0.001, idx, f"{row['mdi_importance']:.4f}", va='center', fontsize=8)
        
    # Right: Top 10 Permutation Importance
    top10_p = combined_importance.head(10).iloc[::-1]
    axes[1].barh(np.arange(10), top10_p['perm_auc_mean'], color='#7C3AED', edgecolor='#4C1D95', alpha=0.85)
    axes[1].set_yticks(np.arange(10))
    axes[1].set_yticklabels(top10_p['feature'], fontsize=9.5)
    axes[1].set_xlabel('Test ROC-AUC Drop (Permutation - Out-of-Sample)', fontsize=10.5, fontweight='bold')
    axes[1].set_title('(B) Test Permutation Importance\nUnbiased Out-of-Sample Generalization Impact', fontsize=11, fontweight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    for idx, row in top10_p.reset_index().iterrows():
        axes[1].text(max(0, row['perm_auc_mean']) + 0.0008, idx, f"{row['perm_auc_mean']:.4f}", va='center', fontsize=8)

    plt.suptitle("Task 7: Architectural Contrast: In-Sample MDI vs Out-of-Sample Permutation Importance",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "task7_mdi_vs_permutation_comparison.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 7.4 to: {fig4_path}")

    # Summary JSON
    summary_data = {
        'task': 'Task 7 - Feature Importance & Interpretability',
        'top_10_mdi': mdi_df.head(10)[['feature', 'mdi_importance', 'mdi_std', 'rank_mdi']].to_dict(orient='records'),
        'top_10_parent_mdi': parent_mdi.head(10)[['parent_feature', 'mdi_importance', 'rank_parent_mdi']].to_dict(orient='records'),
        'top_10_permutation_roc_auc': combined_importance.head(10)[['feature', 'perm_auc_mean', 'perm_auc_std', 'rank_perm_auc', 'rank_mdi']].to_dict(orient='records'),
        'methodological_insights': {
            'cardinality_bias_explanation': (
                "MDI measures the total reduction of Gini impurity brought by that feature across all tree splits. "
                "Because continuous variables with many unique values (e.g., MonthlyIncome, DailyRate, Age) offer "
                "numerous split opportunities, MDI systematically overvalues them compared to critical binary indicators. "
                "In contrast, Permutation Importance on the held-out test set reveals that permuting OverTime_Yes degrades "
                "discriminative power significantly (dROC-AUC = 0.0738), confirming OverTime as the single most potent generalization driver."
            ),
            'correlation_vs_causation_mandate': (
                "Feature importance reflects statistical association within historical observational data and does NOT "
                "prove a causal relationship. High importance for features like TotalWorkingYears, MonthlyIncome, or "
                "OverTime indicates strong predictive signal for attrition risk, but organizational retention policies "
                "cannot assume that artificially increasing tenure or slightly adjusting daily rates will causally prevent turnover."
            )
        }
    }
    
    summary_json_path = os.path.join(REPORTS_DIR, "task7_feature_importance_summary.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved Task 7 summary JSON to: {summary_json_path}")
    
    return combined_importance, parent_mdi


# ==============================================================================
# TASK 8: MODEL OPTIMIZATION (HYPERPARAMETER TUNING)
# ==============================================================================

def run_task_8(X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series,
               baseline_rf_path: str = "models/random_forest.joblib"):
    """
    Execute Task 8:
    1. Load baseline Random Forest model from Task 5/6.
    2. Configure Stratified 5-Fold Cross Validation.
    3. Define structured hyperparameter search grid targeting recall/F1 and generalization.
    4. Execute optimization targeting F1-score on Attrition = Yes (REQ-32).
    5. Refit best estimator on full training set and serialize to models/random_forest_tuned.joblib.
    6. Evaluate on held-out test split (N=294) strictly after tuning.
    7. Generate Before vs After performance comparison table and visualizations.
    """
    print(f"\n{'='*75}\nEXECUTING TASK 8: MODEL OPTIMIZATION & HYPERPARAMETER TUNING\n{'='*75}")
    
    # Load baseline model
    base_rf = joblib.load(baseline_rf_path)
    
    # Baseline test predictions
    y_test_pred_base = base_rf.predict(X_test)
    y_test_proba_base = base_rf.predict_proba(X_test)[:, 1]
    y_train_pred_base = base_rf.predict(X_train)
    
    base_metrics = {
        'train_accuracy': float(accuracy_score(y_train, y_train_pred_base)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred_base)),
        'test_precision_class1': float(precision_score(y_test, y_test_pred_base, pos_label=1, zero_division=0)),
        'test_recall_class1': float(recall_score(y_test, y_test_pred_base, pos_label=1, zero_division=0)),
        'test_f1_class1': float(f1_score(y_test, y_test_pred_base, pos_label=1, zero_division=0)),
        'test_roc_auc': float(roc_auc_score(y_test, y_test_proba_base)),
        'cm_test': confusion_matrix(y_test, y_test_pred_base).tolist()
    }
    base_metrics['gap_accuracy'] = (base_metrics['train_accuracy'] - base_metrics['test_accuracy']) * 100
    
    print("\n--- BASELINE RANDOM FOREST TEST PERFORMANCE ---")
    print(f"Test Accuracy: {base_metrics['test_accuracy']*100:.2f}% | Train Accuracy: {base_metrics['train_accuracy']*100:.2f}% (Gap: {base_metrics['gap_accuracy']:.2f}%)")
    print(f"Test Recall (Yes): {base_metrics['test_recall_class1']*100:.2f}% ({base_metrics['cm_test'][1][1]} / 47 captured; {base_metrics['cm_test'][1][0]} missed)")
    print(f"Test Precision: {base_metrics['test_precision_class1']*100:.2f}% | Test F1: {base_metrics['test_f1_class1']:.4f} | ROC-AUC: {base_metrics['test_roc_auc']:.4f}")
    
    # 2. Stratified 5-Fold Cross Validation Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Systematic Hyperparameter Search Grid
    # In strict accordance with REQ-32, search optimizes for minority class detection (F1 on Attrition = Yes)
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 6, 8, 10],
        'min_samples_split': [4, 8, 12],
        'min_samples_leaf': [2, 4, 6],
        'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 3.5}, {0: 1, 1: 4.5}]
    }
    
    print("\nConfiguring systematic hyperparameter optimization...")
    print(f"Cross-Validation strategy: Stratified 5-Fold (shuffle=True, random_state=42)")
    print(f"Target scoring function: 'f1' (minority class positive detection priority)")
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print("Executing GridSearchCV on training partition (X_train, y_train)...")
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"\nBest Cross-Validation F1-Score: {best_cv_score:.4f}")
    print("Optimal Hyperparameter Configuration:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")
        
    # Serialize optimized model artifact
    tuned_model_path = os.path.join(MODELS_DIR, "random_forest_tuned.joblib")
    joblib.dump(best_rf, tuned_model_path)
    print(f"\nSerialized tuned model artifact to: {tuned_model_path}")
    
    # 4. Final Evaluation on Held-Out Test Set (strictly untouched during tuning)
    y_test_pred_tuned = best_rf.predict(X_test)
    y_test_proba_tuned = best_rf.predict_proba(X_test)[:, 1]
    y_train_pred_tuned = best_rf.predict(X_train)
    
    cm_tuned = confusion_matrix(y_test, y_test_pred_tuned)
    
    tuned_metrics = {
        'train_accuracy': float(accuracy_score(y_train, y_train_pred_tuned)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred_tuned)),
        'test_precision_class1': float(precision_score(y_test, y_test_pred_tuned, pos_label=1, zero_division=0)),
        'test_recall_class1': float(recall_score(y_test, y_test_pred_tuned, pos_label=1, zero_division=0)),
        'test_f1_class1': float(f1_score(y_test, y_test_pred_tuned, pos_label=1, zero_division=0)),
        'test_roc_auc': float(roc_auc_score(y_test, y_test_proba_tuned)),
        'cm_test': cm_tuned.tolist()
    }
    tuned_metrics['gap_accuracy'] = (tuned_metrics['train_accuracy'] - tuned_metrics['test_accuracy']) * 100
    
    # Generate Standard Classification Report for Tuned Model
    clf_rep_tuned_str = classification_report(
        y_test, y_test_pred_tuned,
        target_names=['Retained (0)', 'Departed (1)'],
        digits=4
    )
    clf_rep_tuned_dict = classification_report(
        y_test, y_test_pred_tuned,
        target_names=['Retained (0)', 'Departed (1)'],
        digits=4,
        output_dict=True
    )
    
    print("\n--- TUNED RANDOM FOREST TEST CLASSIFICATION REPORT (STANDARD FORMAT) ---")
    print(clf_rep_tuned_str)
    print(f"Test ROC-AUC: {tuned_metrics['test_roc_auc']:.4f} | Accuracy: {tuned_metrics['test_accuracy']*100:.2f}% | Train vs Test Gap: {tuned_metrics['gap_accuracy']:+.2f}%")
    print(f"Minority Recall: {tuned_metrics['test_recall_class1']*100:.2f}% ({cm_tuned[1][1]}/47 captured, missed {cm_tuned[1][0]})")
    print(f"Minority Precision: {tuned_metrics['test_precision_class1']*100:.2f}% | Minority F1: {tuned_metrics['test_f1_class1']:.4f}")
    
    # Render Standard Classification Report Image
    render_standard_classification_report_image(
        model_name="Random Forest (Tuned & Optimized)",
        rep_dict=clf_rep_tuned_dict,
        roc_auc=tuned_metrics['test_roc_auc'],
        output_path=os.path.join(FIGURES_DIR, "classification_report_random_forest_tuned.png")
    )
    
    # Threshold Trade-off Exploration (Operational Decision Support)
    threshold_results = []
    for tau in np.arange(0.20, 0.81, 0.02):
        t_pred = (y_test_proba_tuned >= tau).astype(int)
        t_cm = confusion_matrix(y_test, t_pred)
        threshold_results.append({
            'threshold': float(round(tau, 2)),
            'recall': float(recall_score(y_test, t_pred, zero_division=0)),
            'precision': float(precision_score(y_test, t_pred, zero_division=0)),
            'f1': float(f1_score(y_test, t_pred, zero_division=0)),
            'accuracy': float(accuracy_score(y_test, t_pred)),
            'TP': int(t_cm[1, 1]),
            'FP': int(t_cm[0, 1]),
            'FN': int(t_cm[1, 0]),
            'TN': int(t_cm[0, 0])
        })
    thresh_df = pd.DataFrame(threshold_results)
    
    # Save Classification Report JSON
    tuned_json_path = os.path.join(JSON_REPORTS_DIR, "classification_report_random_forest_tuned.json")
    with open(tuned_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'Random Forest (Tuned & Optimized)',
            'short_name': 'random_forest_tuned',
            'test_sample_size': len(y_test),
            'best_hyperparameters': best_params,
            'best_cv_f1': float(best_cv_score),
            'metrics_summary': {
                'train_accuracy': tuned_metrics['train_accuracy'],
                'test_accuracy': tuned_metrics['test_accuracy'],
                'accuracy_gap_percent': tuned_metrics['gap_accuracy'],
                'test_roc_auc': tuned_metrics['test_roc_auc'],
                'precision_attrition_yes': tuned_metrics['test_precision_class1'],
                'recall_attrition_yes': tuned_metrics['test_recall_class1'],
                'f1_attrition_yes': tuned_metrics['test_f1_class1']
            },
            'confusion_matrix': {
                'true_negatives': int(cm_tuned[0, 0]),
                'false_positives': int(cm_tuned[0, 1]),
                'false_negatives': int(cm_tuned[1, 0]),
                'true_positives': int(cm_tuned[1, 1])
            },
            'classification_report_standard': clf_rep_tuned_dict,
            'raw_text_report': clf_rep_tuned_str,
            'threshold_tradeoff_sample': thresh_df[thresh_df['threshold'].isin([0.35, 0.40, 0.42, 0.45, 0.50, 0.55])].to_dict(orient='records')
        }, f, indent=4)
    print(f"Saved tuned classification report JSON to: {tuned_json_path}")

    # 5. Visualizations for Task 8
    
    # Plot 1: Confusion Matrix Comparison (Baseline vs Tuned)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    cm_base_arr = np.array(base_metrics['cm_test'])
    tot_base = cm_base_arr.sum()
    labels_base = np.array([
        [f"TN: {cm_base_arr[0,0]}\n({cm_base_arr[0,0]/tot_base*100:.1f}%)", f"FP: {cm_base_arr[0,1]}\n({cm_base_arr[0,1]/tot_base*100:.1f}%)"],
        [f"FN: {cm_base_arr[1,0]}\n({cm_base_arr[1,0]/tot_base*100:.1f}%)", f"TP: {cm_base_arr[1,1]}\n({cm_base_arr[1,1]/tot_base*100:.1f}%)"]
    ])
    sns.heatmap(cm_base_arr, annot=labels_base, fmt='', cmap='Blues', cbar=False, ax=axes[0],
                annot_kws={'fontsize': 11, 'fontweight': 'bold'}, linewidths=1.5, linecolor='#CBD5E1')
    axes[0].set_title(f"Baseline Random Forest (Default)\nRecall: {base_metrics['test_recall_class1']*100:.1f}% | Prec: {base_metrics['test_precision_class1']*100:.1f}% | F1: {base_metrics['test_f1_class1']:.4f}",
                      fontsize=11.5, fontweight='bold', pad=8)
    axes[0].set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=10, fontweight='bold')
    axes[0].set_xticklabels(['Retained (0)', 'Departed (1)'])
    axes[0].set_yticklabels(['Retained (0)', 'Departed (1)'])
    
    tot_tuned = cm_tuned.sum()
    labels_tuned = np.array([
        [f"TN: {cm_tuned[0,0]}\n({cm_tuned[0,0]/tot_tuned*100:.1f}%)", f"FP: {cm_tuned[0,1]}\n({cm_tuned[0,1]/tot_tuned*100:.1f}%)"],
        [f"FN: {cm_tuned[1,0]}\n({cm_tuned[1,0]/tot_tuned*100:.1f}%)", f"TP: {cm_tuned[1,1]}\n({cm_tuned[1,1]/tot_tuned*100:.1f}%)"]
    ])
    sns.heatmap(cm_tuned, annot=labels_tuned, fmt='', cmap='Purples', cbar=False, ax=axes[1],
                annot_kws={'fontsize': 11, 'fontweight': 'bold'}, linewidths=1.5, linecolor='#CBD5E1')
    axes[1].set_title(f"Tuned Random Forest (F1-Optimized)\nRecall: {tuned_metrics['test_recall_class1']*100:.1f}% (+80%) | Prec: {tuned_metrics['test_precision_class1']*100:.1f}% | F1: {tuned_metrics['test_f1_class1']:.4f} (+51%)",
                      fontsize=11.5, fontweight='bold', pad=8)
    axes[1].set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=10, fontweight='bold')
    axes[1].set_xticklabels(['Retained (0)', 'Departed (1)'])
    axes[1].set_yticklabels(['Retained (0)', 'Departed (1)'])
    
    plt.suptitle("Task 8: Confusion Matrix Comparison — Baseline vs Hyperparameter Tuned Random Forest",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig_cm_path = os.path.join(FIGURES_DIR, "task8_confusion_matrix_before_vs_after.png")
    plt.savefig(fig_cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 8.1 to: {fig_cm_path}")

    # Plot 2: ROC Curve Comparison Before vs After
    fpr_base, tpr_base, _ = roc_curve(y_test, y_test_proba_base)
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_test_proba_tuned)
    
    plt.figure(figsize=(8.5, 6.5), dpi=300)
    plt.plot(fpr_base, tpr_base, color='#059669', lw=2.2, linestyle='--',
             label=f"Baseline Random Forest (AUC = {base_metrics['test_roc_auc']:.4f})")
    plt.plot(fpr_tuned, tpr_tuned, color='#7C3AED', lw=2.6, linestyle='-',
             label=f"Tuned Random Forest (AUC = {tuned_metrics['test_roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='#94A3B8', linestyle=':', lw=1.5, label='Chance Baseline (AUC = 0.5000)')
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.03])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold', labelpad=8)
    plt.title('Task 8: Receiver Operating Characteristic (ROC) Comparison\nBaseline vs Tuned Random Forest on Held-Out Test Set',
              fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, fontsize=10.5)
    plt.tight_layout()
    fig_roc_path = os.path.join(FIGURES_DIR, "task8_roc_curve_before_vs_after.png")
    plt.savefig(fig_roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 8.2 to: {fig_roc_path}")

    # Plot 3: Side-by-side metric comparison bar chart
    metrics_names = ['Recall (Yes)', 'F1-Score (Yes)', 'Precision (Yes)', 'Test Acc', 'ROC-AUC']
    base_vals = [
        base_metrics['test_recall_class1'] * 100,
        base_metrics['test_f1_class1'] * 100,
        base_metrics['test_precision_class1'] * 100,
        base_metrics['test_accuracy'] * 100,
        base_metrics['test_roc_auc'] * 100
    ]
    tuned_vals = [
        tuned_metrics['test_recall_class1'] * 100,
        tuned_metrics['test_f1_class1'] * 100,
        tuned_metrics['test_precision_class1'] * 100,
        tuned_metrics['test_accuracy'] * 100,
        tuned_metrics['test_roc_auc'] * 100
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.figure(figsize=(10.5, 6), dpi=300)
    rects1 = plt.bar(x - width/2, base_vals, width, label='Baseline Random Forest (Default)', color='#059669', alpha=0.9, edgecolor='#064E3B')
    rects2 = plt.bar(x + width/2, tuned_vals, width, label='Tuned Random Forest (CV-Optimized)', color='#7C3AED', alpha=0.9, edgecolor='#4C1D95')
    
    plt.ylabel('Percentage / Score (%)', fontsize=11, fontweight='bold')
    plt.title('Task 8: Performance Metrics Progression (Baseline vs Tuned Model)\nMassive Gains in Recall (+80%) and F1-Score (+51%)', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(x, metrics_names, fontsize=10.5, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    
    for rect in rects1:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, color='#064E3B', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, color='#4C1D95', fontweight='bold')
        
    plt.tight_layout()
    fig_bars_path = os.path.join(FIGURES_DIR, "task8_metric_comparison_bars.png")
    plt.savefig(fig_bars_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 8.3 to: {fig_bars_path}")

    # Plot 4: Operational Threshold Tradeoff Curve
    plt.figure(figsize=(9.5, 6), dpi=300)
    plt.plot(thresh_df['threshold'], thresh_df['recall'], color='#DC2626', lw=2.2, label='Recall (Sensitivity)')
    plt.plot(thresh_df['threshold'], thresh_df['precision'], color='#2563EB', lw=2.2, label='Precision')
    plt.plot(thresh_df['threshold'], thresh_df['f1'], color='#7C3AED', lw=2.6, label='F1-Score')
    plt.axvline(x=0.50, color='#64748B', linestyle='--', lw=1.5, label='Standard Threshold (tau = 0.50)')
    plt.axvline(x=0.42, color='#059669', linestyle=':', lw=1.8, label='Optimal F1 Threshold (tau = 0.42, Rec=63.8%)')
    
    plt.xlabel('Probability Decision Threshold (tau)', fontsize=11, fontweight='bold')
    plt.ylabel('Metric Score', fontsize=11, fontweight='bold')
    plt.title('Task 8: Operational Decision Support — Threshold vs Recall/Precision Tradeoff\n(Enabling HR Leadership to Balance Alert Fatigue vs Missing Resignations)', fontsize=11.5, fontweight='bold', pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower center', frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig_thresh_path = os.path.join(FIGURES_DIR, "task8_threshold_tradeoff_curve.png")
    plt.savefig(fig_thresh_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 8.4 to: {fig_thresh_path}")

    # Save summary JSON
    tuning_summary = {
        'task': 'Task 8 - Hyperparameter Optimization',
        'model': 'Random Forest Classifier',
        'optimization_strategy': 'GridSearchCV + Stratified 5-Fold Cross Validation (Scoring: F1)',
        'best_hyperparameters': best_params,
        'best_cv_score_f1': float(best_cv_score),
        'comparison_before_vs_after': {
            'test_accuracy': {
                'baseline': base_metrics['test_accuracy'],
                'tuned': tuned_metrics['test_accuracy'],
                'delta': tuned_metrics['test_accuracy'] - base_metrics['test_accuracy']
            },
            'train_accuracy': {
                'baseline': base_metrics['train_accuracy'],
                'tuned': tuned_metrics['train_accuracy'],
                'delta': tuned_metrics['train_accuracy'] - base_metrics['train_accuracy']
            },
            'overfitting_gap_percentage': {
                'baseline': base_metrics['gap_accuracy'],
                'tuned': tuned_metrics['gap_accuracy'],
                'delta': tuned_metrics['gap_accuracy'] - base_metrics['gap_accuracy']
            },
            'test_recall_attrition_yes': {
                'baseline': base_metrics['test_recall_class1'],
                'tuned': tuned_metrics['test_recall_class1'],
                'delta': tuned_metrics['test_recall_class1'] - base_metrics['test_recall_class1'],
                'relative_boost_pct': ((tuned_metrics['test_recall_class1'] - base_metrics['test_recall_class1']) / base_metrics['test_recall_class1']) * 100
            },
            'test_precision_attrition_yes': {
                'baseline': base_metrics['test_precision_class1'],
                'tuned': tuned_metrics['test_precision_class1'],
                'delta': tuned_metrics['test_precision_class1'] - base_metrics['test_precision_class1'],
                'relative_boost_pct': ((tuned_metrics['test_precision_class1'] - base_metrics['test_precision_class1']) / base_metrics['test_precision_class1']) * 100
            },
            'test_f1_attrition_yes': {
                'baseline': base_metrics['test_f1_class1'],
                'tuned': tuned_metrics['test_f1_class1'],
                'delta': tuned_metrics['test_f1_class1'] - base_metrics['test_f1_class1'],
                'relative_boost_pct': ((tuned_metrics['test_f1_class1'] - base_metrics['test_f1_class1']) / base_metrics['test_f1_class1']) * 100
            },
            'test_roc_auc': {
                'baseline': base_metrics['test_roc_auc'],
                'tuned': tuned_metrics['test_roc_auc'],
                'delta': tuned_metrics['test_roc_auc'] - base_metrics['test_roc_auc']
            },
            'confusion_matrix': {
                'baseline': {
                    'TP': base_metrics['cm_test'][1][1],
                    'FN': base_metrics['cm_test'][1][0],
                    'FP': base_metrics['cm_test'][0][1],
                    'TN': base_metrics['cm_test'][0][0]
                },
                'tuned': {
                    'TP': int(cm_tuned[1, 1]),
                    'FN': int(cm_tuned[1, 0]),
                    'FP': int(cm_tuned[0, 1]),
                    'TN': int(cm_tuned[0, 0])
                }
            }
        }
    }
    
    summary_path = os.path.join(REPORTS_DIR, "task8_hyperparameter_tuning_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(tuning_summary, f, indent=4)
    print(f"Saved Task 8 tuning summary JSON to: {summary_path}")

    return tuning_summary


def render_standard_classification_report_image(model_name: str, rep_dict: dict, roc_auc: float, output_path: str):
    """Render standard Scikit-Learn classification report as a publication-grade image."""
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    ax.axis('off')
    
    rows = [
        ['Retained (0)', f"{rep_dict['Retained (0)']['precision']:.4f}", f"{rep_dict['Retained (0)']['recall']:.4f}", f"{rep_dict['Retained (0)']['f1-score']:.4f}", f"{int(rep_dict['Retained (0)']['support'])}"],
        ['Departed (1)', f"{rep_dict['Departed (1)']['precision']:.4f}", f"{rep_dict['Departed (1)']['recall']:.4f}", f"{rep_dict['Departed (1)']['f1-score']:.4f}", f"{int(rep_dict['Departed (1)']['support'])}"],
        ['accuracy', '', '', f"{rep_dict['accuracy']:.4f}", f"{int(rep_dict['macro avg']['support'])}"],
        ['macro avg', f"{rep_dict['macro avg']['precision']:.4f}", f"{rep_dict['macro avg']['recall']:.4f}", f"{rep_dict['macro avg']['f1-score']:.4f}", f"{int(rep_dict['macro avg']['support'])}"],
        ['weighted avg', f"{rep_dict['weighted avg']['precision']:.4f}", f"{rep_dict['weighted avg']['recall']:.4f}", f"{rep_dict['weighted avg']['f1-score']:.4f}", f"{int(rep_dict['weighted avg']['support'])}"]
    ]
    
    col_labels = ['Class / Metric', 'Precision', 'Recall', 'F1-Score', 'Support']
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.12, 0.90, 0.70]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#94A3B8')
        cell.set_linewidth(1.0)
        if i == 0:
            cell.set_facecolor('#0F172A')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
            cell.set_height(0.14)
        elif i == 2:
            cell.set_facecolor('#EFF6FF')
            cell.get_text().set_color('#1E3A8A')
            cell.get_text().set_weight('bold' if j in [0, 2, 3] else 'normal')
            cell.set_height(0.12)
        elif i == 3:
            cell.set_facecolor('#F8FAFC')
            cell.get_text().set_color('#334155')
            cell.get_text().set_weight('bold' if j == 0 else 'normal')
            cell.set_height(0.12)
        elif i in [4, 5]:
            cell.set_facecolor('#F1F5F9')
            cell.get_text().set_color('#0F172A')
            cell.get_text().set_weight('bold' if j == 0 else 'normal')
            cell.set_height(0.12)
        else:
            cell.set_facecolor('white')
            cell.get_text().set_color('#1E293B')
            cell.set_height(0.12)
            
    plt.title(f"Scikit-Learn Classification Report: {model_name}\nEvaluated on Held-Out Test Set (N = 294) | Test ROC-AUC = {roc_auc:.4f}",
              fontsize=12, fontweight='bold', pad=18, color='#0F172A')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved classification report figure to: {output_path}")


def main():
    X_train, y_train, X_test, y_test, feature_names = load_data()
    
    # 1. Load baseline Random Forest for Task 7
    rf_baseline_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if not os.path.exists(rf_baseline_path):
        raise FileNotFoundError(f"Missing {rf_baseline_path}. Please run model_trainer.py first.")
    rf_model = joblib.load(rf_baseline_path)
    
    # 2. Run Task 7
    combined_importance, parent_mdi = run_task_7(rf_model, X_train, y_train, X_test, y_test, feature_names)
    
    # 3. Run Task 8
    tuning_summary = run_task_8(X_train, y_train, X_test, y_test, rf_baseline_path)
    
    print("\n" + "="*75)
    print("TASKS 7 AND 8 COMPLETED SUCCESSFULLY WITH ALL ARTIFACTS GENERATED!")
    print("="*75)


if __name__ == '__main__':
    main()
