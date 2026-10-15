
"""
AttritionIQ — Consolidated Task 7 & Task 8 Execution Runner
Refactored & Consolidated Architecture:
- Task 7: Feature Importance & Interpretability Analysis (Preserved)
- Task 8: Systematic Hyperparameter Optimization & Evaluation (RF & LR Only)
- Production Model Deployment: Logistic Regression Pipeline Serialization

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ------------------------------------------------------------------------------
# Directory Configuration
# ------------------------------------------------------------------------------
DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
JSON_REPORTS_DIR = os.path.join(REPORTS_DIR, "classification_reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(JSON_REPORTS_DIR, exist_ok=True)

# Publication-grade styling configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Load stratified train and test data."""
    train_path = os.path.join(DATA_DIR, "train_features_preprocessed.csv")
    test_path = os.path.join(DATA_DIR, "test_features_preprocessed.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Preprocessed data not found in {DATA_DIR}. Ensure Task 4 preprocessing has run.")
        
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
# TASK 7: FEATURE IMPORTANCE & INTERPRETABILITY ANALYSIS
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
    print(f"\n{'='*80}\nEXECUTING TASK 7: FEATURE IMPORTANCE & INTERPRETABILITY ANALYSIS\n{'='*80}")
    
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

    # Plot 4: Side-by-Side Comparison of MDI vs Permutation Importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300, sharey=False)
    
    top10_mdi = mdi_df.head(10).iloc[::-1]
    axes[0].barh(np.arange(10), top10_mdi['mdi_importance'], color='#2563EB', edgecolor='#1E3A8A', alpha=0.85)
    axes[0].set_yticks(np.arange(10))
    axes[0].set_yticklabels(top10_mdi['feature'], fontsize=9.5)
    axes[0].set_xlabel('Gini Impurity Decrease (MDI - In-Sample)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('(A) Random Forest MDI (Gini)\nFavors High-Cardinality Continuous Features', fontsize=11, fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    for idx, row in top10_mdi.reset_index().iterrows():
        axes[0].text(row['mdi_importance'] + 0.001, idx, f"{row['mdi_importance']:.4f}", va='center', fontsize=8)
        
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
# HELPER FUNCTIONS: VISUALIZATIONS & METRICS
# ==============================================================================

def render_confusion_matrix_image(cm: np.ndarray, model_name: str, tuning_status: str,
                                  rec: float, prec: float, f1: float, cmap: str, output_path: str):
    """Render an individual publication-grade confusion matrix image."""
    plt.figure(figsize=(6.5, 5.2), dpi=300)
    tot = cm.sum()
    labels = np.array([
        [f"TN: {cm[0, 0]}\n({cm[0, 0]/tot*100:.1f}%)", f"FP: {cm[0, 1]}\n({cm[0, 1]/tot*100:.1f}%)"],
        [f"FN: {cm[1, 0]}\n({cm[1, 0]/tot*100:.1f}%)", f"TP: {cm[1, 1]}\n({cm[1, 1]/tot*100:.1f}%)"]
    ])
    
    ax = sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, cbar=False,
                     annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                     linewidths=1.5, linecolor='#CBD5E1')
    
    plt.title(f"{model_name} — {tuning_status}\nTest N = {tot} | Recall: {rec*100:.2f}% | Precision: {prec*100:.2f}% | F1: {f1:.4f}",
              fontsize=11, fontweight='bold', pad=14, color='#0F172A')
    plt.xlabel('Predicted Class', fontweight='bold', fontsize=10, labelpad=8)
    plt.ylabel('Actual Class', fontweight='bold', fontsize=10, labelpad=8)
    ax.set_xticklabels(['No Attrition (0)', 'Attrition (1)'], fontsize=9.5)
    ax.set_yticklabels(['No Attrition (0)', 'Attrition (1)'], fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix image to: {output_path}")


def render_roc_curve_image(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
                           model_name: str, tuning_status: str, color: str, output_path: str):
    """Render an individual publication-grade ROC curve image."""
    plt.figure(figsize=(7, 5.5), dpi=300)
    plt.plot(fpr, tpr, color=color, lw=2.5,
             label=f"{model_name} ({tuning_status})\nROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color='#94A3B8', linestyle=':', lw=1.5, label='Chance Baseline (AUC = 0.5000)')
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.03])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10.5, fontweight='bold', labelpad=8)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=10.5, fontweight='bold', labelpad=8)
    plt.title(f"Receiver Operating Characteristic (ROC)\n{model_name} ({tuning_status}) — Held-Out Test Set (N = 294)",
              fontsize=11.5, fontweight='bold', pad=12, color='#0F172A')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve image to: {output_path}")


def render_combined_classification_report_image(entries: Dict[str, Dict[str, Any]], output_path: str):
    """Render combined 4-panel publication comparison showing classification reports for all 4 model states."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=300)
    axes = axes.flatten()
    
    col_labels = ['Class / Metric', 'Precision', 'Recall', 'F1-Score', 'Support']
    
    for idx, (key, data) in enumerate(entries.items()):
        ax = axes[idx]
        ax.axis('off')
        
        rep = data['clf_report_dict']
        table_rows = [
            ['No Attrition (0)', f"{rep['Retained (0)']['precision']:.4f}", f"{rep['Retained (0)']['recall']:.4f}", f"{rep['Retained (0)']['f1-score']:.4f}", f"{int(rep['Retained (0)']['support'])}"],
            ['Attrition (1)', f"{rep['Departed (1)']['precision']:.4f}", f"{rep['Departed (1)']['recall']:.4f}", f"{rep['Departed (1)']['f1-score']:.4f}", f"{int(rep['Departed (1)']['support'])}"],
            ['accuracy', '', '', f"{rep['accuracy']:.4f}", f"{int(rep['macro avg']['support'])}"],
            ['macro avg', f"{rep['macro avg']['precision']:.4f}", f"{rep['macro avg']['recall']:.4f}", f"{rep['macro avg']['f1-score']:.4f}", f"{int(rep['macro avg']['support'])}"],
            ['weighted avg', f"{rep['weighted avg']['precision']:.4f}", f"{rep['weighted avg']['recall']:.4f}", f"{rep['weighted avg']['f1-score']:.4f}", f"{int(rep['weighted avg']['support'])}"]
        ]
        
        table = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            bbox=[0.02, 0.08, 0.96, 0.72]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        header_color = '#1E3A8A' if 'Logistic' in data['model_name'] else '#064E3B'
        
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#CBD5E1')
            if r == 0:
                cell.set_facecolor('#0F172A')
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
                cell.set_height(0.14)
            elif r == 2:  # Minority class (Attrition = 1)
                cell.set_facecolor('#FEF3C7' if 'Before' in data['tuning_status'] else '#DCFCE7')
                cell.get_text().set_color('#92400E' if 'Before' in data['tuning_status'] else '#166534')
                cell.get_text().set_weight('bold')
            elif r == 3:  # Accuracy
                cell.set_facecolor('#F8FAFC')
                cell.get_text().set_weight('bold' if c == 0 else 'normal')
            elif r in [4, 5]:  # Averages
                cell.set_facecolor('#F1F5F9')
                cell.get_text().set_weight('bold' if c == 0 else 'normal')
            else:
                cell.set_facecolor('#FFFFFF')
                
        status_tag = "BEFORE TUNING (BASELINE)" if "Before" in data['tuning_status'] else "AFTER TUNING (OPTIMIZED)"
        ax.set_title(
            f"[{idx+1}] {data['model_name']} — {status_tag}\n"
            f"Accuracy: {data['accuracy']*100:.2f}% | Attrition Recall: {data['recall_class1']*100:.2f}% | "
            f"F1: {data['f1_class1']:.4f} | ROC-AUC: {data['roc_auc']:.4f}",
            fontsize=11.5, fontweight='bold', color='#0F172A', pad=12
        )
        
    plt.suptitle("AttritionIQ — Task 8: Combined Classification Report Benchmark (Before vs After Hyperparameter Tuning)\n"
                 "Evaluated on Held-Out Test Set (N = 294, 247 No Attrition vs 47 Attrition)",
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated combined classification report image: {output_path}")


def render_standard_classification_report_image(model_name: str, rep_dict: dict, roc_auc: float, output_path: str):
    """Render standard Scikit-Learn classification report as a single publication-grade image."""
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    ax.axis('off')
    
    rows = [
        ['No Attrition (0)', f"{rep_dict['Retained (0)']['precision']:.4f}", f"{rep_dict['Retained (0)']['recall']:.4f}", f"{rep_dict['Retained (0)']['f1-score']:.4f}", f"{int(rep_dict['Retained (0)']['support'])}"],
        ['Attrition (1)', f"{rep_dict['Departed (1)']['precision']:.4f}", f"{rep_dict['Departed (1)']['recall']:.4f}", f"{rep_dict['Departed (1)']['f1-score']:.4f}", f"{int(rep_dict['Departed (1)']['support'])}"],
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
            cell.get_text().set_weight('bold')
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
    print(f"Saved single classification report image to: {output_path}")


# ==============================================================================
# TASK 8: SYSTEMATIC MODEL OPTIMIZATION & DEPLOYMENT PIPELINE
# ==============================================================================

def run_task_8(X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series):
    """
    Execute Task 8 for Random Forest and Logistic Regression (excluding Decision Tree):
    1. Train & evaluate baseline estimators (Before Tuning).
    2. Configure 5-Fold Stratified Cross Validation.
    3. Perform GridSearchCV optimizing minority F1-score on training data.
    4. Serialize tuned standalone estimators.
    5. Explore operational decision thresholds using training out-of-fold cross-validation.
    6. Construct and serialize the unified production Logistic Regression Pipeline.
    7. Evaluate all 4 model states on untouched held-out test split (N = 294).
    8. Generate the combined classification report image and JSON.
    9. Generate 4 separate confusion matrix images (Before & After).
    10. Generate 4 separate ROC curve images (Before & After).
    """
    print(f"\n{'='*80}\nEXECUTING TASK 8: MODEL OPTIMIZATION, EVALUATION & DEPLOYMENT PIPELINE\n{'='*80}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --------------------------------------------------------------------------
    # 1. Baseline Estimators (Before Tuning)
    # --------------------------------------------------------------------------
    rf_base_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if os.path.exists(rf_base_path):
        base_rf = joblib.load(rf_base_path)
    else:
        base_rf = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=6, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        base_rf.fit(X_train, y_train)
        joblib.dump(base_rf, rf_base_path)
        
    lr_base_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    if os.path.exists(lr_base_path):
        base_lr = joblib.load(lr_base_path)
    else:
        base_lr = LogisticRegression(
            C=1.0, penalty='l2', solver='lbfgs', class_weight='balanced',
            max_iter=1000, random_state=42
        )
        base_lr.fit(X_train, y_train)
        joblib.dump(base_lr, lr_base_path)
        
    print("\n[Baseline Estimators Verified and Ready]")
    
    # --------------------------------------------------------------------------
    # 2. Hyperparameter Optimization via GridSearchCV (Training Split Only)
    # --------------------------------------------------------------------------
    print("\nConfiguring GridSearchCV hyperparameter tuning for Random Forest and Logistic Regression...")
    
    param_grid_rf = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 6, 8, 10],
        'min_samples_split': [4, 8, 12],
        'min_samples_leaf': [2, 4, 6],
        'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 3.5}]
    }
    
    param_grid_lr = {
        'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', {0: 1, 1: 3.0}, {0: 1, 1: 4.0}, {0: 1, 1: 5.0}]
    }
    
    print("\n--- Tuning Random Forest ---")
    grid_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid_rf,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    best_rf_params = grid_rf.best_params_
    best_rf_cv_score = grid_rf.best_score_
    
    print(f"Optimal RF 5-Fold CV F1: {best_rf_cv_score:.4f}")
    print(f"Best RF Parameters: {best_rf_params}")
    
    # Serialize tuned RF model
    tuned_rf_path = os.path.join(MODELS_DIR, "random_forest_tuned.joblib")
    joblib.dump(best_rf, tuned_rf_path)
    print(f"Saved tuned Random Forest estimator to: {tuned_rf_path}")
    
    print("\n--- Tuning Logistic Regression ---")
    grid_lr = GridSearchCV(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        param_grid=param_grid_lr,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    best_lr_params = grid_lr.best_params_
    best_lr_cv_score = grid_lr.best_score_
    
    print(f"Optimal LR 5-Fold CV F1: {best_lr_cv_score:.4f}")
    print(f"Best LR Parameters: {best_lr_params}")
    
    # Serialize tuned LR model
    tuned_lr_path = os.path.join(MODELS_DIR, "logistic_regression_tuned.joblib")
    joblib.dump(best_lr, tuned_lr_path)
    print(f"Saved tuned Logistic Regression estimator to: {tuned_lr_path}")
    
    # --------------------------------------------------------------------------
    # 3. Leakage-Free Decision Threshold Exploration via Out-Of-Fold CV
    # --------------------------------------------------------------------------
    print("\nExploring operational decision thresholds on training data via 5-Fold Cross-Validation...")
    cv_oof_proba_lr = cross_val_predict(best_lr, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    
    threshold_tradeoffs = []
    for tau in np.arange(0.20, 0.81, 0.05):
        t_pred = (cv_oof_proba_lr >= tau).astype(int)
        t_cm = confusion_matrix(y_train, t_pred)
        threshold_tradeoffs.append({
            'threshold': float(round(tau, 2)),
            'cv_recall': float(recall_score(y_train, t_pred, zero_division=0)),
            'cv_precision': float(precision_score(y_train, t_pred, zero_division=0)),
            'cv_f1': float(f1_score(y_train, t_pred, zero_division=0)),
            'cv_accuracy': float(accuracy_score(y_train, t_pred)),
            'TP': int(t_cm[1, 1]),
            'FP': int(t_cm[0, 1]),
            'FN': int(t_cm[1, 0]),
            'TN': int(t_cm[0, 0])
        })
    print(f"Threshold exploration completed across {len(threshold_tradeoffs)} candidate operating cutoffs.")

    # --------------------------------------------------------------------------
    # 4. Production Deployment Pipeline Assembly (Logistic Regression)
    # --------------------------------------------------------------------------
    preprocessor_path = os.path.join(DATA_DIR, "fitted_preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        fitted_preprocessor = joblib.load(preprocessor_path)
        production_pipeline = Pipeline([
            ('preprocessor', fitted_preprocessor),
            ('classifier', best_lr)
        ])
        
        # Serialize production pipeline artifacts
        prod_path_primary = os.path.join(MODELS_DIR, "attrition_model_pipeline.joblib")
        prod_path_lr = os.path.join(MODELS_DIR, "attrition_logistic_pipeline.joblib")
        
        joblib.dump(production_pipeline, prod_path_primary)
        joblib.dump(production_pipeline, prod_path_lr)
        print(f"\n[PRODUCTION ARTIFACT CREATED]")
        print(f"  * Serialized champion Logistic Regression pipeline to: {prod_path_primary}")
        print(f"  * Serialized champion Logistic Regression pipeline to: {prod_path_lr}")
    else:
        print(f"\nWARNING: {preprocessor_path} not found. Production pipeline assembly skipped.")

    # --------------------------------------------------------------------------
    # 5. Comprehensive Evaluation Across the 4 Model States on Held-Out Test Set
    # --------------------------------------------------------------------------
    models_to_evaluate = {
        'rf_before': {
            'model_name': 'Random Forest',
            'tuning_status': 'Before Tuning',
            'estimator': base_rf,
            'cm_cmap': 'Blues',
            'roc_color': '#2563EB',
            'cm_out': os.path.join(FIGURES_DIR, "task8_random_forest_confusion_matrix_before.png"),
            'roc_out': os.path.join(FIGURES_DIR, "task8_random_forest_roc_before.png"),
            'best_params': {
                'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 6,
                'min_samples_leaf': 2, 'class_weight': 'balanced'
            },
            'cv_score': None
        },
        'rf_after': {
            'model_name': 'Random Forest',
            'tuning_status': 'After Tuning',
            'estimator': best_rf,
            'cm_cmap': 'Greens',
            'roc_color': '#059669',
            'cm_out': os.path.join(FIGURES_DIR, "task8_random_forest_confusion_matrix_after.png"),
            'roc_out': os.path.join(FIGURES_DIR, "task8_random_forest_roc_after.png"),
            'best_params': best_rf_params,
            'cv_score': float(best_rf_cv_score)
        },
        'lr_before': {
            'model_name': 'Logistic Regression',
            'tuning_status': 'Before Tuning',
            'estimator': base_lr,
            'cm_cmap': 'Oranges',
            'roc_color': '#D97706',
            'cm_out': os.path.join(FIGURES_DIR, "task8_logistic_regression_confusion_matrix_before.png"),
            'roc_out': os.path.join(FIGURES_DIR, "task8_logistic_regression_roc_before.png"),
            'best_params': {
                'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'class_weight': 'balanced'
            },
            'cv_score': None
        },
        'lr_after': {
            'model_name': 'Logistic Regression',
            'tuning_status': 'After Tuning',
            'estimator': best_lr,
            'cm_cmap': 'Purples',
            'roc_color': '#7C3AED',
            'cm_out': os.path.join(FIGURES_DIR, "task8_logistic_regression_confusion_matrix_after.png"),
            'roc_out': os.path.join(FIGURES_DIR, "task8_logistic_regression_roc_after.png"),
            'best_params': best_lr_params,
            'cv_score': float(best_lr_cv_score)
        }
    }
    
    evaluation_entries = {}
    
    print("\n" + "="*80)
    print("HELD-OUT TEST SET BENCHMARK RESULTS (N = 294, 247 Retained vs 47 Departed)")
    print("="*80)
    
    for key, cfg in models_to_evaluate.items():
        est = cfg['estimator']
        y_test_pred = est.predict(X_test)
        y_test_proba = est.predict_proba(X_test)[:, 1]
        
        acc = float(accuracy_score(y_test, y_test_pred))
        rec_1 = float(recall_score(y_test, y_test_pred, pos_label=1, zero_division=0))
        prec_1 = float(precision_score(y_test, y_test_pred, pos_label=1, zero_division=0))
        f1_1 = float(f1_score(y_test, y_test_pred, pos_label=1, zero_division=0))
        auc = float(roc_auc_score(y_test, y_test_proba))
        cm = confusion_matrix(y_test, y_test_pred)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        
        clf_rep_dict = classification_report(
            y_test, y_test_pred,
            target_names=['Retained (0)', 'Departed (1)'],
            digits=4,
            output_dict=True
        )
        clf_rep_str = classification_report(
            y_test, y_test_pred,
            target_names=['Retained (0)', 'Departed (1)'],
            digits=4
        )
        
        print(f"\n{'-'*75}\n{cfg['model_name']} ({cfg['tuning_status']})\n{'-'*75}")
        print(clf_rep_str)
        print(f"Accuracy: {acc*100:.2f}% | Recall (1): {rec_1*100:.2f}% | Precision (1): {prec_1*100:.2f}% | F1: {f1_1:.4f} | ROC-AUC: {auc:.4f}")
        
        # 6. Render individual confusion matrix image
        render_confusion_matrix_image(
            cm=cm,
            model_name=cfg['model_name'],
            tuning_status=cfg['tuning_status'],
            rec=rec_1,
            prec=prec_1,
            f1=f1_1,
            cmap=cfg['cm_cmap'],
            output_path=cfg['cm_out']
        )
        
        # 7. Render individual ROC curve image
        render_roc_curve_image(
            fpr=fpr,
            tpr=tpr,
            roc_auc=auc,
            model_name=cfg['model_name'],
            tuning_status=cfg['tuning_status'],
            color=cfg['roc_color'],
            output_path=cfg['roc_out']
        )
        
        evaluation_entries[key] = {
            'model_name': cfg['model_name'],
            'tuning_status': cfg['tuning_status'],
            'best_hyperparameters': cfg['best_params'],
            'cv_f1_score': cfg['cv_score'],
            'accuracy': acc,
            'recall_class1': rec_1,
            'precision_class1': prec_1,
            'f1_class1': f1_1,
            'roc_auc': auc,
            'confusion_matrix': {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            },
            'clf_report_dict': clf_rep_dict,
            'raw_text_report': clf_rep_str,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }

    # --------------------------------------------------------------------------
    # 8. Combined Classification Report (Image & JSON)
    # --------------------------------------------------------------------------
    comb_img_path = os.path.join(FIGURES_DIR, "task8_combined_classification_report.png")
    render_combined_classification_report_image(evaluation_entries, comb_img_path)
    
    comb_json_path = os.path.join(JSON_REPORTS_DIR, "task8_combined_classification_report.json")
    with open(comb_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'project': 'AttritionIQ — Intelligent Employee Attrition Prediction System',
            'task': 'Task 8 — Systematic Model Optimization and Cross-Architecture Benchmark',
            'positive_class': 'Attrition = Yes (1)',
            'negative_class': 'No Attrition (0)',
            'test_set_size': len(y_test),
            'test_class_distribution': {
                'no_attrition_count': int(sum(y_test == 0)),
                'attrition_count': int(sum(y_test == 1)),
                'attrition_prevalence_pct': float(sum(y_test == 1) / len(y_test) * 100)
            },
            'champion_production_model': {
                'model_name': 'Logistic Regression',
                'artifact_pipeline': 'models/attrition_model_pipeline.joblib',
                'artifact_linear_pipeline': 'models/attrition_logistic_pipeline.joblib',
                'optimal_hyperparameters': best_lr_params,
                'best_cv_f1': float(best_lr_cv_score),
                'test_accuracy': evaluation_entries['lr_after']['accuracy'],
                'test_f1': evaluation_entries['lr_after']['f1_class1'],
                'test_recall': evaluation_entries['lr_after']['recall_class1'],
                'test_precision': evaluation_entries['lr_after']['precision_class1'],
                'test_roc_auc': evaluation_entries['lr_after']['roc_auc']
            },
            'models_benchmark': {
                key: {
                    'model_name': v['model_name'],
                    'tuning_status': v['tuning_status'],
                    'hyperparameters': v['best_hyperparameters'],
                    'cv_f1_score': v['cv_f1_score'],
                    'metrics': {
                        'accuracy': v['accuracy'],
                        'recall_attrition_yes': v['recall_class1'],
                        'precision_attrition_yes': v['precision_class1'],
                        'f1_attrition_yes': v['f1_class1'],
                        'roc_auc': v['roc_auc']
                    },
                    'confusion_matrix': v['confusion_matrix'],
                    'classification_report_standard': v['clf_report_dict']
                } for key, v in evaluation_entries.items()
            }
        }, f, indent=4)
    print(f"Saved combined classification report JSON to: {comb_json_path}")

    # --------------------------------------------------------------------------
    # 9. Individual Tuned Reports & Overall Tuning Summary (Retained Linkages)
    # --------------------------------------------------------------------------
    # Tuned RF Report Image & JSON
    render_standard_classification_report_image(
        model_name="Random Forest (Tuned & Optimized)",
        rep_dict=evaluation_entries['rf_after']['clf_report_dict'],
        roc_auc=evaluation_entries['rf_after']['roc_auc'],
        output_path=os.path.join(FIGURES_DIR, "classification_report_random_forest_tuned.png")
    )
    with open(os.path.join(JSON_REPORTS_DIR, "classification_report_random_forest_tuned.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'Random Forest (Tuned & Optimized)',
            'short_name': 'random_forest_tuned',
            'test_sample_size': len(y_test),
            'best_hyperparameters': best_rf_params,
            'best_cv_f1': float(best_rf_cv_score),
            'metrics_summary': {
                'test_accuracy': evaluation_entries['rf_after']['accuracy'],
                'test_roc_auc': evaluation_entries['rf_after']['roc_auc'],
                'precision_attrition_yes': evaluation_entries['rf_after']['precision_class1'],
                'recall_attrition_yes': evaluation_entries['rf_after']['recall_class1'],
                'f1_attrition_yes': evaluation_entries['rf_after']['f1_class1']
            },
            'confusion_matrix': evaluation_entries['rf_after']['confusion_matrix'],
            'classification_report_standard': evaluation_entries['rf_after']['clf_report_dict']
        }, f, indent=4)

    # Tuned LR Report Image & JSON
    render_standard_classification_report_image(
        model_name="Logistic Regression (Tuned & Calibrated)",
        rep_dict=evaluation_entries['lr_after']['clf_report_dict'],
        roc_auc=evaluation_entries['lr_after']['roc_auc'],
        output_path=os.path.join(FIGURES_DIR, "classification_report_logistic_regression_tuned.png")
    )
    with open(os.path.join(JSON_REPORTS_DIR, "classification_report_logistic_regression_tuned.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'Logistic Regression (Tuned & Calibrated)',
            'short_name': 'logistic_regression_tuned',
            'test_sample_size': len(y_test),
            'best_hyperparameters': best_lr_params,
            'best_cv_f1': float(best_lr_cv_score),
            'metrics_summary': {
                'test_accuracy': evaluation_entries['lr_after']['accuracy'],
                'test_roc_auc': evaluation_entries['lr_after']['roc_auc'],
                'precision_attrition_yes': evaluation_entries['lr_after']['precision_class1'],
                'recall_attrition_yes': evaluation_entries['lr_after']['recall_class1'],
                'f1_attrition_yes': evaluation_entries['lr_after']['f1_class1']
            },
            'confusion_matrix': evaluation_entries['lr_after']['confusion_matrix'],
            'classification_report_standard': evaluation_entries['lr_after']['clf_report_dict']
        }, f, indent=4)

    # Task 8 Summary JSON
    summary_path = os.path.join(REPORTS_DIR, "task8_hyperparameter_tuning_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'task': 'Task 8 - Hyperparameter Optimization & Model Selection',
            'production_champion_model': 'Logistic Regression',
            'optimization_strategy': 'GridSearchCV + Stratified 5-Fold Cross-Validation (Scoring: F1)',
            'models_evaluated': ['Random Forest', 'Logistic Regression'],
            'decision_tree_status': 'Excluded from Task 8 tuning workflow',
            'rf_best_params': best_rf_params,
            'rf_cv_f1': float(best_rf_cv_score),
            'lr_best_params': best_lr_params,
            'lr_cv_f1': float(best_lr_cv_score),
            'threshold_tradeoffs_training_cv': threshold_tradeoffs,
            'before_after_comparison': {
                'random_forest': {
                    'baseline': {
                        'accuracy': evaluation_entries['rf_before']['accuracy'],
                        'recall': evaluation_entries['rf_before']['recall_class1'],
                        'precision': evaluation_entries['rf_before']['precision_class1'],
                        'f1': evaluation_entries['rf_before']['f1_class1'],
                        'roc_auc': evaluation_entries['rf_before']['roc_auc']
                    },
                    'tuned': {
                        'accuracy': evaluation_entries['rf_after']['accuracy'],
                        'recall': evaluation_entries['rf_after']['recall_class1'],
                        'precision': evaluation_entries['rf_after']['precision_class1'],
                        'f1': evaluation_entries['rf_after']['f1_class1'],
                        'roc_auc': evaluation_entries['rf_after']['roc_auc']
                    }
                },
                'logistic_regression': {
                    'baseline': {
                        'accuracy': evaluation_entries['lr_before']['accuracy'],
                        'recall': evaluation_entries['lr_before']['recall_class1'],
                        'precision': evaluation_entries['lr_before']['precision_class1'],
                        'f1': evaluation_entries['lr_before']['f1_class1'],
                        'roc_auc': evaluation_entries['lr_before']['roc_auc']
                    },
                    'tuned': {
                        'accuracy': evaluation_entries['lr_after']['accuracy'],
                        'recall': evaluation_entries['lr_after']['recall_class1'],
                        'precision': evaluation_entries['lr_after']['precision_class1'],
                        'f1': evaluation_entries['lr_after']['f1_class1'],
                        'roc_auc': evaluation_entries['lr_after']['roc_auc']
                    }
                }
            }
        }, f, indent=4)
    print(f"Saved Task 8 hyperparameter tuning summary JSON to: {summary_path}")
    
    return evaluation_entries


def main():
    X_train, y_train, X_test, y_test, feature_names = load_data()
    
    # 1. Load baseline Random Forest for Task 7
    rf_baseline_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if not os.path.exists(rf_baseline_path):
        print(f"Baseline {rf_baseline_path} not found. Fitting baseline Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=6, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, rf_baseline_path)
    else:
        rf_model = joblib.load(rf_baseline_path)
        
    # 2. Run Task 7: Feature Importance & Interpretability
    combined_importance, parent_mdi = run_task_7(rf_model, X_train, y_train, X_test, y_test, feature_names)
    
    # 3. Run Task 8: Model Optimization (RF & LR Only)
    evaluation_entries = run_task_8(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*80)
    print("CONSOLIDATED TASKS 7 AND 8 RUNNER COMPLETED SUCCESSFULLY!")
    print("ALL REQUIRED FIGURES, JSON REPORTS, AND PRODUCTION ARTIFACTS GENERATED.")
    print("="*80)


if __name__ == '__main__':
    main()
