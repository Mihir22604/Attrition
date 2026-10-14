"""
AttritionIQ - Comprehensive Multi-Model Hyperparameter Tuning & Post-Tuning Classification Reports
Author: Senior Machine Learning Project Analyst & Technical Documentation Specialist
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Directories
DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
JSON_REPORTS_DIR = os.path.join(REPORTS_DIR, "classification_reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(JSON_REPORTS_DIR, exist_ok=True)

# Styling
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
    
    return X_train, y_train, X_test, y_test


def get_tuning_spaces():
    """Define systematic search spaces for all three candidate architectures."""
    tuning_spaces = {
        'Decision Tree (Tuned)': {
            'estimator': DecisionTreeClassifier(random_state=42),
            'short_name': 'dt_tuned',
            'display_name': 'Decision Tree (Tuned & Pruned)',
            'param_grid': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 4, 5, 6, 8],
                'min_samples_split': [4, 10, 20],
                'min_samples_leaf': [2, 4, 8, 12],
                'class_weight': ['balanced', {0: 1, 1: 3.5}, {0: 1, 1: 5.0}]
            },
            'color': '#2563EB',
            'rationale': 'Tune structural depth, purity criterion, and cost-sensitive class weights to eliminate unconstrained tree overfitting while maximizing minority recall.'
        },
        'Random Forest (Tuned)': {
            'estimator': RandomForestClassifier(random_state=42, n_jobs=-1),
            'short_name': 'random_forest_tuned',
            'display_name': 'Random Forest (Tuned & Optimized)',
            'param_grid': {
                'n_estimators': [100, 150, 200],
                'max_depth': [4, 6, 8, 10],
                'min_samples_split': [4, 8, 12],
                'min_samples_leaf': [2, 4, 6],
                'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 3.5}]
            },
            'color': '#059669',
            'rationale': 'Optimize ensemble size, depth bounds, and leaf sample thresholds with balanced subsampling to boost minority departure detection without alert fatigue.'
        },
        'Logistic Regression (Tuned)': {
            'estimator': LogisticRegression(random_state=42, max_iter=1000),
            'short_name': 'logistic_regression_tuned',
            'display_name': 'Logistic Regression (Tuned & Calibrated)',
            'param_grid': {
                'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', {0: 1, 1: 3.0}, {0: 1, 1: 4.0}, {0: 1, 1: 5.0}]
            },
            'color': '#7C3AED',
            'rationale': 'Tune inverse regularization strength (C), norm penalty (L1 sparsity vs L2 shrinkage), and class-specific loss multipliers on standardized features.'
        }
    }
    return tuning_spaces


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
    print(f"Saved classification report image to: {output_path}")


def render_combined_tuned_reports_image(tuned_results: dict, output_path: str):
    """Render combined 3-panel publication comparison showing standard classification reports for all tuned models."""
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.18)
    
    for idx, (name, data) in enumerate(tuned_results.items()):
        ax = fig.add_subplot(gs[0, idx])
        ax.axis('off')
        
        rep = data['classification_report_test_dict']
        table_rows = [
            ['Retained (0)', f"{rep['Retained (0)']['precision']:.4f}", f"{rep['Retained (0)']['recall']:.4f}", f"{rep['Retained (0)']['f1-score']:.4f}", f"{int(rep['Retained (0)']['support'])}"],
            ['Departed (1)', f"{rep['Departed (1)']['precision']:.4f}", f"{rep['Departed (1)']['recall']:.4f}", f"{rep['Departed (1)']['f1-score']:.4f}", f"{int(rep['Departed (1)']['support'])}"],
            ['accuracy', '', '', f"{rep['accuracy']:.4f}", f"{int(rep['macro avg']['support'])}"],
            ['macro avg', f"{rep['macro avg']['precision']:.4f}", f"{rep['macro avg']['recall']:.4f}", f"{rep['macro avg']['f1-score']:.4f}", f"{int(rep['macro avg']['support'])}"],
            ['weighted avg', f"{rep['weighted avg']['precision']:.4f}", f"{rep['weighted avg']['recall']:.4f}", f"{rep['weighted avg']['f1-score']:.4f}", f"{int(rep['weighted avg']['support'])}"]
        ]
        
        col_labels = ['Class / Metric', 'Precision', 'Recall', 'F1', 'Support']
        table = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            bbox=[0.02, 0.08, 0.96, 0.75]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor('#CBD5E1')
            if i == 0:
                cell.set_facecolor('#0F172A')
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
                cell.set_height(0.14)
            elif i == 2:
                cell.set_facecolor('#EFF6FF')
                cell.get_text().set_color('#1E3A8A')
                cell.get_text().set_weight('bold' if j in [0, 2, 3] else 'normal')
            elif i == 3:
                cell.set_facecolor('#F8FAFC')
            elif i in [4, 5]:
                cell.set_facecolor('#F1F5F9')
                cell.get_text().set_weight('bold' if j == 0 else 'normal')
            else:
                cell.set_facecolor('#FFFFFF')
                
        ax.set_title(
            f"{data['display_name']}\nTest Acc: {data['test_accuracy']*100:.2f}% | Recall (1): {data['test_recall_class1']*100:.2f}%\nF1 (1): {data['test_f1_class1']:.4f} | ROC-AUC: {data['test_roc_auc']:.4f}",
            fontsize=11, fontweight='bold', color='#0F172A', pad=12
        )
        
    plt.suptitle("AttritionIQ — Combined Standard Classification Report for All Models After Hyperparameter Tuning\n"
                 "Evaluated on Held-Out Test Split (N = 294, 247 Retained vs 47 Departed)",
                 fontsize=13.5, fontweight='bold', color='#0F172A', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated combined tuned classification report image: {output_path}")


def main():
    X_train, y_train, X_test, y_test = load_data()
    tuning_spaces = get_tuning_spaces()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tuned_results = {}
    
    print("\n" + "="*80)
    print("STARTING SYSTEMATIC HYPERPARAMETER TUNING ACROSS ALL CANDIDATE MODELS")
    print("Optimization Objective: F1-Score on Attrition = Yes (Minority Class Detection Priority)")
    print("="*80)
    
    for name, config in tuning_spaces.items():
        print(f"\n{'-'*75}\nTUNING ARCHITECTURE: {name}\n{'-'*75}")
        print(f"Rationale: {config['rationale']}")
        print(f"Executing GridSearchCV with 5-Fold Stratified CV on training set (N = 1,176)...")
        
        grid = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        
        best_estimator = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_f1 = grid.best_score_
        
        # Serialize model
        model_path = os.path.join(MODELS_DIR, f"{config['short_name']}.joblib")
        joblib.dump(best_estimator, model_path)
        print(f"Saved tuned model to: {model_path}")
        print(f"Best 5-Fold CV F1-Score: {best_cv_f1:.4f}")
        print("Best Hyperparameters:")
        for k, v in best_params.items():
            print(f"  * {k}: {v}")
            
        # Evaluate on held-out test split
        y_test_pred = best_estimator.predict(X_test)
        y_test_proba = best_estimator.predict_proba(X_test)[:, 1]
        y_train_pred = best_estimator.predict(X_train)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        gap_acc = (train_acc - test_acc) * 100
        
        test_prec_1 = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_rec_1 = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_f1_1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
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
        
        print("\nTest Classification Report (After Tuning):")
        print(clf_rep_str)
        print(f"Test Accuracy: {test_acc*100:.2f}% | Recall (1): {test_rec_1*100:.2f}% | Precision (1): {test_prec_1*100:.2f}%")
        print(f"F1-Score (1): {test_f1_1:.4f} | ROC-AUC: {test_roc_auc:.4f} | Generalization Gap: {gap_acc:+.2f}%")
        
        # Save individual classification report image
        rep_img_path = os.path.join(FIGURES_DIR, f"classification_report_{config['short_name']}.png")
        render_standard_classification_report_image(config['display_name'], clf_rep_dict, test_roc_auc, rep_img_path)
        
        # Save individual classification report JSON
        rep_json_path = os.path.join(JSON_REPORTS_DIR, f"classification_report_{config['short_name']}.json")
        with open(rep_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': name,
                'display_name': config['display_name'],
                'short_name': config['short_name'],
                'best_hyperparameters': best_params,
                'best_cv_f1': float(best_cv_f1),
                'metrics_summary': {
                    'train_accuracy': float(train_acc),
                    'test_accuracy': float(test_acc),
                    'accuracy_gap_percent': float(gap_acc),
                    'test_roc_auc': float(test_roc_auc),
                    'precision_attrition_yes': float(test_prec_1),
                    'recall_attrition_yes': float(test_rec_1),
                    'f1_attrition_yes': float(test_f1_1)
                },
                'confusion_matrix': {
                    'true_negatives': int(cm_test[0, 0]),
                    'false_positives': int(cm_test[0, 1]),
                    'false_negatives': int(cm_test[1, 0]),
                    'true_positives': int(cm_test[1, 1])
                },
                'classification_report_standard': clf_rep_dict,
                'raw_text_report': clf_rep_str
            }, f, indent=4)
        print(f"Saved classification report JSON to: {rep_json_path}")
        
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        
        tuned_results[name] = {
            'display_name': config['display_name'],
            'short_name': config['short_name'],
            'color': config['color'],
            'best_params': best_params,
            'best_cv_f1': float(best_cv_f1),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'accuracy_gap_percent': float(gap_acc),
            'test_precision_class1': float(test_prec_1),
            'test_recall_class1': float(test_rec_1),
            'test_f1_class1': float(test_f1_1),
            'test_roc_auc': float(test_roc_auc),
            'confusion_matrix_test': cm_test.tolist(),
            'classification_report_test_dict': clf_rep_dict,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
    # Render combined 3-panel classification report image
    comb_img_path = os.path.join(FIGURES_DIR, "combined_classification_report_all_tuned_models.png")
    render_combined_tuned_reports_image(tuned_results, comb_img_path)
    
    # Save combined classification report JSON
    comb_json_path = os.path.join(JSON_REPORTS_DIR, "combined_classification_report_all_tuned_models.json")
    with open(comb_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'project': 'AttritionIQ — Intelligent Employee Attrition Prediction System',
            'task': 'Task 8 - Post-Hyperparameter Tuning Cross-Architecture Benchmark',
            'population': {'train_records': len(y_train), 'test_records': len(y_test)},
            'models': {
                name: {
                    'display_name': data['display_name'],
                    'short_name': data['short_name'],
                    'best_hyperparameters': data['best_params'],
                    'best_cv_f1': data['best_cv_f1'],
                    'metrics': {
                        'train_accuracy': data['train_accuracy'],
                        'test_accuracy': data['test_accuracy'],
                        'accuracy_gap_percent': data['accuracy_gap_percent'],
                        'test_roc_auc': data['test_roc_auc'],
                        'test_precision_class_1': data['test_precision_class1'],
                        'test_recall_class_1': data['test_recall_class1'],
                        'test_f1_class_1': data['test_f1_class1'],
                        'true_positives': data['confusion_matrix_test'][1][1],
                        'false_negatives': data['confusion_matrix_test'][1][0],
                        'false_positives': data['confusion_matrix_test'][0][1],
                        'true_negatives': data['confusion_matrix_test'][0][0]
                    },
                    'standard_classification_report': data['classification_report_test_dict']
                } for name, data in tuned_results.items()
            }
        }, f, indent=4)
    print(f"\nSaved combined tuned classification report JSON to: {comb_json_path}")
    
    # Generate Overlaid ROC Curves for All Tuned Models
    plt.figure(figsize=(8.5, 6.5), dpi=300)
    for name, data in tuned_results.items():
        plt.plot(data['fpr'], data['tpr'], color=data['color'], lw=2.4,
                 label=f"{data['display_name']} (AUC = {data['test_roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='#94A3B8', linestyle=':', lw=1.5, label='Chance Baseline (AUC = 0.5000)')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.03])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold', labelpad=8)
    plt.title('AttritionIQ: Overlaid ROC Curves Across All Tuned Models\nEvaluated on Held-Out Test Set (N = 294)',
              fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    roc_out_path = os.path.join(FIGURES_DIR, "task8_all_tuned_models_roc_curves.png")
    plt.savefig(roc_out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated overlaid ROC curve for all tuned models: {roc_out_path}")
    
    # Generate Overlaid Confusion Matrices for All Tuned Models
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    for idx, (name, data) in enumerate(tuned_results.items()):
        ax = axes[idx]
        cm = np.array(data['confusion_matrix_test'])
        tot = cm.sum()
        labels = np.array([
            [f"TN: {cm[0,0]}\n({cm[0,0]/tot*100:.1f}%)", f"FP: {cm[0,1]}\n({cm[0,1]/tot*100:.1f}%)"],
            [f"FN: {cm[1,0]}\n({cm[1,0]/tot*100:.1f}%)", f"TP: {cm[1,1]}\n({cm[1,1]/tot*100:.1f}%)"]
        ])
        cmap = 'Blues' if idx == 0 else ('Greens' if idx == 1 else 'Purples')
        sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, cbar=False, ax=ax,
                    annot_kws={'fontsize': 10.5, 'fontweight': 'bold'}, linewidths=1.5, linecolor='#CBD5E1')
        ax.set_title(f"{data['display_name']}\nRecall: {data['test_recall_class1']*100:.1f}% | Prec: {data['test_precision_class1']*100:.1f}%\nF1: {data['test_f1_class1']:.4f}",
                     fontsize=10.5, fontweight='bold', pad=8)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=9.5)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=9.5)
        ax.set_xticklabels(['Retained (0)', 'Departed (1)'])
        ax.set_yticklabels(['Retained (0)', 'Departed (1)'])
        
    plt.suptitle("AttritionIQ: Confusion Matrix Breakdown Across All Tuned Architectures (Test Set N = 294)",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    cm_out_path = os.path.join(FIGURES_DIR, "task8_all_tuned_models_confusion_matrices.png")
    plt.savefig(cm_out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated confusion matrices for all tuned models: {cm_out_path}")
    
    print("\n" + "="*80)
    print("ALL MODELS SUCCESSFULLY TUNED & EVALUATED WITH POST-TUNING REPORTS GENERATED!")
    print("="*80)


if __name__ == '__main__':
    main()
