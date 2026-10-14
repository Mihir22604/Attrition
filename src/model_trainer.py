"""
AttritionIQ - Model Training & Evaluation Engine (Tasks 5 & 6)
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

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Configuration & Paths
DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
JSON_REPORTS_DIR = os.path.join(REPORTS_DIR, "classification_reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(JSON_REPORTS_DIR, exist_ok=True)

# Styling configuration for publication-quality figures
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8


def load_preprocessed_data():
    """Load stratified train and test datasets prepared in Task 4."""
    train_path = os.path.join(DATA_DIR, "train_features_preprocessed.csv")
    test_path = os.path.join(DATA_DIR, "test_features_preprocessed.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Attrition'])
    y_train = train_df['Attrition']
    
    X_test = test_df.drop(columns=['Attrition'])
    y_test = test_df['Attrition']
    
    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Loaded test data:     {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Train class balance:  0={sum(y_train==0)}, 1={sum(y_train==1)} (Pos Rate: {y_train.mean()*100:.2f}%)")
    print(f"Test class balance:   0={sum(y_test==0)}, 1={sum(y_test==1)} (Pos Rate: {y_test.mean()*100:.2f}%)")
    
    return X_train, y_train, X_test, y_test, list(X_train.columns)


def get_models() -> Dict[str, Dict[str, Any]]:
    """Define models for Task 5 experimentation."""
    models = {
        'Decision Tree (Unconstrained)': {
            'estimator': DecisionTreeClassifier(
                criterion='gini',
                max_depth=None,
                random_state=42
            ),
            'short_name': 'dt_unconstrained',
            'display_name': 'Decision Tree (Unconstrained)',
            'description': 'Unpruned baseline decision tree without depth restriction. Subject to empirical memorization/overfitting.'
        },
        'Decision Tree (Regularized & Balanced)': {
            'estimator': DecisionTreeClassifier(
                criterion='gini',
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            ),
            'short_name': 'dt_regularized',
            'display_name': 'Decision Tree (Regularized & Balanced)',
            'description': 'Pruned decision tree (max_depth=4, min_samples_leaf=5) with balanced class weights to enhance positive class sensitivity.'
        },
        'Random Forest (Balanced)': {
            'estimator': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=6,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'short_name': 'random_forest',
            'display_name': 'Random Forest (Balanced Ensemble)',
            'description': 'Ensemble of 100 decorrelated decision trees with balanced class weights and leaf regularization to minimize generalization error.'
        },
        'Logistic Regression (Balanced Baseline)': {
            'estimator': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'short_name': 'logistic_regression',
            'display_name': 'Logistic Regression (Balanced Baseline)',
            'description': 'Parametric linear baseline model with L2 regularization and balanced class weights operating on standardized features.'
        }
    }
    return models


def train_and_evaluate_all():
    """Execute Task 5 training and Task 6 evaluation workflows."""
    X_train, y_train, X_test, y_test, feature_names = load_preprocessed_data()
    models = get_models()
    
    results = {}
    
    for name, config in models.items():
        print(f"\n{'='*70}\nTraining: {name}\n{'='*70}")
        clf = config['estimator']
        
        # 1. Train model
        clf.fit(X_train, y_train)
        
        # Persist trained model artifact
        model_save_path = os.path.join(MODELS_DIR, f"{config['short_name']}.joblib")
        joblib.dump(clf, model_save_path)
        print(f"Saved model artifact to: {model_save_path}")
        
        # 2. Inferences
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        y_train_proba = clf.predict_proba(X_train)[:, 1]
        y_test_proba = clf.predict_proba(X_test)[:, 1]
        
        # 3. Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        train_roc_auc = roc_auc_score(y_train, y_train_proba)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        
        # Minority class metrics (Attrition = 1 / Yes)
        test_prec_1 = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_rec_1 = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        test_f1_1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        
        # Train minority class metrics
        train_prec_1 = precision_score(y_train, y_train_pred, pos_label=1, zero_division=0)
        train_rec_1 = recall_score(y_train, y_train_pred, pos_label=1, zero_division=0)
        train_f1_1 = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)
        
        # Confusion Matrices
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
        # Standard classification reports
        clf_rep_test_str = classification_report(
            y_test, y_test_pred,
            target_names=['Retained (0)', 'Departed (1)'],
            digits=4
        )
        clf_rep_test_dict = classification_report(
            y_test, y_test_pred,
            target_names=['Retained (0)', 'Departed (1)'],
            digits=4,
            output_dict=True
        )
        
        clf_rep_train_str = classification_report(
            y_train, y_train_pred,
            target_names=['Retained (0)', 'Departed (1)'],
            digits=4
        )
        clf_rep_train_dict = classification_report(
            y_train, y_train_pred,
            target_names=['Retained (0)', 'Departed (1)'],
            digits=4,
            output_dict=True
        )
        
        # ROC Curve coordinates
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba)
        
        # Overfitting gap
        accuracy_gap = (train_acc - test_acc) * 100
        f1_gap = (train_f1_1 - test_f1_1) * 100
        
        print("\n--- TEST CLASSIFICATION REPORT (STANDARD FORMAT) ---")
        print(clf_rep_test_str)
        print(f"ROC-AUC (Test): {test_roc_auc:.4f} | Accuracy (Test): {test_acc*100:.2f}%")
        print(f"Train vs Test Accuracy Gap: {accuracy_gap:+.2f}%")
        
        results[name] = {
            'short_name': config['short_name'],
            'display_name': config['display_name'],
            'description': config['description'],
            'model_path': model_save_path,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_roc_auc': float(train_roc_auc),
            'test_roc_auc': float(test_roc_auc),
            'train_precision_class1': float(train_prec_1),
            'test_precision_class1': float(test_prec_1),
            'train_recall_class1': float(train_rec_1),
            'test_recall_class1': float(test_rec_1),
            'train_f1_class1': float(train_f1_1),
            'test_f1_class1': float(test_f1_1),
            'accuracy_gap_percent': float(accuracy_gap),
            'f1_gap_percent': float(f1_gap),
            'confusion_matrix_test': cm_test.tolist(),
            'confusion_matrix_train': cm_train.tolist(),
            'classification_report_test_dict': clf_rep_test_dict,
            'classification_report_test_str': clf_rep_test_str,
            'classification_report_train_dict': clf_rep_train_dict,
            'classification_report_train_str': clf_rep_train_str,
            'fpr_test': fpr_test.tolist(),
            'tpr_test': tpr_test.tolist()
        }
        
        # Save individual model classification report JSON
        indiv_json_path = os.path.join(JSON_REPORTS_DIR, f"classification_report_{config['short_name']}.json")
        with open(indiv_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': name,
                'short_name': config['short_name'],
                'test_sample_size': len(y_test),
                'test_class_distribution': {'0': int(sum(y_test==0)), '1': int(sum(y_test==1))},
                'metrics_summary': {
                    'accuracy': float(test_acc),
                    'roc_auc': float(test_roc_auc),
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
                'classification_report_standard': clf_rep_test_dict,
                'raw_text_report': clf_rep_test_str
            }, f, indent=4)
        print(f"Saved individual classification report JSON to: {indiv_json_path}")
        
    # Save combined classification report JSON for all models
    combined_json_path = os.path.join(JSON_REPORTS_DIR, "combined_classification_report_all_models.json")
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'project': 'AttritionIQ — Intelligent Employee Attrition Prediction System',
            'task': 'Task 6 - Model Evaluation & Comparison',
            'population': {'train_records': len(y_train), 'test_records': len(y_test)},
            'models': {
                name: {
                    'display_name': data['display_name'],
                    'short_name': data['short_name'],
                    'metrics': {
                        'train_accuracy': data['train_accuracy'],
                        'test_accuracy': data['test_accuracy'],
                        'accuracy_gap_pct': data['accuracy_gap_percent'],
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
                } for name, data in results.items()
            }
        }, f, indent=4)
    print(f"\nSaved combined classification report JSON to: {combined_json_path}")
    
    return results, X_train, y_train, X_test, y_test, feature_names


def render_standard_classification_report_image(model_name: str, rep_dict: dict, roc_auc: float, output_path: str):
    """
    Render standard Scikit-Learn classification report as a publication-grade image.
    Strictly follows the standard tabular format:
    Rows: Retained (0), Departed (1), Accuracy, Macro Avg, Weighted Avg.
    Columns: Precision, Recall, F1-Score, Support.
    """
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    ax.axis('off')
    
    # Extract standard rows
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
    
    # Styling cells
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#94A3B8')
        cell.set_linewidth(1.0)
        if i == 0:
            # Header
            cell.set_facecolor('#0F172A')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
            cell.set_height(0.14)
        elif i == 2:
            # Minority Class Highlight
            cell.set_facecolor('#EFF6FF')
            cell.get_text().set_color('#1E3A8A')
            cell.get_text().set_weight('bold' if j in [0, 2, 3] else 'normal')
            cell.set_height(0.12)
        elif i == 3:
            # Accuracy row
            cell.set_facecolor('#F8FAFC')
            cell.get_text().set_color('#334155')
            cell.get_text().set_weight('bold' if j == 0 else 'normal')
            cell.set_height(0.12)
        elif i in [4, 5]:
            # Averages
            cell.set_facecolor('#F1F5F9')
            cell.get_text().set_color('#0F172A')
            cell.get_text().set_weight('bold' if j == 0 else 'normal')
            cell.set_height(0.12)
        else:
            cell.set_facecolor('#FFFFFF')
            cell.get_text().set_color('#1E293B')
            cell.set_height(0.12)
            
    # Title & Subtitle
    plt.suptitle(f"Classification Report (Standard Format): {model_name}",
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.94)
    plt.title(f"Evaluated on Stratified Test Split (N = 294) | ROC-AUC Score: {roc_auc:.4f}",
              fontsize=10.5, color='#475569', pad=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated standard classification report image: {output_path}")


def render_combined_classification_report_image(results: dict, output_path: str):
    """
    Render comprehensive combined classification report comparing all models in standard form.
    """
    fig = plt.figure(figsize=(15, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, (name, data) in enumerate(results.items()):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
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
            bbox=[0.02, 0.05, 0.96, 0.78]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        
        # Color coding
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor('#CBD5E1')
            if i == 0:
                cell.set_facecolor('#1E293B')
                cell.get_text().set_color('white')
                cell.get_text().set_weight('bold')
                cell.set_height(0.14)
            elif i == 2:
                cell.set_facecolor('#FEF3C7' if 'Unconstrained' in name else '#EFF6FF')
                cell.get_text().set_color('#92400E' if 'Unconstrained' in name else '#1E3A8A')
                cell.get_text().set_weight('bold' if j in [0, 2] else 'normal')
            elif i == 3:
                cell.set_facecolor('#F8FAFC')
            elif i in [4, 5]:
                cell.set_facecolor('#F1F5F9')
                cell.get_text().set_weight('bold' if j == 0 else 'normal')
            else:
                cell.set_facecolor('#FFFFFF')
                
        ax.set_title(
            f"{data['display_name']}\nTest Acc: {data['test_accuracy']*100:.2f}% | Recall (1): {data['test_recall_class1']*100:.2f}% | ROC-AUC: {data['test_roc_auc']:.4f}",
            fontsize=11.5, fontweight='bold', color='#0F172A', pad=12
        )
        
    plt.suptitle("AttritionIQ — Combined Standard Classification Report Benchmarking All Models\n"
                 "Evaluated on Stratified Out-of-Sample Test Set (N = 294, 247 Retained vs 47 Departed)",
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated combined classification report image: {output_path}")


def render_all_evaluation_figures(results: dict):
    """
    Generate all required publication-quality figures:
    1. Overlaid ROC Curves
    2. 2x2 Confusion Matrices
    3. Train vs Test Performance (Overfitting Diagnostics)
    4. Individual Standard Classification Reports
    5. Combined Standard Classification Report
    """
    # 1. Overlaid ROC Curves
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    colors = {
        'Decision Tree (Unconstrained)': '#DC2626',
        'Decision Tree (Regularized & Balanced)': '#2563EB',
        'Random Forest (Balanced)': '#059669',
        'Logistic Regression (Balanced Baseline)': '#7C3AED'
    }
    
    for name, data in results.items():
        fpr = data['fpr_test']
        tpr = data['tpr_test']
        auc = data['test_roc_auc']
        c = colors.get(name, '#333333')
        lw = 2.5 if 'Random Forest' in name or 'Regularized' in name else 1.8
        ls = '--' if 'Unconstrained' in name else '-'
        ax.plot(fpr, tpr, color=c, lw=lw, ls=ls, label=f"{name} (AUC = {auc:.4f})")
        
    ax.plot([0, 1], [0, 1], color='#94A3B8', linestyle=':', lw=1.5, label='No Discrimination Baseline (AUC = 0.5000)')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.03])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', color='#1E293B')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12, fontweight='bold', color='#1E293B')
    ax.set_title('AttritionIQ: Overlaid ROC Curves Across All Models\nComparative Out-of-Sample Discriminative Capacity',
                 fontsize=13.5, fontweight='bold', color='#0F172A', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
    
    roc_fig_path = os.path.join(FIGURES_DIR, "overlaid_roc_curves.png")
    plt.tight_layout()
    plt.savefig(roc_fig_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Generated overlaid ROC curve figure: {roc_fig_path}")

    # 2. Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), dpi=300)
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        cm = np.array(data['confusion_matrix_test'])
        
        # Annotations with percentages and labels
        total = cm.sum()
        group_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
        group_counts = [f"{val:d}" for val in cm.flatten()]
        group_percentages = [f"{val/total*100:.1f}%" for val in cm.flatten()]
        labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        sns.heatmap(
            cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'},
            linewidths=1.5, linecolor='#CBD5E1'
        )
        
        ax.set_title(
            f"{data['display_name']}\nTest Acc: {data['test_accuracy']*100:.1f}% | Recall (Yes): {data['test_recall_class1']*100:.1f}%",
            fontsize=11.5, fontweight='bold', color='#0F172A', pad=10
        )
        ax.set_xlabel('Predicted Label (0 = Retained, 1 = Departed)', fontsize=10.5, fontweight='bold')
        ax.set_ylabel('True Label (0 = Retained, 1 = Departed)', fontsize=10.5, fontweight='bold')
        ax.set_xticklabels(['Retained (0)', 'Departed (1)'], fontsize=10)
        ax.set_yticklabels(['Retained (0)', 'Departed (1)'], fontsize=10)
        
    plt.suptitle("AttritionIQ: Confusion Matrix Breakdown Across All Evaluated Models\n"
                 "Test Population: N = 294 (247 Actual Retained, 47 Actual Departed)",
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.98)
    
    cm_fig_path = os.path.join(FIGURES_DIR, "confusion_matrices_all_models.png")
    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Generated confusion matrices figure: {cm_fig_path}")

    # 3. Train vs Test Performance (Overfitting Diagnostics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    model_names_short = [
        "DT (Unconstrained)",
        "DT (Regularized)",
        "Random Forest",
        "Logistic Reg"
    ]
    train_accuracies = [d['train_accuracy']*100 for d in results.values()]
    test_accuracies = [d['test_accuracy']*100 for d in results.values()]
    train_f1s = [d['train_f1_class1']*100 for d in results.values()]
    test_f1s = [d['test_f1_class1']*100 for d in results.values()]
    
    x = np.arange(len(model_names_short))
    width = 0.35
    
    # Accuracy comparison
    rects1 = ax1.bar(x - width/2, train_accuracies, width, label='Train Accuracy', color='#93C5FD', edgecolor='#2563EB')
    rects2 = ax1.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='#1D4ED8', edgecolor='#1E3A8A')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Accuracy: Training vs Test Set\n(Diagnosing Empirical Memorization)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names_short, rotation=15, ha='right', fontsize=9.5)
    ax1.set_ylim([0, 110])
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True)
    
    # Add values on bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # F1-Score (Class 1) comparison
    rects3 = ax2.bar(x - width/2, train_f1s, width, label='Train F1 (Class 1)', color='#FCA5A5', edgecolor='#DC2626')
    rects4 = ax2.bar(x + width/2, test_f1s, width, label='Test F1 (Class 1)', color='#B91C1C', edgecolor='#7F1D1D')
    
    ax2.set_ylabel('F1-Score for Attrition=Yes (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Minority Class F1-Score: Training vs Test Set\n(True Generalization on Departures)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names_short, rotation=15, ha='right', fontsize=9.5)
    ax2.set_ylim([0, 110])
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True)
    
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.suptitle("AttritionIQ: Overfitting Diagnosis & Generalization Gap Across Models",
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.98)
    
    overfit_fig_path = os.path.join(FIGURES_DIR, "train_vs_test_overfitting_comparison.png")
    plt.tight_layout()
    plt.savefig(overfit_fig_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Generated overfitting diagnosis figure: {overfit_fig_path}")

    # 4. Individual Standard Classification Reports as Images
    for name, data in results.items():
        img_out = os.path.join(FIGURES_DIR, f"classification_report_{data['short_name']}.png")
        render_standard_classification_report_image(
            model_name=data['display_name'],
            rep_dict=data['classification_report_test_dict'],
            roc_auc=data['test_roc_auc'],
            output_path=img_out
        )
        
    # 5. Combined Classification Report Image
    comb_img_out = os.path.join(FIGURES_DIR, "combined_classification_report_all_models.png")
    render_combined_classification_report_image(results, comb_img_out)


def visualize_decision_tree_structure(results: dict, feature_names: list):
    """
    Generate structural visualization for the Regularized Decision Tree (Task 5 deliverable).
    """
    model_path = os.path.join(MODELS_DIR, "dt_regularized.joblib")
    dt_reg = joblib.load(model_path)
    
    fig, ax = plt.subplots(figsize=(24, 12), dpi=300)
    plot_tree(
        dt_reg,
        feature_names=feature_names,
        class_names=['Retained', 'Departed'],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        max_depth=3,  # Show top 3 levels for clean legibility
        proportion=True
    )
    ax.set_title("AttritionIQ: Regularized Decision Tree Architecture (Top 3 Decision Tiers)\n"
                 "Config: max_depth=4, min_samples_leaf=5, class_weight='balanced'",
                 fontsize=16, fontweight='bold', pad=20)
    
    tree_fig_path = os.path.join(FIGURES_DIR, "decision_tree_visualization.png")
    plt.tight_layout()
    plt.savefig(tree_fig_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Generated decision tree architecture visualization: {tree_fig_path}")


if __name__ == '__main__':
    print("="*75)
    print("STARTING ATTRITIONIQ TASK 5 & TASK 6 EXECUTION ENGINE")
    print("="*75)
    results, X_train, y_train, X_test, y_test, feature_names = train_and_evaluate_all()
    print("\nRendering all publication-grade evaluation figures...")
    render_all_evaluation_figures(results)
    print("\nRendering Decision Tree architectural visualization...")
    visualize_decision_tree_structure(results, feature_names)
    print("\nAll model training, evaluations, JSON exports, and figures generated successfully!")
