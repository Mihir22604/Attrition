"""
AttritionIQ - Exploratory Data Analysis (EDA) Module (Task 3)
Author: Senior Machine Learning Project Analyst & Technical Documentation Specialist
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(input_csv: str = "data/processed/cleaned_hr_employee_attrition.csv",
            output_dir: str = "reports/figures") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    
    # Configure plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    results = {}
    
    # 1. Target Variable Analysis
    total_employees = len(df)
    left_count = (df['Attrition'] == 'Yes').sum()
    retained_count = (df['Attrition'] == 'No').sum()
    attrition_rate = (left_count / total_employees) * 100.0
    
    results['target_stats'] = {
        'total_employees': int(total_employees),
        'left_count': int(left_count),
        'retained_count': int(retained_count),
        'attrition_rate_pct': float(round(attrition_rate, 4))
    }
    
    # Figure 1: Target Class Distribution & Attrition Rate
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    palette_target = {'No': '#2b5c8f', 'Yes': '#d9534f'}
    
    # Bar plot
    sns.countplot(data=df, x='Attrition', ax=ax[0], palette=palette_target, order=['No', 'Yes'])
    ax[0].set_title("Enterprise Attrition Headcount Distribution", fontsize=13, fontweight='bold', pad=12)
    ax[0].set_ylabel("Employee Count", fontsize=11)
    ax[0].set_xlabel("Attrition Status", fontsize=11)
    for p in ax[0].patches:
        height = p.get_height()
        pct = (height / total_employees) * 100
        ax[0].annotate(f"{int(height):,}\n({pct:.1f}%)",
                       (p.get_x() + p.get_width() / 2., height / 2),
                       ha='center', va='center', fontsize=11, color='white', fontweight='bold')
                       
    # Pie/Donut Chart
    ax[1].pie([retained_count, left_count], labels=['Retained (No)', 'Departed (Yes)'],
              autopct='%1.2f%%', startangle=140, colors=['#2b5c8f', '#d9534f'],
              explode=(0, 0.08), wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2),
              textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax[1].set_title(f"Baseline Enterprise Attrition Rate: {attrition_rate:.2f}%", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_target_distribution.png")
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    results['fig1'] = fig1_path

    # Figure 2: Univariate Distribution of Key Continuous Predictors (Age, MonthlyIncome, YearsAtCompany)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Age
    sns.histplot(data=df, x='Age', hue='Attrition', kde=True, ax=axes[0], palette=palette_target, bins=20, element='step')
    axes[0].set_title("Age Distribution by Attrition", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Age (Years)", fontsize=10)
    axes[0].set_ylabel("Density / Count", fontsize=10)
    
    # MonthlyIncome
    sns.histplot(data=df, x='MonthlyIncome', hue='Attrition', kde=True, ax=axes[1], palette=palette_target, bins=25, element='step')
    axes[1].set_title("Monthly Income ($) Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Monthly Income (USD)", fontsize=10)
    axes[1].set_ylabel("Density / Count", fontsize=10)
    
    # YearsAtCompany
    sns.histplot(data=df, x='YearsAtCompany', hue='Attrition', kde=True, ax=axes[2], palette=palette_target, bins=20, element='step')
    axes[2].set_title("Company Tenure Distribution", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Years at Company", fontsize=10)
    axes[2].set_ylabel("Density / Count", fontsize=10)
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_univariate_continuous_distributions.png")
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close()
    results['fig2'] = fig2_path

    # Figure 3: Department-Wise Attrition Rate & Composition
    dept_ct = pd.crosstab(df['Department'], df['Attrition'])
    dept_rate = (dept_ct['Yes'] / dept_ct.sum(axis=1) * 100).sort_values(ascending=False)
    results['dept_rate'] = dept_rate.to_dict()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    dept_ct.loc[dept_rate.index].plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[0], edgecolor='none')
    axes[0].set_title("Headcount by Department & Attrition", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Department", fontsize=10)
    axes[0].set_ylabel("Headcount", fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend(title="Attrition", labels=['No', 'Yes'])
    
    sns.barplot(x=dept_rate.index, y=dept_rate.values, ax=axes[1], palette='Reds_r')
    axes[1].axhline(attrition_rate, color='black', linestyle='--', linewidth=1.5, label=f"Org Avg ({attrition_rate:.1f}%)")
    axes[1].set_title("Attrition Rate (%) by Department", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Department", fontsize=10)
    axes[1].set_ylabel("Attrition Rate (%)", fontsize=10)
    axes[1].tick_params(axis='x', rotation=15)
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 0.5),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_department_attrition.png")
    plt.savefig(fig3_path, bbox_inches='tight')
    plt.close()
    results['fig3'] = fig3_path

    # Figure 4: JobRole Disproportionate Turnover Analysis
    role_ct = pd.crosstab(df['JobRole'], df['Attrition'])
    role_rate = (role_ct['Yes'] / role_ct.sum(axis=1) * 100).sort_values(ascending=True)
    results['role_rate'] = role_rate.to_dict()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#d9534f' if r > attrition_rate else '#2b5c8f' for r in role_rate.values]
    ax.barh(role_rate.index, role_rate.values, color=colors, edgecolor='none', height=0.65)
    ax.axvline(attrition_rate, color='darkred', linestyle='--', linewidth=1.5, label=f"Enterprise Avg ({attrition_rate:.1f}%)")
    ax.set_title("Attrition Rate (%) by Job Role (Ranked Order)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Attrition Rate (%)", fontsize=11)
    ax.set_ylabel("Job Role", fontsize=11)
    for i, v in enumerate(role_rate.values):
        ax.text(v + 0.6, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=10)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_jobrole_attrition.png")
    plt.savefig(fig4_path, bbox_inches='tight')
    plt.close()
    results['fig4'] = fig4_path

    # Figure 5: OverTime & BusinessTravel Impact on Turnover
    ot_ct = pd.crosstab(df['OverTime'], df['Attrition'])
    ot_rate = (ot_ct['Yes'] / ot_ct.sum(axis=1) * 100)
    bt_ct = pd.crosstab(df['BusinessTravel'], df['Attrition'])
    bt_rate = (bt_ct['Yes'] / bt_ct.sum(axis=1) * 100).sort_values(ascending=False)
    results['ot_rate'] = ot_rate.to_dict()
    results['bt_rate'] = bt_rate.to_dict()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Overtime
    sns.barplot(x=ot_rate.index, y=ot_rate.values, ax=axes[0], palette=['#2b5c8f', '#d9534f'])
    axes[0].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
    axes[0].set_title("Attrition Rate: Mandatory OverTime Impact", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Attrition Rate (%)", fontsize=10)
    axes[0].set_xlabel("OverTime Requirement", fontsize=10)
    for p in axes[0].patches:
        h = p.get_height()
        axes[0].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.6),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].legend()
    
    # Business Travel
    sns.barplot(x=bt_rate.index, y=bt_rate.values, ax=axes[1], palette='Purples_r')
    axes[1].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
    axes[1].set_title("Attrition Rate by Business Travel Frequency", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Attrition Rate (%)", fontsize=10)
    axes[1].set_xlabel("Business Travel Level", fontsize=10)
    for p in axes[1].patches:
        h = p.get_height()
        axes[1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.5),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "fig5_overtime_and_travel.png")
    plt.savefig(fig5_path, bbox_inches='tight')
    plt.close()
    results['fig5'] = fig5_path

    # Figure 6: Subjective Well-being: JobSatisfaction & WorkLifeBalance
    js_ct = pd.crosstab(df['JobSatisfaction'], df['Attrition'])
    js_rate = (js_ct['Yes'] / js_ct.sum(axis=1) * 100)
    wlb_ct = pd.crosstab(df['WorkLifeBalance'], df['Attrition'])
    wlb_rate = (wlb_ct['Yes'] / wlb_ct.sum(axis=1) * 100)
    results['js_rate'] = js_rate.to_dict()
    results['wlb_rate'] = wlb_rate.to_dict()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    satisfaction_labels = ['1: Low', '2: Medium', '3: High', '4: Very High']
    sns.barplot(x=[satisfaction_labels[i-1] for i in js_rate.index], y=js_rate.values, ax=axes[0], palette='Blues_r')
    axes[0].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
    axes[0].set_title("Attrition Rate by Job Satisfaction Rating", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Attrition Rate (%)", fontsize=10)
    axes[0].set_xlabel("Self-Reported Job Satisfaction", fontsize=10)
    for p in axes[0].patches:
        h = p.get_height()
        axes[0].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.4),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].legend()
    
    wlb_labels = ['1: Bad', '2: Good', '3: Better', '4: Best']
    sns.barplot(x=[wlb_labels[i-1] for i in wlb_rate.index], y=wlb_rate.values, ax=axes[1], palette='Oranges_r')
    axes[1].axhline(attrition_rate, color='black', linestyle='--', label=f"Org Avg ({attrition_rate:.1f}%)")
    axes[1].set_title("Attrition Rate by Work-Life Balance Rating", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Attrition Rate (%)", fontsize=10)
    axes[1].set_xlabel("Self-Reported Work-Life Balance", fontsize=10)
    for p in axes[1].patches:
        h = p.get_height()
        axes[1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.4),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "fig6_satisfaction_and_wlb.png")
    plt.savefig(fig6_path, bbox_inches='tight')
    plt.close()
    results['fig6'] = fig6_path

    # Figure 7: Compensation Economics: MonthlyIncome by JobLevel and Attrition (Boxplot)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='JobLevel', y='MonthlyIncome', hue='Attrition', palette=palette_target, ax=ax, fliersize=2)
    ax.set_title("Monthly Income Distribution by Job Level & Attrition", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Organizational Job Level (1: Associate to 5: Director/Exec)", fontsize=11)
    ax.set_ylabel("Monthly Income (USD)", fontsize=11)
    ax.legend(title="Attrition", loc='upper left')
    plt.tight_layout()
    fig7_path = os.path.join(output_dir, "fig7_compensation_by_joblevel.png")
    plt.savefig(fig7_path, bbox_inches='tight')
    plt.close()
    results['fig7'] = fig7_path

    # Figure 8: Full Pearson Correlation Matrix Heatmap across Numerical & Ordinal Attributes
    numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
    # add binary target for correlation
    numeric_df['Attrition_Numeric'] = (df['Attrition'] == 'Yes').astype(int)
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.5, vmax=1.0, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, annot=False, ax=ax)
    ax.set_title("Multivariate Pearson Correlation Matrix (Numerical & Ordinal Features)", fontsize=15, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig8_path = os.path.join(output_dir, "fig8_correlation_heatmap.png")
    plt.savefig(fig8_path, bbox_inches='tight')
    plt.close()
    results['fig8'] = fig8_path

    # Figure 9: Tenure Trajectory Interrelationships (YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(data=df, x='YearsAtCompany', y='YearsWithCurrManager', hue='Attrition',
                    palette=palette_target, alpha=0.7, ax=axes[0], s=50)
    axes[0].plot([0, 40], [0, 40], color='gray', linestyle='--', label="1:1 Ratio")
    axes[0].set_title("Company Tenure vs. Tenure with Current Manager", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Years at Company", fontsize=10)
    axes[0].set_ylabel("Years with Current Manager", fontsize=10)
    axes[0].legend(title="Attrition")
    
    # DistanceFromHome vs Attrition (KDE)
    sns.kdeplot(data=df, x='DistanceFromHome', hue='Attrition', palette=palette_target,
                common_norm=False, fill=True, alpha=0.3, ax=axes[1])
    axes[1].set_title("Commute Distance Density: Departed vs Retained", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Distance From Home (Miles/Km)", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    plt.tight_layout()
    fig9_path = os.path.join(output_dir, "fig9_tenure_and_commute_dynamics.png")
    plt.savefig(fig9_path, bbox_inches='tight')
    plt.close()
    results['fig9'] = fig9_path

    # Extract top correlated features with Attrition
    attrition_corrs = corr_matrix['Attrition_Numeric'].drop('Attrition_Numeric').sort_values()
    results['top_negative_corrs'] = attrition_corrs.head(5).to_dict()
    results['top_positive_corrs'] = attrition_corrs.tail(5).to_dict()
    
    return results

if __name__ == '__main__':
    res = run_eda()
    print("EDA Execution Complete!")
    print(f"Target Stats: {res['target_stats']}")
    print(f"Department Attrition Rates: {res['dept_rate']}")
    print(f"OverTime Rates: {res['ot_rate']}")
    print(f"Job Role Rates: {res['role_rate']}")
    print(f"Top Positive Corrs with Attrition: {res['top_positive_corrs']}")
    print(f"Top Negative Corrs with Attrition: {res['top_negative_corrs']}")
