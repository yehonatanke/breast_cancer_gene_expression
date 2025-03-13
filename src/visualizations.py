# Visualization Functions

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_clinical_distributions(clinical_df, figsize=(15, 10)):
    columns = ['age_at_diagnosis', 'lymph_nodes_examined_positive', 
              'mutation_count', 'nottingham_prognostic_index', 
              'overall_survival_months', 'tumor_size']
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    
    survived_color = '#2C7BB6'  
    died_color = '#D7191C'      
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        survived = clinical_df[col][clinical_df['overall_survival']==1].dropna()
        died = clinical_df[col][clinical_df['overall_survival']==0].dropna()
        
        sns.kdeplot(survived, color=survived_color, fill=True, alpha=0.5, ax=ax)
        sns.kdeplot(died, color=died_color, fill=True, alpha=0.5, ax=ax)
        
        ax.text(0.02, 0.95, f"Survived (n={len(survived)})", transform=ax.transAxes, 
                color=survived_color, fontweight='bold')
        ax.text(0.02, 0.90, f"Mean: {survived.mean():.1f}", transform=ax.transAxes,
                color=survived_color)
        ax.text(0.02, 0.85, f"Med: {survived.median():.1f}", transform=ax.transAxes,
                color=survived_color)
        
        ax.text(0.70, 0.95, f"Deceased (n={len(died)})", transform=ax.transAxes, 
                color=died_color, fontweight='bold')
        ax.text(0.70, 0.90, f"Mean: {died.mean():.1f}", transform=ax.transAxes,
                color=died_color)
        ax.text(0.70, 0.85, f"Med: {died.median():.1f}", transform=ax.transAxes,
                color=died_color)
        
        ax.set_title(col.replace('_', ' ').title())
        ax.set_xlabel('')
        
        if i == 0:
            ax.legend(['Survived', 'Deceased'])
    
    fig.suptitle('Clinical Features by Survival Status', fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_missing_data(df, n=10, figsize=(12, 8), color_palette="viridis", save_path=None):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total_NaN', 'Percent_NaN'])
    
    missing_data_subset = missing_data.head(n)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    
    colors = sns.color_palette("rocket")

    sns.barplot(x=missing_data_subset.index, y='Percent_NaN', data=missing_data_subset, 
                ax=axes[0], palette=colors)
    sns.barplot(x=missing_data_subset.index, y='Total_NaN', data=missing_data_subset, 
                ax=axes[1], palette=colors)
    
    original_labels = missing_data_subset.index
    formatted_labels = format_x_labels(original_labels)
    axes[1].set_xticklabels(formatted_labels, rotation=0) 
    
    # Customize axes
    axes[0].set_title('Missing Data Analysis', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Missing Values (%)', fontsize=12)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.7)
    
    axes[1].set_xlabel('Features', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(labelsize=8)

    for i, p in enumerate(missing_data_subset['Percent_NaN']):
        axes[0].text(i, p + 1, f'{p:.1f}%', ha='center', fontsize=9)
    
    for i, t in enumerate(missing_data_subset['Total_NaN']):
        axes[1].text(i, t + max(missing_data_subset['Total_NaN'])*0.02, 
                    f'{int(t)}', ha='center', fontsize=9)
    
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.figtext(0.01, 0.01, f'Generated | '
                f'Total columns: {df.shape[1]} | '
                f'Total rows: {df.shape[0]}', 
                fontsize=8, color='gray')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, missing_data


def plot_standardized_distribution(df, title='Distribution of Standardized Attributes', 
                                  figsize=(15, 5), palette='viridis'):    
    fig, ax = plt.subplots(figsize=figsize)
    
    num_df = df.select_dtypes(include=np.number)
    ss = StandardScaler()
    std = ss.fit_transform(num_df)
    std_df = pd.DataFrame(std, index=num_df.index, columns=num_df.columns)
    
    melted_df = pd.melt(std_df)
    
    sns.boxplot(y="variable", x="value", data=melted_df, palette=palette, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Standardized Value (z-score)', fontsize=14)
    ax.set_ylabel('Attribute', fontsize=14)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    return fig, ax


def plot_treatment_venn(clinical_df, color='Blues'):
    """Create Venn diagram of treatment combinations"""
    df_subsets = create_treatment_subsets(clinical_df)
    sizes = [np.shape(df)[0] for df in df_subsets]
    proportiondeath = [np.mean(df["overall_survival"]) for df in df_subsets]
    
    fig, ax = plt.subplots(figsize=(8,6))
    v = venn3(subsets=sizes, set_labels=("Chemo", "Radio ", "Hormonal"), 
              ax=ax, alpha=0.6, set_colors=sns.color_palette(color))
    
    for text in v.set_labels:
        text.set_fontsize(14)
    ax.set_title("Patients by treatment group", size=20)
    plt.show()

def plot_clinical_correlation(clinical_df):
    """Plot correlation heatmap of clinical attributes"""
    fig, axs = plt.subplots(figsize=(13, 10))
    categorical_columns = clinical_df.select_dtypes(include=['object']).columns.tolist()
    unwanted_columns = ['patient_id', 'death_from_cancer']
    categorical_columns = [ele for ele in categorical_columns if ele not in unwanted_columns]
    no_id_clinical_df = pd.get_dummies(clinical_df.drop('patient_id', axis=1), 
                                      columns=categorical_columns)
    
    mask = np.triu(np.ones_like(no_id_clinical_df.corr(), dtype=bool))
    sns.heatmap(no_id_clinical_df.corr(), ax=axs, mask=mask, 
                cmap=sns.diverging_palette(180, 10, as_cmap=True))
    plt.title('Correlation between the Clinical Attributes')
    adjust_plot_margins()
    plt.show()
    return no_id_clinical_df

def plot_gene_expression_heatmap(genetic_df):
    """Plot heatmap of gene expression data"""
    fig, axs = plt.subplots(figsize=(17, 10))
    sns.heatmap(genetic_df.drop(['patient_id', 'overall_survival'], axis=1), 
                ax=axs, cmap=sns.diverging_palette(180, 10, as_cmap=True))
    plt.title('Gene Expression Heatmap')
    adjust_plot_margins()
    plt.show()


# Visualization Function for Models
def visualize_model_performance(results, y_test, color='Blues', title="Accuracy scores for basic models", 
                              inds=None):
    """Visualize model performance with accuracy bars and ROC curves"""
    if inds is None:
        inds = range(1, len(results) + 1)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15,6))
    fig.suptitle(title, fontsize=16)
    
    labels = list(results.keys())
    cv_scores = [results[name]['cv_score'] for name in labels]
    test_scores = [results[name]['test_score'] for name in labels]
    preds = [results[name]['pred'] for name in labels]
    
    # Accuracy bars
    ax1.bar(inds, cv_scores, color=sns.color_palette(color)[5], alpha=0.3, 
            hatch="x", edgecolor="none", label="CrossValidation Set")
    ax1.bar(inds, test_scores, color=sns.color_palette(color)[0], label="Testing set")
    ax1.set_ylim(0.4, 1)
    ax1.set_ylabel("Accuracy score")
    ax1.axhline(0.5793, color="black", linestyle="--")
    ax1.set_title("Accuracy scores", fontsize=17)
    ax1.set_xticks(inds)
    ax1.set_xticklabels(labels, size=12, rotation=40, ha="right")
    ax1.legend()

    # ROC curves
    if isinstance(y_test, list):
        for label, pred, yt in zip(labels, preds, y_test):
            fpr, tpr, _ = roc_curve(yt.values, pred)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=f'{label} (area = {roc_auc:.2f})', linewidth=2)
    else:
        for label, pred in zip(labels, preds):
            fpr, tpr, _ = roc_curve(y_test.values, pred)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=f'{label} (area = {roc_auc:.2f})', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2)
    ax2.set_xlim([-0.05, 1.0])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc="lower right", prop={'size': 12})
    ax2.set_title("ROC curve", fontsize=17)
    
    plt.show()

