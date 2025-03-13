# Statistical Analysis Functions


import numpy as np


def calculate_treatment_stats(clinical_df):
    """Calculate statistics for treatment groups"""
    df_subsets = create_treatment_subsets(clinical_df)
    sizes = [np.shape(df)[0] for df in df_subsets]
    proportiondeath = [np.mean(df["overall_survival"]) for df in df_subsets]
    return sizes, proportiondeath


def get_statistical_summaries(clinical_df):
    """Generate statistical summaries for clinical columns"""
    num_columns = ['age_at_diagnosis', 'lymph_nodes_examined_positive', 
                   'mutation_count', 'nottingham_prognostic_index', 
                   'overall_survival_months', 'tumor_size']
    cat_columns = ['chemotherapy', 'cohort', 'neoplasm_histologic_grade',
                   'hormone_therapy', 'overall_survival', 'radio_therapy', 
                   'tumor_stage']
    
    num_summary = clinical_df[num_columns].describe().T
    cat_columns.extend(clinical_df.select_dtypes(include=['object']).columns.tolist())
    cat_summary = clinical_df[cat_columns].astype('category').describe().T
    return num_summary, cat_summary

def analyze_no_treatment_group(clinical_df):
    """Analyze statistics for no-treatment group"""
    no_treatment = clinical_df[(clinical_df['chemotherapy'] == 0) & 
                              (clinical_df['hormone_therapy'] == 0) & 
                              (clinical_df['radio_therapy'] == 0)]
    
    stats = {
        "count": no_treatment.shape[0],
        "survival_proportion": np.mean(no_treatment["overall_survival"]),
        "baseline_survival": np.mean(clinical_df["overall_survival"])
    }
    return stats
