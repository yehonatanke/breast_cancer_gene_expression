# Data Preprocessing Functions

import pandas as pd


def preprocess_genetic_data(df):
    """Process genetic data by dropping unnecessary features"""
    genetic_features_to_drop = df.columns[520:]
    genetic_df = df.drop(genetic_features_to_drop, axis=1)
    genetic_features_to_drop = genetic_df.columns[4:35]
    genetic_df = genetic_df.drop(genetic_features_to_drop, axis=1)
    genetic_df = genetic_df.drop(['age_at_diagnosis', 'type_of_breast_surgery', 'cancer_type'], axis=1)
    genetic_df = genetic_df.iloc[:, :-174]
    genetic_df['overall_survival'] = df['overall_survival']
    return genetic_df


def preprocess_mutation_data(df):
    """Process mutation data"""
    mutation_features_to_drop = df.columns[4:520]
    mutation_df = df.drop(mutation_features_to_drop, axis=1)
    mutation_df = mutation_df.drop(['age_at_diagnosis', 'type_of_breast_surgery', 'cancer_type'], axis=1)
    
    for column in mutation_df.columns[1:]:
        mutation_df[column] = pd.to_numeric(mutation_df[column], errors='coerce').fillna(1).astype(int)
    
    mutation_df.insert(loc=1, column='overall_survival', value=df['overall_survival'])
    return mutation_df
