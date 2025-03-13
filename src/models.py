# Model Training and Evaluation Functions


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from xgboost import XGBClassifier


def model_metrics(model, kfold, X_train, X_test, y_train, y_test):
    """Calculate and print model performance metrics"""
    model.fit(X_train, y_train)
    
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print("CV scores:", results)
    print("CV Standard Deviation:", results.std())
    print("\nCV Mean score:", results.mean())
    print("Train score:", model.score(X_train, y_train))
    print("Test score:", model.score(X_test, y_test))
    
    pred = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:")
    print(classification_report(y_test, pred))
    
    return pred, model.score(X_test, y_test), results.mean()

def run_basic_classifiers(X_train, X_test, y_train, y_test, kfold, color='Blues'):
    """Run and evaluate basic classification models"""
    BOLD = '\033[1m'
    END = '\033[0m'
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model configurations
    models = {
        'KNN': GridSearchCV(KNeighborsClassifier(), 
                           {"n_neighbors": [5,15,25,30,35,40,100], "weights": ["uniform","distance"]}, 
                           n_jobs=-1, cv=4),
        'Logistic Regression': GridSearchCV(LogisticRegression(random_state=42), 
                                          {"penalty": ["l1","l2"], "C": np.logspace(-2,4,100)}, 
                                          n_jobs=-1, cv=4),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'SVC': SVC(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{BOLD}{name} Model:{END}")
        pred, test_score, cv_score = model_metrics(model, kfold, X_train_scaled, 
                                                 X_test_scaled, y_train, y_test)
        results[name] = {'pred': pred, 'test_score': test_score, 'cv_score': cv_score}
    
    visualize_model_performance(results, y_test, color)
    return results

def run_rf_et_gridsearch(X_train, X_test, y_train, y_test, kfold, color='Blues'):
    """Run Random Forest and Extra Trees with GridSearchCV"""
    BOLD = '\033[1m'
    END = '\033[0m'
    
    rf_params = {'max_features': [2, 3, 5, 7, 8]}
    et_params = {'max_depth': [1, 2, 3, 4, 5, 8]}
    
    print(f"\n{BOLD}Grid Search with Random Forest:{END}")
    rf_gs = GridSearchCV(RandomForestClassifier(n_estimators=100), rf_params, cv=5, verbose=1)
    rf_pred, rf_test, rf_cv = model_metrics(rf_gs, kfold, X_train, X_test, y_train, y_test)
    
    print(f"\n{BOLD}Grid Search with Extra Trees:{END}")
    et_gs = GridSearchCV(ExtraTreesClassifier(n_estimators=100), et_params, cv=5, verbose=1)
    et_pred, et_test, et_cv = model_metrics(et_gs, kfold, X_train, X_test, y_train, y_test)
    
    results = {
        'Random Forest': {'pred': rf_pred, 'test_score': rf_test, 'cv_score': rf_cv},
        'Extra Trees': {'pred': et_pred, 'test_score': et_test, 'cv_score': et_cv}
    }
    visualize_model_performance(results, y_test, color, title="Random Forest and Extra Trees with Grid Search")
    return rf_gs.best_estimator_, et_gs.best_estimator_, results

def run_xgboost_analysis(clinical_df, df, genetic_df, combin_geneatic_df, kfold):
    """Run XGBoost analysis on different data subsets"""
    xgb_params = {
        'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 5,
        'min_child_weight': 1, 'gamma': 0, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'objective': 'binary:logistic',
        'nthread': 4, 'scale_pos_weight': 1, 'seed': 27
    }
    
    # Prepare data subsets
    data_subsets = {
        'Clinical': prepare_clinical_data(clinical_df),
        'Genetic': prepare_genetic_data_subset(genetic_df),
        'All': prepare_all_data(df),
        'Combined': prepare_combined_data(clinical_df, combin_geneatic_df)
    }
    
    results = {}
    y_test_sets = []
    for name, (X, y) in data_subsets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                          random_state=42, stratify=y)
        xgb = XGBClassifier(**xgb_params)
        pred, test_score, cv_score = model_metrics(xgb, kfold, X_train, X_test, y_train, y_test)
        results[name] = {'pred': pred, 'test_score': test_score, 'cv_score': cv_score}
        y_test_sets.append(y_test)
    
    visualize_model_performance(results, y_test_sets, 'Blues', 
                              "Performance of XGBoost with different subsets",
                              range(1, len(results) + 1))
    return results

# Data Preparation Functions for Modeling
def prepare_clinical_data(clinical_df):
    """Prepare clinical data for modeling"""
    categorical_columns = [col for col in clinical_df.select_dtypes(include=['object']).columns 
                         if col not in ['patient_id', 'death_from_cancer']]
    dummies_df = pd.get_dummies(clinical_df.drop('patient_id', axis=1), 
                              columns=categorical_columns, dummy_na=True)
    dummies_df.dropna(inplace=True)
    X = dummies_df.drop(['death_from_cancer', 'overall_survival'], axis=1)
    y = dummies_df['overall_survival']
    return X, y

def prepare_genetic_data_subset(genetic_df):
    """Prepare genetic data for modeling"""
    X = genetic_df.drop(['patient_id', 'overall_survival'], axis=1)
    y = genetic_df['overall_survival']
    return X, y

def prepare_all_data(df):
    """Prepare combined clinical and genetic data"""
    features_to_drop = df.columns[520:]
    df = df.drop(features_to_drop, axis=1)
    categorical_columns = [col for col in df.select_dtypes(include=['object']).columns 
                         if col not in ['patient_id', 'death_from_cancer']]
    dummies_df = pd.get_dummies(df.drop('patient_id', axis=1), 
                              columns=categorical_columns, dummy_na=True)
    dummies_df.dropna(inplace=True)
    X = dummies_df.drop(['death_from_cancer', 'overall_survival'], axis=1)
    y = dummies_df['overall_survival']
    return X, y

def prepare_combined_data(clinical_df, combin_geneatic_df):
    """Prepare combined clinical and genetic data from separate sources"""
    clinical_df_new = pd.merge(clinical_df, combin_geneatic_df, 
                             left_index=True, right_index=True, 
                             sort='patient_id', how='outer')
    categorical_columns = [col for col in clinical_df_new.select_dtypes(include=['object']).columns 
                         if col not in ['patient_id', 'death_from_cancer']]
    dummies_df = pd.get_dummies(clinical_df_new.drop('patient_id', axis=1), 
                              columns=categorical_columns, dummy_na=True)
    X = dummies_df.drop(['death_from_cancer', 'overall_survival'], axis=1)
    y = dummies_df['overall_survival']
    return X, y


# Main Modeling Function
def run_complete_model_analysis(clinical_df, df, genetic_df, combin_geneatic_df=None):
    """Run complete modeling analysis with all classifier types"""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Clinical data analysis
    X_clin, y_clin = prepare_clinical_data(clinical_df)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clin, y_clin, 
                                                              test_size=0.33, 
                                                              random_state=42, 
                                                              stratify=y_clin)
    
    # Genetic data analysis
    X_gen, y_gen = prepare_genetic_data_subset(genetic_df)
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gen, y_gen, 
                                                              test_size=0.33, 
                                                              random_state=42, 
                                                              stratify=y_gen)
    
    # Results dictionary
    results = {
        'basic_clinical': run_basic_classifiers(X_train_c, X_test_c, y_train_c, y_test_c, kfold),
        'basic_genetic': run_basic_classifiers(X_train_g, X_test_g, y_train_g, y_test_g, kfold),
        'rf_et_clinical': run_rf_et_gridsearch(X_train_c, X_test_c, y_train_c, y_test_c, kfold)
    }
    
    if combin_geneatic_df is not None:
        results['xgboost'] = run_xgboost_analysis(clinical_df, df, genetic_df, combin_geneatic_df, kfold)
    
    return results


# Main Analysis Function
def run_clinical_analysis(clinical_df, df):
    """Run complete clinical data analysis"""
    # Create subsets
    died, survived = create_survival_subsets(clinical_df)
    alive, died_cancer, died_not_cancer = create_death_cause_subsets(clinical_df)
    
    # Generate visualizations
    plot_treatment_venn(clinical_df)
    no_id_clinical_df = plot_clinical_correlation(clinical_df)
    
    # Calculate correlations with survival
    corr_survival = no_id_clinical_df.corr()['overall_survival'].sort_values(ascending=False)
    corr_df = pd.DataFrame({'Correlation': corr_survival})
    
    # Get statistical summaries
    num_summary, cat_summary = get_statistical_summaries(clinical_df)
    
    # Analyze no-treatment group
    no_treatment_stats = analyze_no_treatment_group(clinical_df)
    
    return {
        'survival_subsets': (died, survived),
        'death_cause_subsets': (alive, died_cancer, died_not_cancer),
        'correlation_df': corr_df,
        'numerical_summary': num_summary,
        'categorical_summary': cat_summary,
        'no_treatment_stats': no_treatment_stats
    }

clinical_df = df.copy
results = run_clinical_analysis(clinical_df, original_df)
