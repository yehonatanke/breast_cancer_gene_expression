import os
import pandas as pd
from google.colab import files


def load_kaggle_dataset_to_df(dataset_name, output_dir='/content/data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        !pip install -q kaggle

        from google.colab import files
        print("Please upload your kaggle.json file from Kaggle account settings if not already uploaded")
        uploaded = files.upload()

        !mkdir -p ~/.kaggle
        !mv kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json

        !kaggle datasets download -d {dataset_name} -p {output_dir}

        for file in os.listdir(output_dir):
            if file.endswith('.zip'):
                !unzip -q {os.path.join(output_dir, file)} -d {output_dir}
                !rm {os.path.join(output_dir, file)}

        csv_file = None
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                csv_file = os.path.join(output_dir, file)
                break

        if csv_file:
            df = pd.read_csv(csv_file)
            print(f"Dataset loaded successfully from {csv_file}")
            return df
        else:
            print("No CSV file found in the dataset")
            return None

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


def format_x_labels(labels):
    formatted_labels = []
    for label in labels:
        label = label.replace('_', ' ')
        label = ' '.join(word.capitalize() for word in label.split())
        
        words = label.split()
        if len(words) > 1:
            formatted_label = '\n'.join(words)
        else:
            formatted_label = label
            
        formatted_labels.append(formatted_label)
    return formatted_labels


# Data Subset Creation Functions
def create_survival_subsets(clinical_df):
    """Create subsets based on survival status"""
    died = clinical_df[clinical_df['overall_survival'] == 0]
    survived = clinical_df[clinical_df['overall_survival'] == 1]
    return died, survived


def create_death_cause_subsets(clinical_df):
    """Create subsets based on cause of death"""
    alive = clinical_df[clinical_df['death_from_cancer'] == 'Living']
    died_cancer = clinical_df[clinical_df['death_from_cancer'] == 'Died of Disease']
    died_not_cancer = clinical_df[clinical_df['death_from_cancer'] == 'Died of Other Causes']
    return alive, died_cancer, died_not_cancer


def create_treatment_subsets(clinical_df):
    """Create subsets based on treatment combinations"""
    chemo = clinical_df[(clinical_df["chemotherapy"] == True) & 
                       (clinical_df["radio_therapy"] == False) & 
                       (clinical_df["hormone_therapy"] == False)]
    radio = clinical_df[(clinical_df["chemotherapy"] == False) & 
                       (clinical_df["radio_therapy"] == True) & 
                       (clinical_df["hormone_therapy"] == False)]
    hormonal = clinical_df[(clinical_df["chemotherapy"] == False) & 
                         (clinical_df["radio_therapy"] == False) & 
                         (clinical_df["hormone_therapy"] == True)]
    chemo_radio = clinical_df[(clinical_df["chemotherapy"] == True) & 
                            (clinical_df["radio_therapy"] == True) & 
                            (clinical_df["hormone_therapy"] == False)]
    radio_hormonal = clinical_df[(clinical_df["chemotherapy"] == False) & 
                               (clinical_df["radio_therapy"] == True) & 
                               (clinical_df["hormone_therapy"] == True)]
    hormonal_chemo = clinical_df[(clinical_df["chemotherapy"] == True) & 
                               (clinical_df["radio_therapy"] == False) & 
                               (clinical_df["hormone_therapy"] == True)]
    all_3 = clinical_df[(clinical_df["chemotherapy"] == True) & 
                       (clinical_df["radio_therapy"] == True) & 
                       (clinical_df["hormone_therapy"] == True)]
    return [chemo, radio, hormonal, chemo_radio, radio_hormonal, hormonal_chemo, all_3]



