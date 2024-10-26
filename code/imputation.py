import os
import argparse
import joblib
import pandas as pd
import utils
import model_training

import warnings
warnings.filterwarnings('ignore')

def check_model_exists(model_path_allele1, model_path_allele2):
    """
    Check if the trained models exist.

    Parameters:
    model_path_allele1 (str): Path to the Allele1 model.
    model_path_allele2 (str): Path to the Allele2 model.

    Returns:
    bool: True if both models exist, False otherwise.
    """
    exists_allele1 = os.path.exists(model_path_allele1)
    exists_allele2 = os.path.exists(model_path_allele2)

    return exists_allele1 and exists_allele2

def main(retrain):
    # Define the paths to the models
    allele1_model_path = 'models/model_allele1.pkl'
    allele2_model_path = 'models/model_allele2.pkl'

    # Check if models exist
    if retrain:
        print("Start training of models.")
        model_training.train()
    elif not check_model_exists(allele1_model_path, allele2_model_path):
        print("One or both models do not exist. Start training of models.")
        model_training.train()

    print("Loading test data...")
    test_data = utils.preprocessing('test')
    X_val = test_data.drop(['Allele1', 'Allele2'], axis=1)

    print("Loading models...")
    model_allele1 = joblib.load(allele1_model_path)
    model_allele2 = joblib.load(allele2_model_path)

    print("Making predictions...")
    y_allele1 = model_allele1.predict(X_val, num_iteration=model_allele1._best_iteration)
    y_allele2 = model_allele2.predict(X_val, num_iteration=model_allele2._best_iteration)

    label_encoder_STR_Name = joblib.load('models/label_encoder_STR Name.pkl')
    test_data = utils.decode_name_column(test_data, 'STR_Name', label_encoder_STR_Name)
    test_data = test_data[['animal_id', 'STR_Name']]

    test_data['Allele1'] = y_allele1
    test_data['Allele2'] = y_allele2

    allele1_mean = test_data['Allele1'].mean()
    allele2_mean = test_data['Allele2'].mean()

    test_data = test_data.groupby(['animal_id', 'STR_Name'], as_index=False).agg({
    'Allele1': 'mean',
    'Allele2': 'mean'
    })

    str_test_df = pd.read_csv(f'data/STR_test.csv', delimiter=';').rename(columns={'STR Name': 'STR_Name'})
    str_test_df = str_test_df[['animal_id', 'STR_Name']]
    str_test_df = str_test_df.merge(test_data, on=['animal_id', 'STR_Name'], how='left')

    str_test_df['Allele1'].fillna(allele1_mean, inplace=True)
    str_test_df['Allele2'].fillna(allele2_mean, inplace=True)

    str_test_df.to_csv('data/STR_test_imputed.csv', index=False, sep=';')
    print("Imputed file saved as STR_test_imputed.csv.")

if __name__ == "__main__":
    # Argument parsing for the command-line interface
    parser = argparse.ArgumentParser(description="Imputation SNP data to STR data")
    parser.add_argument("--retrain", action='store_true', help="If set, retrain the models")
    parser.add_argument("--no-retrain", dest='retrain', action='store_false', help="If set, do not retrain the models")

    # Parse the arguments
    args = parser.parse_args()
    main(args.retrain)
