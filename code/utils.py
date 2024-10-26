import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
import joblib
import re

def encode_name_column(df, column_name):
    """
    Encodes a given column into numeric features.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    column_name (str): Column name to encode.

    Returns:
    pd.DataFrame: Dataframe with encoded features.
    """
    df[f'{column_name}_text'] = df[column_name].apply(lambda x: re.split(r'(\d+)', x)[0])
    df[f'{column_name}_num'] = df[column_name].apply(lambda x: int(re.split(r'(\d+)', x)[1]) if re.split(r'(\d+)', x)[1].isdigit() else 0)
    
    label_encoder = LabelEncoder()
    df[f'{column_name}_text_encoded'] = label_encoder.fit_transform(df[f'{column_name}_text'])
    joblib.dump(label_encoder, f'models/label_encoder_{column_name}.pkl')
    
    df.drop(columns=[column_name, f'{column_name}_text'], inplace=True)
    return df

def decode_name_column(df, column_name, label_encoder):
    """
    Decodes numeric features back into the original string format.

    Parameters:
    df (pd.DataFrame): Dataframe with encoded features.
    column_name (str): Original column name that was encoded.
    label_encoder (LabelEncoder): The fitted LabelEncoder used for encoding.

    Returns:
    pd.DataFrame: Dataframe with the original column restored.
    """
    # Decode the text part using the inverse transform of LabelEncoder
    df[f'{column_name}_text'] = label_encoder.inverse_transform(df[f'{column_name}_text_encoded'])
    
    # Combine the decoded text part and the numerical part to form the original column
    df[column_name] = df[f'{column_name}_text'] + df[f'{column_name}_num'].astype(str)
    
    # Drop the intermediate encoded columns
    df.drop(columns=[f'{column_name}_text', f'{column_name}_text_encoded', f'{column_name}_num'], inplace=True)
    
    return df

def reduce_memory_usage(df):
    """
    Reduces memory usage of a dataframe by downcasting numeric columns.

    Parameters:
    df (pd.DataFrame): Input dataframe.

    Returns:
    pd.DataFrame: Optimized dataframe.
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

def preprocessing(option):
    # Loading and preprocessing data
    print("Loading SNP data...")
    snp_df = pd.read_csv('data/FinalReport.csv', delimiter=';').drop_duplicates()
    
    print("Loading SNP map data...")
    snp_map_df = pd.read_csv('data/snp_map_file.csv', delimiter=';').drop_duplicates()
    snp_map_df = snp_map_df[['Name', 'Chromosome', 'Position', 'GenTrain Score', 'SNP']]
    snp_map_df['Chromosome'] = pd.to_numeric(snp_map_df['Chromosome'], errors='coerce')
    snp_map_df['Position'] = pd.to_numeric(snp_map_df['Position'], errors='coerce')
    snp_map_df['GenTrain Score'] = pd.to_numeric(snp_map_df['GenTrain Score'], errors='coerce')

    print(f"Loading STR data for {option}...")
    str_df = pd.read_csv(f'data/STR_{option}.csv', delimiter=';').drop_duplicates()

    # Data preparation
    print("Merging SNP data and SNP map...")
    snp_df = snp_df.rename(columns={'SNP Name': 'Name'}).merge(snp_map_df, on='Name', how='left')

    # Cleanup
    del snp_map_df
    gc.collect()
    
    print("Processing categorical columns...")
    snp_df['Allele1 - AB'] = snp_df['Allele1 - AB'].astype('category')
    snp_df['Allele2 - AB'] = snp_df['Allele2 - AB'].astype('category')
    snp_df['is_homozygous'] = (snp_df['Allele1 - Forward'] == snp_df['Allele2 - Forward']).astype(int)
    
    print("Creating dummy variables...")
    snp_df = pd.get_dummies(snp_df, columns=['SNP', 'Allele1 - AB', 'Allele2 - AB', 'Allele1 - Forward', 'Allele2 - Forward'], drop_first=True)

    print("Encoding text columns...")
    snp_df = encode_name_column(snp_df, 'Name')
    str_df = encode_name_column(str_df, 'STR Name')

    # Merging data
    print("Merging STR data with SNP data...")
    data = str_df.merge(snp_df, on=['animal_id'], how='inner')

    # Cleanup
    del str_df, snp_df
    gc.collect()

    print("Optimizing memory usage...")
    data = reduce_memory_usage(data)

    data.columns = data.columns.str.replace(r'[^\w]', '_', regex=True)
    data = data.drop_duplicates()

    print("Data preprocessing complete.")
    return data
