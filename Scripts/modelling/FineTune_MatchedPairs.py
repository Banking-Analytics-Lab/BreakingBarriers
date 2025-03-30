"""
FineTune_MatchedPairs.py

This script fine-tunes an LSTM model on matched director pairs using network score data.
It includes:
    1. Loading network score data from configurable paths.
    2. Filtering the data for different groups (all, male, female) – assuming matched data
       (e.g., male_data and female_data) is available.
    3. Defining a final_df function to reformat the data.
    4. Defining an LSTM model and a training function (run_model) with cross-validation over time splits.
    5. Performing a hyperparameter grid search and saving results based on a job array ID
       (SLURM_ARRAY_TASK_ID) for different data sources.
       
Users should update the configuration variables below so that file paths are hidden.
"""

# ---------------------------
# 1. Import Libraries and Set Global Settings
# ---------------------------
import os
import re
import math
import pickle
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

from fuzzywuzzy import fuzz

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ---------------------------
# 2. Configuration Variables
# ---------------------------
# Set your environment variables or update the defaults below.
DATA_DIR = os.getenv('DATA_DIR', '/path/to/your/data')  # Directory containing input CSVs.
RESULT_DIR = os.getenv('RESULT_DIR', '/path/to/results')  # Directory to save output results.
NETSCORES_PATH = os.path.join(DATA_DIR, 'multi_indexed_df_reverse_non_10_years_lagged_byhand.csv')
DIRECTOR_RECORD_PATH = os.path.join(DATA_DIR, 'DirectorTimeRecord_updated.csv')

# ---------------------------
# 3. Load Data and Define Data Sources
# ---------------------------
networkscores = pd.read_csv(NETSCORES_PATH, index_col=False)

# NOTE: The following variables (male_data and female_data) are assumed to be available from your matching step.
# They should contain the matched pairs with a 'DirectorID' column.
networkscores_male = networkscores[networkscores['DirectorID'].isin(male_data.DirectorID.unique())]
networkscores_female = networkscores[networkscores['DirectorID'].isin(female_data.DirectorID.unique())]
networkscores_match = networkscores[networkscores['DirectorID'].isin(
    list(set(female_data.DirectorID.unique()).union(set(male_data.DirectorID.unique())))
)]

# ---------------------------
# 4. Define Helper Functions and Model
# ---------------------------
def final_df(channels, networkscores):
    """
    Process the networkscores DataFrame to extract relevant network score data for selected channels.
    
    Parameters:
        channels (list): List of channel regex patterns.
        networkscores (DataFrame): The raw network scores.
    
    Returns:
        DataFrame: Processed DataFrame with columns: individual, year, network_scores, director_appointment_next_year.
    """
    condition = networkscores['Score'].str.contains('|'.join(channels))
    filtered_networkscores = networkscores[condition]
    melted_df = filtered_networkscores.melt(id_vars=['DirectorID', 'Score'], var_name='year', value_name='network_score')
    pivot_df = melted_df.pivot_table(index=['DirectorID', 'year'], columns='Score', values='network_score', aggfunc='first').reset_index()
    pivot_df['network_scores'] = pivot_df[pivot_df.columns[2:]].apply(list, axis=1)
    pivot_df = pivot_df[pivot_df['network_scores'].apply(lambda x: not all(pd.isnull(v) for v in x))]
    final_df = pivot_df[['DirectorID', 'year', 'network_scores']].copy()
    final_df.rename(columns={'DirectorID': 'individual'}, inplace=True)
    final_df['director_appointment_next_year'] = 0

    results = pd.read_csv(DIRECTOR_RECORD_PATH, index_col=None)
    results['Latest_End'] = results['Latest_End'].astype(int)
    director_end_year_dict = results.set_index('DirectorID')[['Latest_End', 'director_dummy']].T.to_dict()
    final_df['year'] = final_df['year'].astype(int)

    def fill_director_appointment(row):
        director_id = row['individual']
        year = row['year']
        if director_id in director_end_year_dict and director_end_year_dict[director_id]['director_dummy'] == 1:
            if year == director_end_year_dict[director_id]['Latest_End']:
                return 1
        return 0

    final_df['director_appointment_next_year'] = final_df.apply(fill_director_appointment, axis=1)
    return final_df

class LSTMModel(nn.Module):
    """
    LSTM model for processing network score sequences.
    
    Parameters:
        input_dim (int): Dimensionality of the input.
        hidden_dim (int): Number of hidden units.
    """
    def __init__(self, input_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        out, _ = self.lstm(x)
        out = out * mask.unsqueeze(-1)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def run_model(channels, final_df, hidden_dim, learning_rate, num_epochs):
    """
    Train and evaluate the LSTM model using cross-validation.
    
    Parameters:
        channels (list): List of channel regex patterns.
        final_df (DataFrame): Processed data.
        hidden_dim (int): LSTM hidden dimension.
        learning_rate (float): Optimizer learning rate.
        num_epochs (int): Training epochs.
    
    Returns:
        tuple: (val_years, val_auc)
    """
    n_splits = 12
    window_size = 10
    individuals = final_df['individual'].unique()
    years = sorted(final_df['year'].unique())
    
    def front_pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]
        max_length = max(lengths)
        padded_sequences = torch.zeros((len(sequences), max_length, sequences[0].shape[-1]))
        mask = torch.zeros((len(sequences), max_length))
        for i, seq in enumerate(sequences):
            length = len(seq)
            padded_sequences[i, -length:] = seq
            mask[i, -length:] = 1
        return padded_sequences, mask
    
    val_auc = []
    val_years = []
    for i in range(n_splits):
        start_year = years[0]
        end_year = years[i + window_size - 1]
        validation_year = years[i + window_size]
        val_years.append(validation_year)
        
        train_df = final_df[(final_df['year'] >= start_year) & (final_df['year'] <= end_year)]
        val_df = final_df[final_df['year'] == validation_year]
        
        X_train, y_train = [], []
        for individual in individuals:
            individual_data = train_df[train_df['individual'] == individual]
            if not individual_data.empty:
                sequence = torch.tensor([scores for scores in individual_data['network_scores']], dtype=torch.float32)
                target = individual_data['director_appointment_next_year'].values[-1]
                X_train.append(sequence)
                y_train.append(target)
        X_train, mask_train = front_pad_sequences(X_train)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        X_val, y_val = [], []
        for individual in individuals:
            individual_data = val_df[val_df['individual'] == individual]
            if not individual_data.empty:
                sequence = torch.tensor([scores for scores in individual_data['network_scores']], dtype=torch.float32)
                target = individual_data['director_appointment_next_year'].values[-1]
                X_val.append(sequence)
                y_val.append(target)
        if X_val:
            X_val, mask_val = front_pad_sequences(X_val)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            
            input_dim = len(channels) * 5
            model = LSTMModel(input_dim, hidden_dim)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            
            for _ in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train, mask_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                outputs = model(X_val, mask_val)
                y_pred_val = torch.sigmoid(outputs)
                val_auc.append(roc_auc_score(y_val, y_pred_val))
        else:
            print(f"No data in the validation set for split {i+1}")
    
    return val_years, val_auc

# ---------------------------
# 5. Define Hyperparameter Grid and Processing Functions
# ---------------------------
param_grid = {
    'learning_rate': [0.1, 0.05, 0.01],
    'hidden_dim': [64, 128, 256],
    'num_epochs': [50, 100, 150]
}
grid = ParameterGrid(param_grid)
channels = ['(?<!P)OA', 'POA', 'CE', 'PE', 'EDU']
measurements = ['degree', 'closeness', 'betweeness', '(?<!Person_)PageRank', 'Person_PageRank']

def process_dataset_channels(data_source, grid, channels):
    auc_records_channel = {}
    for ch in tqdm(combinations(channels, 1)):
        auc_records_channel[ch] = {}
        final_df_data = final_df(list(ch), data_source)
        for _ in tqdm(range(len(grid))):
            params = grid[_]
            learning_rate = params['learning_rate']
            hidden_dim = params['hidden_dim']
            num_epochs = params['num_epochs']
            val_years, auc_for_each_year = run_model(list(ch), final_df_data, hidden_dim, learning_rate, num_epochs)
            param_key = str(params)
            auc_records_channel[ch][param_key] = dict(zip(val_years, auc_for_each_year))
    return auc_records_channel

def process_dataset_measurements(data_source, grid, measurements):
    auc_records_measurement = {}
    for measure in tqdm(combinations(measurements, 1)):
        auc_records_measurement[measure] = {}
        final_df_data = final_df(list(measure), data_source)
        for _ in tqdm(range(len(grid))):
            params = grid[_]
            learning_rate = params['learning_rate']
            hidden_dim = params['hidden_dim']
            num_epochs = params['num_epochs']
            val_years, auc_for_each_year = run_model(list(measure), final_df_data, hidden_dim, learning_rate, num_epochs)
            param_key = str(params)
            auc_records_measurement[measure][param_key] = dict(zip(val_years, auc_for_each_year))
    return auc_records_measurement

def process_dataset_run_all(data_source, grid, channels):
    results = {}
    final_df_data = final_df(channels, data_source)
    for _ in tqdm(range(len(grid))):
        params = grid[_]
        learning_rate = params['learning_rate']
        hidden_dim = params['hidden_dim']
        num_epochs = params['num_epochs']
        val_years, auc_for_each_year = run_model(channels, final_df_data, hidden_dim, learning_rate, num_epochs)
        param_key = str(params)
        results[param_key] = dict(zip(val_years, auc_for_each_year))
    return results

# ---------------------------
# 6. Execute Processing Based on Job Array ID
# ---------------------------
# The environment variable SLURM_ARRAY_TASK_ID is used to choose which dataset to process.
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

if job_id == 0:
    data_source = networkscores_match
    results = process_dataset_run_all(data_source, grid, channels)
    suffix = 'all'
    file_path_all = os.path.join(RESULT_DIR, f'results_match_{suffix}_all.pkl')
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 1:
    data_source = networkscores_match
    suffix = 'all'
    results_channels = process_dataset_channels(data_source, grid, channels)
    file_path_channels = os.path.join(RESULT_DIR, f'results_match_{suffix}_channels.pkl')
    with open(file_path_channels, 'wb') as file:
        pickle.dump(results_channels, file)
elif job_id == 2:
    data_source = networkscores_match
    results_measurements = process_dataset_measurements(data_source, grid, measurements)
    suffix = 'all'
    file_path_measurements = os.path.join(RESULT_DIR, f'results_match_{suffix}_measurements.pkl')
    with open(file_path_measurements, 'wb') as file:
        pickle.dump(results_measurements, file)
elif job_id == 3:
    data_source = networkscores_male
    results = process_dataset_run_all(data_source, grid, channels)
    suffix = 'male'
    file_path_all = os.path.join(RESULT_DIR, f'results_match_{suffix}_all.pkl')
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 4:
    data_source = networkscores_male
    results_channels = process_dataset_channels(data_source, grid, channels)
    suffix = 'male'
    file_path_channels = os.path.join(RESULT_DIR, f'results_match_{suffix}_channels.pkl')
    with open(file_path_channels, 'wb') as file:
        pickle.dump(results_channels, file)
elif job_id == 5:
    data_source = networkscores_male
    results_measurements = process_dataset_measurements(data_source, grid, measurements)
    suffix = 'male'
    file_path_measurements = os.path.join(RESULT_DIR, f'results_match_{suffix}_measurements.pkl')
    with open(file_path_measurements, 'wb') as file:
        pickle.dump(results_measurements, file)
elif job_id == 6:
    data_source = networkscores_female
    results = process_dataset_run_all(data_source, grid, channels)
    suffix = 'female'
    file_path_all = os.path.join(RESULT_DIR, f'results_match_{suffix}_all.pkl')
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 7:
    data_source = networkscores_female
    results_channels = process_dataset_channels(data_source, grid, channels)
    suffix = 'female'
    file_path_channels = os.path.join(RESULT_DIR, f'results_match_{suffix}_channels.pkl')
    with open(file_path_channels, 'wb') as file:
        pickle.dump(results_channels, file)
elif job_id == 8:
    data_source = networkscores_female
    results_measurements = process_dataset_measurements(data_source, grid, measurements)
    suffix = 'female'
    file_path_measurements = os.path.join(RESULT_DIR, f'results_match_{suffix}_measurements.pkl')
    with open(file_path_measurements, 'wb') as file:
        pickle.dump(results_measurements, file)
else:
    raise ValueError("Invalid job ID")
