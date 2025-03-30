"""
fine_tune_gender.py

This script fine-tunes an LSTM model on matched candidate pairs using updated network scores.
It is designed for gender-based analysis — the same script is used for both female and male candidates.
Set the environment variable GENDER to either "female" or "male" to choose which dataset to use.
The script uses SLURM_ARRAY_TASK_ID to control sub-experiments (e.g., all channels, single channel, measurements).

Usage:
    export GENDER=female
    export DATA_DIR="/your/data/path"
    export OUTPUT_DIR="/your/output/path"
    export DB_PATH="/your/db/path.db"
    python fine_tune_gender.py
"""

import os
import re
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from fuzzywuzzy import fuzz

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset

import sqlite3
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Set R_HOME if needed (this should be set appropriately on your system)
os.environ['R_HOME'] = os.environ.get('R_HOME', '/path/to/R')

# ---------------------
# Configuration (All sensitive paths are set via environment variables)
# ---------------------
DATA_DIR = os.environ.get('DATA_DIR', '/path/to/data')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/path/to/results/')
DB_PATH = os.environ.get('DB_PATH', '/path/to/db.db')

# Use GENDER environment variable to decide which CSV to load for network scores.
GENDER = os.environ.get("GENDER", "female").lower()  # default to "female" if not set
if GENDER == "female":
    networkscores = pd.read_csv(os.path.join(DATA_DIR, "multi_indexed_df_reverse_non_10_years_lagged_byhand_person_female.csv"), index_col=False)
elif GENDER == "male":
    networkscores = pd.read_csv(os.path.join(DATA_DIR, "multi_indexed_df_reverse_non_10_years_lagged_byhand_person_male.csv"), index_col=False)
else:
    raise ValueError("GENDER must be 'female' or 'male'.")

# The output folder for results is chosen based on gender.
OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f"{GENDER}_results_match/")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load other data files (these paths are hidden via DATA_DIR)
static_feature = pd.read_csv(os.path.join(DATA_DIR, "static_features.csv"), index_col=False)
df_background = pd.read_csv(os.path.join(DATA_DIR, "background_director_after_2010_update_v2.csv"), index_col=False)
directors_career_path = pd.read_csv(os.path.join(DATA_DIR, "directors_career_path_trail_with_first_appointment.csv"), index_col=False)
y = pd.read_csv(os.path.join(DATA_DIR, "target_value.csv"), index_col=False)

# Filter career paths to include only those with available network scores
directors_career_path = directors_career_path[directors_career_path.DirectorID.isin(networkscores.DirectorID.unique())]

# ---------------------
# Data Cleaning & Preprocessing
# ---------------------
directors_career_path['ProcessedRoleName'] = directors_career_path['RoleName'].apply(lambda x: re.split(r"/|-", x)[0].rstrip())

def clean_role_name(role_name, position):
    role_name = re.sub(r'Vice\s+President', 'VP', role_name, flags=re.IGNORECASE)
    role_name = re.sub(r'\bChief.*?Officer\b', 'C-suite', role_name, flags=re.IGNORECASE)
    role_name = re.sub(r'\bC(\w+)O\b', r'C-suite', role_name, flags=re.IGNORECASE)
    positions = ["director", "manager", "Executive", "Officer", "Supervisor", "CEO", "CFO",
                 "COO", "Chairman", "Co", "Commissioner", "Secretary", "Dean", "Deputy", "Head",
                 "President", "ED", "VP", "Counsel", "Treasurer", "Minister", "MD", "Professor",
                 "Faculty", "Instructor", "Advisor", "Council", "Member", "NED", "Clerk", "Partner", "Representative"]
    pattern = r"\b(?:{})\b".format("|".join(positions))
    preserved_terms = ["Assistant", "Associate", "Chief", "Senior", "Vice", "Adjunct", "Acting", "Interim"]
    matches = re.findall(pattern, role_name, flags=re.IGNORECASE)
    if matches:
        preserved = [term for term in preserved_terms if term in role_name]
        role_name = " ".join(preserved + matches)
    return role_name

directors_career_path['ProcessedRoleName'] = [
    clean_role_name(role_name, position) 
    for role_name, position in zip(directors_career_path['ProcessedRoleName'], directors_career_path['DirectorPosition'])
]

def categorize_role(role):
    roles = ['VP','Supervisory Director','C-suite','Manager','Director','Various Positions',
             'President','Executive Director','Managing Director','Executive','Partner','Head',
             'Advisor','Associate','Officer','Senior VP','Founder']
    for r in roles:
        if fuzz.ratio(r, role) >= 50:
            return r
    return 'Other'

directors_career_path['ProcessedRoleName'] = directors_career_path['ProcessedRoleName'].apply(categorize_role)
directors_career_path.loc[directors_career_path['ProcessedRoleName'] == 'ED', 'ProcessedRoleName'] = 'Executive Director'
directors_career_path.loc[directors_career_path['ProcessedRoleName'] == 'MD', 'ProcessedRoleName'] = 'Managing Director'

# ---------------------
# Load additional organizational data from database
# ---------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

query1 = f"""
SELECT * FROM org_summary_brdlevel
WHERE BoardID IN {tuple(directors_career_path.CompanyID.unique())}
"""
df_org = pd.read_sql_query(query1, conn)
df_org.drop_duplicates(subset=['BoardID'], inplace=True, keep='last')
df_org.loc[df_org.Sector == '', 'Sector'] = 'Other'
df_org.loc[df_org.Index == '', 'Index'] = 'No Index'

directors_career_path = directors_career_path.merge(
    df_org[['BoardID','Sector','OrgType','Index']],
    how='left', left_on='CompanyID', right_on='BoardID'
).drop(columns=['BoardID'])

for col in ['Sector', 'Index', 'OrgType']:
    directors_career_path[col].fillna('Other', inplace=True)
directors_career_path["ProcessedRoleName"] = directors_career_path["ProcessedRoleName"].astype(str).str.replace("-", "_")
directors_career_path["Index"] = directors_career_path["Index"].astype(str).str.replace("-", "_")

directors_career_path_sorted = directors_career_path.sort_values(by=['DirectorID', 'DateStartRole'], ascending=[True, False])
top_15_records = directors_career_path_sorted.groupby('DirectorID').head(15)
final_sorted_directors_career_path = top_15_records.sort_values(by=['DirectorID', 'DateStartRole'], ascending=[True, True])

def concatenate_paths(x):
    return '-'.join(x)

career_paths = directors_career_path.groupby('DirectorID')['ProcessedRoleName'].apply(concatenate_paths)
sector_paths = directors_career_path.groupby('DirectorID')['Sector'].apply(concatenate_paths)
orgtype_paths = directors_career_path.groupby('DirectorID')['RowType'].apply(concatenate_paths)
index_paths = directors_career_path.groupby('DirectorID')['Index'].apply(concatenate_paths)

result_df = pd.DataFrame({
    'DirectorID': career_paths.index,
    'CareerPath': career_paths.values,
    'SectorPath': sector_paths.values,
    'OrgTypePath': orgtype_paths.values,
    'IndexPath': index_paths.values
})

def clean_non_utf8(text):
    return text.encode('ascii', errors='ignore').decode('utf-8')

for col in result_df.select_dtypes(include=[object]):
    result_df[col] = result_df[col].apply(clean_non_utf8)

# ---------------------
# Use TraMineR (via rpy2) to compute sequence dissimilarities
# ---------------------
pandas2ri.activate()
utils = importr('utils')
TraMineR = importr('TraMineR')
r_df = pandas2ri.py2rpy(result_df)

ro.r('''
library(TraMineR)
create_career_seq <- function(df_r) {
    df_r$CareerPath <- as.factor(df_r$CareerPath)
    seqdef(df_r[, "CareerPath"])
}
create_sector_seq <- function(df_r) {
    df_r$SectorPath <- as.factor(df_r$SectorPath)
    seqdef(df_r[, "SectorPath"])
}
create_orgtype_seq <- function(df_r) {
    df_r$OrgTypePath <- as.factor(df_r$OrgTypePath)
    seqdef(df_r[, "OrgTypePath"])
}
create_index_seq <- function(df_r) {
    df_r$IndexPath <- as.factor(df_r$IndexPath)
    seqdef(df_r[, "IndexPath"])
}
calculate_dissimilarity_career <- function(seq_career) {
    sm_career <- seqsubm(seq_career, method = "TRATE")
    seqdist(seq_career, method = "OM", sm = sm_career)
}
calculate_dissimilarity_sector <- function(seq_sector) {
    sm_sector <- seqsubm(seq_sector, method = "TRATE")
    seqdist(seq_sector, method = "OM", sm = sm_sector)
}
calculate_dissimilarity_orgtype <- function(seq_orgtype) {
    sm_orgtype <- seqsubm(seq_orgtype, method = "TRATE")
    seqdist(seq_orgtype, method = "OM", sm = sm_orgtype)
}
calculate_dissimilarity_index <- function(seq_index) {
    sm_index <- seqsubm(seq_index, method = "TRATE")
    seqdist(seq_index, method = "OM", sm = sm_index)
}
''')

career_seq = ro.globalenv['create_career_seq'](r_df)
sector_seq = ro.globalenv['create_sector_seq'](r_df)
orgtype_seq = ro.globalenv['create_orgtype_seq'](r_df)
index_seq = ro.globalenv['create_index_seq'](r_df)

diss_career = ro.globalenv['calculate_dissimilarity_career'](career_seq)
diss_sector = ro.globalenv['calculate_dissimilarity_sector'](sector_seq)
diss_orgtype = ro.globalenv['calculate_dissimilarity_orgtype'](orgtype_seq)
diss_index = ro.globalenv['calculate_dissimilarity_index'](index_seq)

def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

normalized_diss_career = normalize_matrix(diss_career)
normalized_diss_sector = normalize_matrix(diss_sector)
normalized_diss_orgtype = normalize_matrix(diss_orgtype)
normalized_diss_index = normalize_matrix(diss_index)

encoder = OneHotEncoder(sparse=False)
nationality_encoded = encoder.fit_transform(df_background[['Region']])
categories = encoder.categories_[0]
feature_names = ['Nationality_' + cat for cat in categories]
static_features_encoded = pd.DataFrame(nationality_encoded, columns=feature_names)
from scipy.spatial.distance import pdist, squareform
dissimilarity_nationality = squareform(pdist(static_features_encoded, metric='hamming'))
normalized_diss_region = normalize_matrix(dissimilarity_nationality)

binary_features = df_background[['Has_MBA', 'Has_PhD', 'Has_JD']]
dissimilarity_binary = squareform(pdist(binary_features, metric='hamming'))
normalized_diss_degree = normalize_matrix(dissimilarity_binary)

experience = df_background[['experience']]
dissimilarity_experience = squareform(pdist(experience, metric='euclidean'))
normalized_diss_experience = normalize_matrix(dissimilarity_experience)

combined_diss_matrix = (normalized_diss_career + normalized_diss_sector +
                        normalized_diss_orgtype + normalized_diss_index +
                        normalized_diss_region + normalized_diss_degree +
                        normalized_diss_experience)

result_df = result_df.merge(final_sorted_directors_career_path[['DirectorID','Gender']].drop_duplicates(), how='left', on='DirectorID')
females = result_df[result_df['Gender'] == 1].index
males = result_df[result_df['Gender'] == 0].index

# ---------------------
# Matching
# ---------------------
caliper = 0.15
matches = {}
matched_males = set()
matched_females = set()
for f in females:
    if f not in matched_females:
        male_distances = combined_diss_matrix[f, males] / 7
        eligible_males = [(idx, m) for idx, m in enumerate(males) if (male_distances[idx] <= caliper and m not in matched_males)]
        if eligible_males:
            sorted_eligible = sorted(eligible_males, key=lambda x: male_distances[x[0]])
            nearest_male = sorted_eligible[0][1]
            matches[f] = nearest_male
            matched_males.add(nearest_male)
            matched_females.add(f)

female_indices = [f for f, m in matches.items() if m is not None]
female_data = result_df.loc[female_indices].drop_duplicates()
male_indices = [m for f, m in matches.items() if m is not None]
male_data = result_df.loc[male_indices].drop_duplicates()

# (Optionally, these datasets can be used later for matching-based fine-tuning.)
networkscores_male = networkscores[networkscores['DirectorID'].isin(male_data.DirectorID.unique())]
networkscores_female = networkscores[networkscores['DirectorID'].isin(female_data.DirectorID.unique())]
networkscores_match = networkscores[networkscores['DirectorID'].isin(list(set(female_data.DirectorID.unique()).union(set(male_data.DirectorID.unique()))))]

def final_df_func(channels, networkscores):
    condition = networkscores['Score'].str.contains('|'.join(channels))
    filtered_networkscores = networkscores[condition]
    melted_df = filtered_networkscores.melt(id_vars=['DirectorID', 'Score'], var_name='year', value_name='network_score')
    pivot_df = melted_df.pivot_table(index=['DirectorID', 'year'], columns='Score', values='network_score', aggfunc='first').reset_index()
    pivot_df['network_scores'] = pivot_df[pivot_df.columns[2:]].apply(list, axis=1)
    pivot_df = pivot_df[pivot_df['network_scores'].apply(lambda x: not all(pd.isnull(v) for v in x))]
    final_df = pivot_df[['DirectorID', 'year', 'network_scores']].copy()
    final_df.rename(columns={'DirectorID': 'individual', 'year': 'year'}, inplace=True)
    final_df['director_appointment_next_year'] = 0
    results = pd.read_csv(os.path.join(DATA_DIR, "DirectorTimeRecord_updated.csv"), index_col=None)
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

# ---------------------
# LSTM Model and Training
# ---------------------
class LSTMModel(nn.Module):
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
        final_df_data = final_df_func(list(ch), data_source)
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
        final_df_data = final_df_func(list(measure), data_source)
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
    final_df_data = final_df_func(channels, data_source)
    for _ in tqdm(range(len(grid))):
        params = grid[_]
        learning_rate = params['learning_rate']
        hidden_dim = params['hidden_dim']
        num_epochs = params['num_epochs']
        val_years, auc_for_each_year = run_model(channels, final_df_data, hidden_dim, learning_rate, num_epochs)
        param_key = str(params)
        results[param_key] = dict(zip(val_years, auc_for_each_year))
    return results

# ---------------------
# Job Array Branching using SLURM_ARRAY_TASK_ID
# ---------------------
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

# For IDs 0-10 we use networkscores_match (all scores), then for single-channel and measurement breakdown,
# and finally for processing the matched female (if GENDER is female) or male (if GENDER is male) dataset.
if job_id == 0:
    data_source = networkscores  # using the gender-specific networkscores loaded above
    suffix = 'all'
    results = process_dataset_run_all(data_source, grid, channels)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_all.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 1:
    data_source = networkscores
    channel_single = channels[0:1]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 2:
    data_source = networkscores
    channel_single = channels[1:2]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 3:
    data_source = networkscores
    channel_single = channels[2:3]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 4:
    data_source = networkscores
    channel_single = channels[3:4]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 5:
    data_source = networkscores
    channel_single = channels[4:5]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 6:
    data_source = networkscores
    measurements_single = measurements[0:1]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 7:
    data_source = networkscores
    measurements_single = measurements[1:2]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 8:
    data_source = networkscores
    measurements_single = measurements[2:3]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 9:
    data_source = networkscores
    measurements_single = measurements[3:4]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 10:
    data_source = networkscores
    measurements_single = measurements[4:5]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
# For job IDs 11 and beyond, process the matched dataset.
elif job_id == 11:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_run_all(data_source, grid, channels)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_all.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 12:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    channel_single = channels[0:1]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_channels(data_source, grid, channel_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 13:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    channel_single = channels[1:2]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_channels(data_source, grid, channel_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 14:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    channel_single = channels[2:3]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_channels(data_source, grid, channel_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 15:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    channel_single = channels[3:4]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_channels(data_source, grid, channel_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 16:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    channel_single = channels[4:5]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_channels(data_source, grid, channel_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 17:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    measurements_single = measurements[0:1]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_measurements(data_source, grid, measurements_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 18:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    measurements_single = measurements[1:2]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_measurements(data_source, grid, measurements_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 19:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    measurements_single = measurements[2:3]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_measurements(data_source, grid, measurements_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 20:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    measurements_single = measurements[3:4]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_measurements(data_source, grid, measurements_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 21:
    data_source = networkscores_female if GENDER == 'female' else networkscores_male
    measurements_single = measurements[4:5]
    suffix = 'female' if GENDER == 'female' else 'male'
    results = process_dataset_measurements(data_source, grid, measurements_single)
    file_path_all = os.path.join(OUTPUT_FOLDER, f"results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl")
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
else:
    raise ValueError("Invalid job ID")
