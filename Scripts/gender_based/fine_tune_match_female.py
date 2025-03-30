import pandas as pd
import sqlite3
import time
import numpy as np
from tqdm import tqdm
import math
import re
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist, squareform
import pickle
from sklearn.model_selection import ParameterGrid

# Clean up RoleName
from fuzzywuzzy import fuzz
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn, optim

from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from torchtext.vocab import Vocab
from itertools import chain,combinations
# from sklearn.preprocessing import LabelEncoder
import torchtext
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report



import os
os.environ['R_HOME'] = '/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/r/4.3.1/lib64/R'

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random_seed = 3402
set_seed(random_seed)

# static features
static_feature = pd.read_csv('~/projects/def-cbravo/jetzhou/SocialNetworkData/Data/static_features.csv', index_col = False)
# df_background = pd.read_csv('~/projects/def-cbravo/jetzhou/SocialNetworkData/Data/background_director_after_2010_update.csv', index_col = False)
df_background = pd.read_csv('~/projects/def-cbravo/jetzhou/SocialNetworkData/Data/background_director_after_2010_update_v2.csv', index_col = False)

# Career Path (without first appointment)
directors_career_path = pd.read_csv("~/projects/def-cbravo/jetzhou/SocialNetworkData/Data/directors_career_path_trail_with_first_appointment.csv", index_col = False)

y = pd.read_csv('~/projects/def-cbravo/jetzhou/SocialNetworkData/Data/target_value.csv',index_col=False) #25808 directors/SMs

# networkscores=pd.read_csv('~/projects/defcbravo/jetzhou/SocialNetworkData/Data/multi_indexed_df_reverse_non_10_years_lagged_byhand.csv',index_col=False)
networkscores=pd.read_csv('~/projects/def-cbravo/jetzhou/SocialNetworkData/Data/multi_indexed_df_reverse_non_10_years_lagged_byhand_person_female.csv',index_col=False)
directors_career_path = directors_career_path[directors_career_path.DirectorID.isin(networkscores.DirectorID.unique())]

# Remove text after "/" or "-" and eliminate trailing whitespace in the RoleName column
directors_career_path['ProcessedRoleName'] = directors_career_path['RoleName'].apply(lambda x: re.split(r"/|-", x)[0].rstrip())
# processed_unique_count = directors_career_path['ProcessedRoleName'].nunique()
# print(f"Number of fewer unique RoleNames: {processed_unique_count}")

def clean_role_name(role_name, position):
    role_name = re.sub(r'Vice\s+President', 'VP', role_name, flags=re.IGNORECASE)
    
    # Replace role names starting with "Chief" and ending with "Officior" with "C-cuite"
    role_name = re.sub(r'\bChief.*?Officer\b', 'C-suite', role_name, flags=re.IGNORECASE)

    # Replace individual instances of "C+something+O" with "C-cuite"
    role_name = re.sub(r'\bC(\w+)O\b', r'C-suite', role_name, flags=re.IGNORECASE)
    positions = ["director", "manager", "Executive", "Officer", "Supervisor", "CEO", "CFO", "COO", "Chairman", "Co", "Commissioner", "Secretary", "Dean", "Deputy", "Head", "President", "ED", "VP", "Counsel", "Treasurer", "Minister", "VP", "MD", "Professor", "Faculty", "Instructor", "Advisor", "Council", "Member", "NED", "Clerk", "Partner", "Representative"]
    pattern = r"\b(?:{})\b".format("|".join(positions))
    preserved_terms = ["Assistant", "Associate", "Chief", "Senior", "Vice", "Adjunct", "Acting", "Interim"]
    # Find matches in the role name
    matches = re.findall(pattern, role_name, flags=re.IGNORECASE)

    # Check if there are matches
    if matches:
        # Extract the preserved terms from the role name
        preserved = [term for term in preserved_terms if term in role_name]

        # Concatenate the preserved terms and the matched positions
        role_name = " ".join(preserved + matches)
    return role_name

directors_career_path['ProcessedRoleName'] = [clean_role_name(role_name, position) for role_name, position in zip(directors_career_path['ProcessedRoleName'], directors_career_path['DirectorPosition'])]
# print(directors_career_path['ProcessedRoleName'].nunique())

# Cleaning the roles, removing titles, punctuations and normalizing the case
from collections import Counter
clean_roles = []
for role in directors_career_path['ProcessedRoleName']:
    clean_roles.append(role)

# Counting the occurrences of each role
role_counter = Counter(clean_roles)

# Getting the 20 most common roles
# most_common_roles = role_counter.most_common(40)

# print(*most_common_roles)

directors_career_path.loc[directors_career_path['ProcessedRoleName'] == 'ED', 'ProcessedRoleName'] = 'Executive Director'
directors_career_path.loc[directors_career_path['ProcessedRoleName'] == 'MD', 'ProcessedRoleName'] = 'Managing Director'

roles = ['VP','Supervisory Director','C-suite','Manager','Director','Various Positions',
        'President','Executive Director','Managing Director','Executive','Partner','Head','Advisor','Associate','Officer','Senior VP','Founder',]
def categorize_role(role):
    for r in roles:
        if fuzz.ratio(r, role) >= 50: # 50 is our similarity threshold
            return r
    return 'Other'
directors_career_path['ProcessedRoleName'] = directors_career_path['ProcessedRoleName'].apply(categorize_role)
# directors_career_path['ProcessedRoleName'].nunique()

home_dir = '/home/jetzhou'

# db_path = os.path.join(home_dir, 'projects/def-cbravo/jetzhou/SocialNetworkData/base.db')
db_path = os.path.join(home_dir, 'projects/def-cbravo/jetzhou/base.db')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

query1 = """
SELECT * FROM org_summary_brdlevel
WHERE BoardID IN {}
""".format(tuple(directors_career_path.CompanyID.unique()))
df = pd.read_sql_query(query1, conn)

df.drop_duplicates(subset=['BoardID'], inplace=True, keep='last')
df.loc[df.Sector == '', 'Sector'] = 'Other'
df.loc[df.Index == '', 'Index'] = 'No Index'

directors_career_path = directors_career_path.merge(df[['BoardID','Sector','OrgType','Index']], how = 'left', left_on= 'CompanyID', right_on= 'BoardID').drop(columns = ['BoardID'])
directors_career_path['OrgType'].isna().sum()

directors_career_path['Sector'].fillna('Other', inplace=True)
directors_career_path['Index'].fillna('No Index', inplace=True)
directors_career_path['OrgType'].fillna('Other', inplace=True)

directors_career_path["ProcessedRoleName"] = directors_career_path["ProcessedRoleName"].astype(str).str.replace("-", "_")
directors_career_path["Index"] = directors_career_path["Index"].astype(str).str.replace("-", "_")

# Sort by IndividualID and StartDate in descending order
directors_career_path_sorted = directors_career_path.sort_values(by=['DirectorID', 'DateStartRole'], ascending=[True, False])

# Group by IndividualID and keep only the top 15 records for each individual
top_15_records = directors_career_path_sorted.groupby('DirectorID').head(15)

# Resort the filtered records by StartDate in ascending order
final_sorted_directors_career_path = top_15_records.sort_values(by=['DirectorID', 'DateStartRole'], ascending=[True, True])

# Sort the DataFrame by ID and StartDate
data = final_sorted_directors_career_path

# Function to concatenate job titles and companies
concatenate_paths = lambda x: '-'.join(x)

# Group by ID and apply the concatenation function
career_paths = data.groupby('DirectorID')['ProcessedRoleName'].apply(concatenate_paths)
# company_paths = data.groupby('DirectorID')['CompanyName'].apply(concatenate_paths)
sector_paths = data.groupby('DirectorID')['Sector'].apply(concatenate_paths)
orgtype_paths = data.groupby('DirectorID')['RowType'].apply(concatenate_paths)
index_paths = data.groupby('DirectorID')['Index'].apply(concatenate_paths)

# Join the paths back to a DataFrame with IDs
result_df = pd.DataFrame({
    'DirectorID': career_paths.index,
    'CareerPath': career_paths.values,
    # 'CompanyPath': company_paths.values,
    'SectorPath': sector_paths.values,
    'OrgTypePath': orgtype_paths.values,
    'IndexPath': index_paths.values
})

# result_df.head()

def clean_non_utf8(text):
    return text.encode('ascii', errors='ignore').decode('utf-8')

# Apply the cleaning function to all string columns
for col in result_df.select_dtypes(include=[object]):  # object usually indicates string-like columns
    result_df[col] = result_df[col].apply(clean_non_utf8)

# result_df.head()



# Activate the pandas2ri interface
pandas2ri.activate()

# Import R's utility package
utils = importr('utils')

# Install TraMineR if not already installed
# utils.install_packages('TraMineR')

# Import TraMineR
TraMineR = importr('TraMineR')

df = pandas2ri.py2rpy(result_df)

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

# Create sequence objects
career_seq = ro.globalenv['create_career_seq'](df)
sector_seq = ro.globalenv['create_sector_seq'](df)
orgtype_seq = ro.globalenv['create_orgtype_seq'](df)
index_seq = ro.globalenv['create_index_seq'](df)

# Calculate dissimilarities
diss_career = ro.globalenv['calculate_dissimilarity_career'](career_seq)
diss_sector = ro.globalenv['calculate_dissimilarity_sector'](sector_seq)
diss_orgtype = ro.globalenv['calculate_dissimilarity_orgtype'](orgtype_seq)
diss_index = ro.globalenv['calculate_dissimilarity_index'](index_seq)

def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

normalized_diss_career = normalize_matrix(diss_career)
normalized_diss_sector = normalize_matrix(diss_sector)
normalized_diss_orgtype = normalize_matrix(diss_orgtype)
normalized_diss_index = normalize_matrix(diss_index)

# One-hot encode the 'Nationality' column
encoder = OneHotEncoder(sparse=False)
nationality_encoded = encoder.fit_transform(df_background[['Region']])

# Manually create column names for the one-hot encoded features
categories = encoder.categories_[0]
feature_names = ['Nationality_' + cat for cat in categories]
static_features_encoded = pd.DataFrame(nationality_encoded, columns=feature_names)
dissimilarity_nationality = squareform(pdist(static_features_encoded, metric='hamming'))
normalized_diss_region = normalize_matrix(dissimilarity_nationality)

# Assuming df is your DataFrame with the binary features
binary_features = df_background[['Has_MBA', 'Has_PhD', 'Has_JD']]

# Calculate dissimilarities for binary features
dissimilarity_binary = squareform(pdist(binary_features, metric='hamming'))
normalized_diss_degree = normalize_matrix(dissimilarity_binary)

# Experience dissimilarity
experience = df_background[['experience']]
dissimilarity_experience = squareform(pdist(experience, metric='euclidean'))
normalized_diss_experience = normalize_matrix(dissimilarity_experience)


combined_diss_matrix = normalized_diss_career + normalized_diss_sector + normalized_diss_orgtype + normalized_diss_index + normalized_diss_region + normalized_diss_degree + normalized_diss_experience

result_df = result_df.merge(final_sorted_directors_career_path[['DirectorID','Gender']].drop_duplicates(), how = 'left', on = 'DirectorID')

females = result_df[result_df['Gender'] == 1].index
males = result_df[result_df['Gender'] == 0].index

######################################################################
# Set a caliper (maximum allowed dissimilarity)
caliper = 0.15  # Adjust this based on your specific case #0.7/7 or 0.1/1; 1.05/7 or 0.15/1

matches = {}
matched_males = set()  # Track males already matched
matched_females = set()  # Track females already matched

for f in females:
    if f not in matched_females:
        # Calculate distances for this female to all males
        male_distances = combined_diss_matrix[f, males]/7
        
        # Find eligible males within the caliper and not already matched
        eligible_males = [(idx, m) for idx, m in enumerate(males) if (male_distances[idx] <= caliper and m not in matched_males)]

        if eligible_males:
            # Sort eligible males by distance
            sorted_eligible_males = sorted(eligible_males, key=lambda x: male_distances[x[0]])
            
            # Choose the nearest eligible male
            nearest_male = sorted_eligible_males[0][1]

            matches[f] = nearest_male
            matched_males.add(nearest_male)
            matched_females.add(f)
###########################################################################
# Extract female indices from the matches
female_indices = [f for f, m in matches.items() if m is not None]

# Extract female data from the original DataFrame
female_data = result_df.loc[female_indices].drop_duplicates()

# Extract male indices from the matches
male_indices = [m for f, m in matches.items() if m is not None]

# Extract male data from the original DataFrame
male_data = result_df.loc[male_indices].drop_duplicates()

networkscores_male = networkscores[networkscores['DirectorID'].isin(male_data.DirectorID.unique())]
networkscores_female = networkscores[networkscores['DirectorID'].isin(female_data.DirectorID.unique())]
networkscores_match = networkscores[networkscores['DirectorID'].isin(list(set(female_data.DirectorID.unique()).union(set(male_data.DirectorID.unique()))))]

def final_df(channels, networkscores):
    # Filter the networkscores DataFrame based on the channels
    condition = networkscores['Score'].str.contains('|'.join(channels))
    filtered_networkscores = networkscores[condition]

    # Melt and Pivot DataFrame
    melted_df = filtered_networkscores.melt(id_vars=['DirectorID', 'Score'], var_name='year', value_name='network_score')
    pivot_df = melted_df.pivot_table(index=['DirectorID', 'year'], columns='Score', values='network_score', aggfunc='first').reset_index()

    # Convert the network measurement columns into a single list column
    pivot_df['network_scores'] = pivot_df[pivot_df.columns[2:]].apply(list, axis=1)

    # Filter out rows where all network scores are NaN
    pivot_df = pivot_df[pivot_df['network_scores'].apply(lambda x: not all(pd.isnull(v) for v in x))]

    # Keep only the required columns
    final_df = pivot_df[['DirectorID', 'year', 'network_scores']].copy()

    # Rename the columns to match the desired format
    final_df.rename(columns={'DirectorID': 'individual', 'year': 'year'}, inplace=True)

    # Add the 'director_appointment_next_year' column as missing (NaN)
    final_df['director_appointment_next_year'] = 0

    results = pd.read_csv("~/projects/def-cbravo/jetzhou/SocialNetworkData/Data/DirectorTimeRecord_updated.csv",index_col=None)
    results['Latest_End'] = results['Latest_End'].astype(int)
    # Create a dictionary from the director_time_record DataFrame for easy lookups
    director_end_year_dict = results.set_index('DirectorID')[['Latest_End', 'director_dummy']].T.to_dict()
    # Convert the 'year' column in the final DataFrame to integer type
    final_df['year'] = final_df['year'].astype(int)

    # Create a dictionary from the director_time_record DataFrame for easy lookups
    director_end_year_dict = results.set_index('DirectorID')[['Latest_End', 'director_dummy']].T.to_dict()

    # Define a function to determine the value of 'director_appointment_next_year'
    def fill_director_appointment(row):
        director_id = row['individual']
        year = row['year']
        if director_id in director_end_year_dict and director_end_year_dict[director_id]['director_dummy'] == 1:
            if year == director_end_year_dict[director_id]['Latest_End']:
                return 1
        return 0

    # Apply the function to the final DataFrame
    final_df['director_appointment_next_year'] = final_df.apply(fill_director_appointment, axis=1)
    return final_df

# final_df_male = final_df(channels, networkscores_male)
# final_df_female = final_df(channels, networkscores_female)
# final_df_match=final_df(channels, networkscores_match)

class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, dropout=0.5, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x, mask):
            out, _ = self.lstm(x)
            out = out * mask.unsqueeze(-1)  # Apply the mask to the output tensor
            out = self.fc(out[:, -1, :])  # Use the last output only (corresponding to the last non-padded input)
            return out.squeeze()

def run_model(channels, final_df, hidden_dim,learning_rate, num_epochs):
    # Define your cross-validation parameters
    n_splits = 12  # Number of splits
    window_size = 10  # Number of years for the training window
    # gap = 1  # Gap between training and validation data

    # Get unique individuals and years
    individuals = final_df['individual'].unique()
    years = sorted(final_df['year'].unique())

    # Define the front padding function
    def front_pad_sequences(sequences):
        # Get the length of each sequence
        lengths = [len(seq) for seq in sequences]
        # Get the maximum length
        max_length = max(lengths)
        # Create a tensor of zeros with the shape of (max_length, len(sequences), num_network_scores)
        padded_sequences = torch.zeros((len(sequences), max_length, sequences[0].shape[-1]))
        mask = torch.zeros((len(sequences), max_length))
        # Fill the padded_sequences tensor with the sequences
        for i, seq in enumerate(sequences):
            length = len(seq)
            padded_sequences[i, -length:] = seq
            mask[i, -length:] = 1
        return padded_sequences, mask
    
    # Parameters
    input_dim = len(channels) * 5
    hidden_dim = hidden_dim
    learning_rate = learning_rate
    num_epochs = num_epochs

    val_auc = []
    val_years = []
    # print("Setup ready")

    for i in range(n_splits):
        start_year = years[0]
        end_year = years[i + window_size-1]
        validation_year = years[i + window_size]
        val_years.append(validation_year)
        # print(f"start_year:{start_year}, end_year:{end_year}, validation_year:{validation_year}")
        # start_years.append(start_year)
        # end_years.append(end_year)
        # print(f'Start_year: {start_year}, End_year: {end_year}, Validation_year: {validation_year}')
        
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
        # print(f'X_train shape : {X_train[-1].shape}')
        # print(f'mask_train shape : {mask_train[-1].shape}')
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
            # print(f'X_val shape : {X_val[-1].shape}')
            # print(f'mask_val shape : {mask_val[-1].shape}')
            y_val = torch.tensor(y_val, dtype=torch.float32)

            model = LSTMModel(input_dim, hidden_dim)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)

            for _ in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train, mask_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Validation metrics
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

# param_grid = {
#     'learning_rate': [0.1,0.05],
#     'hidden_dim': [64],
#     'num_epochs': [50]
# }

# Create a grid of hyperparameters
grid = ParameterGrid(param_grid)

def process_dataset_channels(data_source, grid, channels):
    auc_records_channel = {}
    for ch in tqdm(combinations(channels, 1)):
        auc_records_channel[ch] = {}
        final_df_data = final_df(list(ch), data_source)
        # print(final_df_data.head())
        # print(f"length is {len(final_df_data.network_scores[0])}")
        for _ in tqdm(range(len(grid))):
            params = grid[_]
            learning_rate = params['learning_rate']
            hidden_dim = params['hidden_dim']
            num_epochs = params['num_epochs']
            # print(f"Combination: {ch}")
            # print(f"Paras: {params}")
            val_years, auc_for_each_year = run_model(list(ch), final_df_data, hidden_dim, learning_rate, num_epochs)
            param_key = str(params)
            auc_records_channel[ch][param_key] = dict(zip(val_years, auc_for_each_year))
    return auc_records_channel

def process_dataset_measurements(data_source, grid, measurements):
    auc_records_measurement = {}
    for measure in tqdm(combinations(measurements, 1)):
        auc_records_measurement[measure] = {}
        final_df_data = final_df(list(measure), data_source)
        # print(final_df_data.head())
        # print(f"length is {len(final_df_data.network_scores[0])}")
        for _ in tqdm(range(len(grid))):
            params = grid[_]
            learning_rate = params['learning_rate']
            hidden_dim = params['hidden_dim']
            num_epochs = params['num_epochs']
            # print(f"Combination: {ch}")
            # print(f"Paras: {params}")
            val_years, auc_for_each_year = run_model(list(measure), final_df_data, hidden_dim, learning_rate, num_epochs)
            param_key = str(params)
            auc_records_measurement[measure][param_key] = dict(zip(val_years, auc_for_each_year))
    return auc_records_measurement

def process_dataset_run_all(data_source, grid,channels):
    results = {}
    final_df_data = final_df(channels, data_source)
    for _ in tqdm(range(len(grid))):
        params = grid[_]
        learning_rate = params['learning_rate']
        hidden_dim = params['hidden_dim']
        num_epochs = params['num_epochs']
        # Assuming 'run_model' is your function to train the model and it returns validation years and AUC for each year
        val_years, auc_for_each_year = run_model(channels, final_df_data, hidden_dim, learning_rate, num_epochs)
        # Convert the parameters to a string key
        param_key = str(params)
        # Store the results
        results[param_key] = dict(zip(val_years, auc_for_each_year))
    return results

## case for all
channels = ['(?<!P)OA', 'POA', 'CE', 'PE', 'EDU']
measurements = ['degree', 'closeness', 'betweeness', '(?<!Person_)PageRank', 'Person_PageRank']

# Check the job array ID environment variable
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

# Select the appropriate data source and process the dataset
################## All *11 #####################
if job_id == 0: # all with all scores
    data_source = networkscores_match
    suffix = 'all'
    results = process_dataset_run_all(data_source, grid, channels)
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_all.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
## move into channels with all
elif job_id == 1:
    data_source = networkscores_match
    channel_single = channels[0:1]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 2:
    data_source = networkscores_match
    channel_single = channels[1:2]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 3:
    data_source = networkscores_match
    channel_single = channels[2:3]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 4:
    data_source = networkscores_match
    channel_single = channels[3:4]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 5:
    data_source = networkscores_match
    channel_single = channels[4:5]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
## move into measurements
elif job_id == 6:
    data_source = networkscores_match
    measurements_single = measurements[0:1]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 7:
    data_source = networkscores_match
    measurements_single = measurements[1:2]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 8:
    data_source = networkscores_match
    measurements_single = measurements[2:3]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 9:
    data_source = networkscores_match
    measurements_single = measurements[3:4]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 10:
    data_source = networkscores_match
    measurements_single = measurements[4:5]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'all'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
################## Females * 11 ##################
elif job_id == 11:
    data_source = networkscores_female
    suffix = 'female'
    results = process_dataset_run_all(data_source, grid, channels)
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_all.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
## move into channels with all
elif job_id == 12:
    data_source = networkscores_female
    channel_single = channels[0:1]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 13:
    data_source = networkscores_female
    channel_single = channels[1:2]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 14:
    data_source = networkscores_female
    channel_single = channels[2:3]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 15:
    data_source = networkscores_female
    channel_single = channels[3:4]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 16:
    data_source = networkscores_female
    channel_single = channels[4:5]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
## move into measurements
elif job_id == 17:
    data_source = networkscores_female
    measurements_single = measurements[0:1]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 18:
    data_source = networkscores_female
    measurements_single = measurements[1:2]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 19:
    data_source = networkscores_female
    measurements_single = measurements[2:3]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 20:
    data_source = networkscores_female
    measurements_single = measurements[3:4]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 21:
    data_source = networkscores_female
    measurements_single = measurements[4:5]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'female'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
################## Males * 11 ##################
elif job_id == 22:
    data_source = networkscores_male
    suffix = 'male'
    results = process_dataset_run_all(data_source, grid, channels)
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_all.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
## move into channels with all
elif job_id == 23:
    data_source = networkscores_male
    channel_single = channels[0:1]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 24:
    data_source = networkscores_male
    channel_single = channels[1:2]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 25:
    data_source = networkscores_male
    channel_single = channels[2:3]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 26:
    data_source = networkscores_male
    channel_single = channels[3:4]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 27:
    data_source = networkscores_male
    channel_single = channels[4:5]
    results = process_dataset_channels(data_source, grid, channel_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_channels_{channel_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
## move into measurements
elif job_id == 28:
    data_source = networkscores_male
    measurements_single = measurements[0:1]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 29:
    data_source = networkscores_male
    measurements_single = measurements[1:2]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 30:
    data_source = networkscores_male
    measurements_single = measurements[2:3]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 31:
    data_source = networkscores_male
    measurements_single = measurements[3:4]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
elif job_id == 32:
    data_source = networkscores_male
    measurements_single = measurements[4:5]
    results = process_dataset_measurements(data_source, grid, measurements_single)
    suffix = 'male'
    file_path_all = f'/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/female_results_match/results_match_nearest_{suffix}_measurements_{measurements_single[0]}.pkl'
    with open(file_path_all, 'wb') as file:
        pickle.dump(results, file)
else:
    raise ValueError("Invalid job ID")