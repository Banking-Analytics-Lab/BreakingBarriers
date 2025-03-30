"""
MatchingPairs.py

This script processes career and network data to compute dissimilarity matrices using TraMineR
(via rpy2) and then matches female directors to male directors based on a caliper threshold.

Main steps:
1. Import libraries and set global settings.
2. Define configuration variables (for file paths).
3. Define utility functions (set_seed, cleaning role names, categorization, normalization).
4. Load data using configurable paths.
5. Preprocess and clean career data.
6. Compute dissimilarity matrices using R's TraMineR via rpy2.
7. Combine dissimilarity matrices and perform matching.
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
from collections import Counter
from itertools import combinations

from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

from fuzzywuzzy import fuzz

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import warnings
warnings.filterwarnings("ignore")

# Set R_HOME environment variable if needed
os.environ['R_HOME'] = '/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/r/4.3.1/lib64/R'

# ---------------------------
# 2. Configuration Variables
# ---------------------------
# Users should update these variables to point to their own data and database paths.
DATA_DIR = os.getenv('DATA_DIR', '/path/to/your/data')  # e.g., '/home/username/data'
HOME_DIR = os.getenv('HOME_DIR', '/path/to/your/home')  # e.g., '/home/username'
DB_PATH = os.path.join(HOME_DIR, 'path/to/database.db')  # e.g., '/home/username/mydatabase.db'

# ---------------------------
# 3. Define Utility Functions
# ---------------------------
def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_seed = 3402
set_seed(random_seed)

def clean_role_name(role_name, position):
    """
    Clean and standardize a role name.
    
    Parameters:
        role_name (str): Original role name.
        position (str): Director position.
    
    Returns:
        str: Cleaned role name.
    """
    role_name = re.sub(r'Vice\s+President', 'VP', role_name, flags=re.IGNORECASE)
    role_name = re.sub(r'\bChief.*?Officer\b', 'C-suite', role_name, flags=re.IGNORECASE)
    role_name = re.sub(r'\bC(\w+)O\b', r'C-suite', role_name, flags=re.IGNORECASE)
    positions = ["director", "manager", "Executive", "Officer", "Supervisor", "CEO", "CFO",
                 "COO", "Chairman", "Co", "Commissioner", "Secretary", "Dean", "Deputy",
                 "Head", "President", "ED", "VP", "Counsel", "Treasurer", "Minister", "MD",
                 "Professor", "Faculty", "Instructor", "Advisor", "Council", "Member", "NED",
                 "Clerk", "Partner", "Representative"]
    pattern = r"\b(?:{})\b".format("|".join(positions))
    preserved_terms = ["Assistant", "Associate", "Chief", "Senior", "Vice", "Adjunct", "Acting", "Interim"]
    matches = re.findall(pattern, role_name, flags=re.IGNORECASE)
    if matches:
        preserved = [term for term in preserved_terms if term in role_name]
        role_name = " ".join(preserved + matches)
    return role_name

def categorize_role(role):
    """
    Categorize a role based on similarity to a predefined list.
    
    Parameters:
        role (str): Cleaned role name.
    
    Returns:
        str: Categorized role.
    """
    roles = ['VP', 'Supervisory Director', 'C-suite', 'Manager', 'Director', 'Various Positions',
             'President', 'Executive Director', 'Managing Director', 'Executive', 'Partner',
             'Head', 'Advisor', 'Associate', 'Officer', 'Senior VP', 'Founder']
    for r in roles:
        if fuzz.ratio(r, role) >= 50:
            return r
    return 'Other'

def normalize_matrix(matrix):
    """
    Normalize a matrix so that its values lie between 0 and 1.
    
    Parameters:
        matrix (np.array): Input matrix.
    
    Returns:
        np.array: Normalized matrix.
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

# ---------------------------
# 4. Load Data
# ---------------------------
# Load CSV files using configurable DATA_DIR
static_feature = pd.read_csv(os.path.join(DATA_DIR, 'static_features.csv'), index_col=False)
df_background = pd.read_csv(os.path.join(DATA_DIR, 'background_director_after_2010.csv'), index_col=False)
directors_career_path = pd.read_csv(os.path.join(DATA_DIR, 'directors_career_path_trail_with_first_appointment.csv'), index_col=False)
y = pd.read_csv(os.path.join(DATA_DIR, 'target_value.csv'), index_col=False)
networkscores = pd.read_csv(os.path.join(DATA_DIR, 'multi_indexed_df_reverse_non_10_years_lagged_byhand.csv'), index_col=False)

# Ensure career paths only include directors present in network scores
directors_career_path = directors_career_path[directors_career_path.DirectorID.isin(networkscores.DirectorID.unique())]

# ---------------------------
# 5. Preprocess and Clean Career Data
# ---------------------------
# Remove text after "/" or "-" from RoleName
directors_career_path['ProcessedRoleName'] = directors_career_path['RoleName'].apply(lambda x: re.split(r"/|-", x)[0].rstrip())

# Clean and categorize role names
directors_career_path['ProcessedRoleName'] = [
    categorize_role(clean_role_name(role_name, position))
    for role_name, position in zip(directors_career_path['ProcessedRoleName'], directors_career_path['DirectorPosition'])
]

print("Unique ProcessedRoleNames:", directors_career_path['ProcessedRoleName'].nunique())

# Update specific role names if needed
directors_career_path.loc[directors_career_path['ProcessedRoleName'] == 'ED', 'ProcessedRoleName'] = 'Executive Director'
directors_career_path.loc[directors_career_path['ProcessedRoleName'] == 'MD', 'ProcessedRoleName'] = 'Managing Director'

# Merge with organizational summary data from the database
conn = sqlite3.connect(DB_PATH)
query1 = """
SELECT * FROM org_summary_brdlevel
WHERE BoardID IN {}
""".format(tuple(directors_career_path.CompanyID.unique()))
df_org = pd.read_sql_query(query1, conn)
df_org.drop_duplicates(subset=['BoardID'], inplace=True, keep='last')
df_org.loc[df_org.Sector == '', 'Sector'] = 'Other'
df_org.loc[df_org.Index == '', 'Index'] = 'No Index'

directors_career_path = directors_career_path.merge(
    df_org[['BoardID', 'Sector', 'OrgType', 'Index']],
    how='left', left_on='CompanyID', right_on='BoardID'
).drop(columns=['BoardID'])

directors_career_path['Sector'].fillna('Other', inplace=True)
directors_career_path['Index'].fillna('No Index', inplace=True)
directors_career_path['OrgType'].fillna('Other', inplace=True)

directors_career_path["ProcessedRoleName"] = directors_career_path["ProcessedRoleName"].astype(str).str.replace("-", "_")
directors_career_path["Index"] = directors_career_path["Index"].astype(str).str.replace("-", "_")

# Sort and select top 15 records per DirectorID
directors_career_path_sorted = directors_career_path.sort_values(by=['DirectorID', 'DateStartRole'], ascending=[True, False])
top_15_records = directors_career_path_sorted.groupby('DirectorID').head(15)
final_sorted_directors_career_path = top_15_records.sort_values(by=['DirectorID', 'DateStartRole'], ascending=[True, True])

# Concatenate career path information for each director
concatenate_paths = lambda x: '-'.join(x)
career_paths = final_sorted_directors_career_path.groupby('DirectorID')['ProcessedRoleName'].apply(concatenate_paths)
sector_paths = final_sorted_directors_career_path.groupby('DirectorID')['Sector'].apply(concatenate_paths)
orgtype_paths = final_sorted_directors_career_path.groupby('DirectorID')['RowType'].apply(concatenate_paths)
index_paths = final_sorted_directors_career_path.groupby('DirectorID')['Index'].apply(concatenate_paths)

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

# ---------------------------
# 6. Compute Dissimilarity Matrices with TraMineR via rpy2
# ---------------------------
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

normalized_diss_career = normalize_matrix(diss_career)
normalized_diss_sector = normalize_matrix(diss_sector)
normalized_diss_orgtype = normalize_matrix(diss_orgtype)
normalized_diss_index = normalize_matrix(diss_index)

# ---------------------------
# 7. Combine Dissimilarity Matrices and Perform Matching
# ---------------------------
# One-hot encode the 'Region' column from background data
encoder = OneHotEncoder(sparse=False)
nationality_encoded = encoder.fit_transform(df_background[['Region']])
categories = encoder.categories_[0]
feature_names = ['Nationality_' + cat for cat in categories]
static_features_encoded = pd.DataFrame(nationality_encoded, columns=feature_names)
dissimilarity_nationality = squareform(pdist(static_features_encoded, metric='hamming'))
normalized_diss_region = normalize_matrix(dissimilarity_nationality)

# Compute binary features dissimilarity (MBA, PhD, JD)
binary_features = df_background[['Has_MBA', 'Has_PhD', 'Has_JD']]
dissimilarity_binary = squareform(pdist(binary_features, metric='hamming'))
normalized_diss_degree = normalize_matrix(dissimilarity_binary)

# Compute experience dissimilarity
experience = df_background[['experience']]
dissimilarity_experience = squareform(pdist(experience, metric='euclidean'))
normalized_diss_experience = normalize_matrix(dissimilarity_experience)

combined_diss_matrix = (normalized_diss_career + normalized_diss_sector +
                        normalized_diss_orgtype + normalized_diss_index +
                        normalized_diss_region + normalized_diss_degree +
                        normalized_diss_experience)

# Merge gender information into result_df
result_df = result_df.merge(
    final_sorted_directors_career_path[['DirectorID', 'Gender']].drop_duplicates(),
    how='left', on='DirectorID'
)

# Define matching indices (assuming Gender==1 indicates female)
females = result_df[result_df['Gender'] == 1].index
males = result_df[result_df['Gender'] == 0].index

caliper = 1  # Adjust the caliper as needed

matches = {}
matched_males = set()
matched_females = set()

for f in females:
    if f not in matched_females:
        male_distances = combined_diss_matrix[f, males]
        eligible_males = [(idx, m) for idx, m in enumerate(males)
                          if (male_distances[idx] <= caliper and m not in matched_males)]
        if eligible_males:
            sorted_eligible = sorted(eligible_males, key=lambda x: male_distances[x[0]])
            nearest_male = sorted_eligible[0][1]
            matches[f] = nearest_male
            matched_males.add(nearest_male)
            matched_females.add(f)

# Extract matched pairs data
female_data = result_df.loc[[f for f, m in matches.items() if m is not None]]
male_data = result_df.loc[[m for f, m in matches.items() if m is not None]]

# Save the matching result (optional)
with open('matched_pairs.pkl', 'wb') as f:
    pickle.dump(matches, f)

print("Matching complete. Number of matched pairs:", len(matches))
