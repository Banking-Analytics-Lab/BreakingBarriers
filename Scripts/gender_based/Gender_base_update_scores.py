import multiprocessing
import sqlite3
import pandas as pd
import networkx as nx
from tqdm import tqdm
import os

# ---------------------
# Helper Functions
# ---------------------
def create_dictionary(array):
    dictionary = {}
    for element in array:
        dictionary[element] = 1 / len(array)
    return dictionary

def calculate_person_page_rank(year_table, year_table_lag, filtered_b, person_director_ids):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    director_ids = pd.concat([year_table_lag['Head'], year_table_lag['Tail']]).dropna().unique()
    director_ids = [director_id for director_id in director_ids if director_id in person_director_ids]
    result = filtered_b[filtered_b['DirectorID'].isin(director_ids)]
    source_node_id = result['DirectorID'].unique()
    if len(source_node_id) == 0:
        person_pageranks = nx.pagerank(G, alpha=0.85, max_iter=100, tol=0.000001)
    else:
        personalized_weights = create_dictionary(source_node_id)
        person_pageranks = nx.pagerank(G, alpha=0.85, max_iter=100, tol=0.000001, personalization=personalized_weights)
    return person_pageranks

def calculate_page_rank(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    return nx.pagerank(G)

def calculate_closeness(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    return nx.closeness_centrality(G)
    
def calculate_betweeness(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    return nx.betweenness_centrality(G)
    
def calculate_degree(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    return nx.degree_centrality(G)

def calculate_abs_degree(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    return dict(G.degree())

# ---------------------
# Configuration via Environment Variables
# ---------------------
DATA_DIR = os.environ.get('DATA_DIR', '/path/to/data')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/path/to/results')
DB_PATH = os.environ.get('DB_PATH', '/path/to/db.db')
GENDER = os.environ.get("GENDER", "female").lower()  # Choose "female" or "male"

# Output folder based on gender
OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f"{GENDER}_results_match")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ---------------------
# Database and Data Loading
# ---------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

query1 = """
SELECT BoardID FROM company_profile
WHERE HOCountryName='Canada' AND CountryOfQuote='CANADA' AND CCCountryName='Canada'
"""
board_ids = [x[0] for x in cursor.execute(query1).fetchall()]

query2 = f"""
SELECT *
FROM org_composition_all
WHERE CompanyID IN {tuple(board_ids)}
"""
directors_df = pd.read_sql_query(query2, conn)
# For filtering, we use directors with specific seniority
filtered_b = directors_df.loc[directors_df['Seniority'].isin(['Executive Director', 'Supervisory Director'])]

query4 = "SELECT * FROM dir_profile"
dir_profile = pd.read_sql_query(query4, conn)
male_director_ids = dir_profile[dir_profile['Gender'] == 'M']['DirectorID'].unique()
female_director_ids = dir_profile[dir_profile['Gender'] == 'F']['DirectorID'].unique()

# Based on the chosen gender, select the corresponding director IDs for propagation
if GENDER == "male":
    propagation_ids = male_director_ids
elif GENDER == "female":
    propagation_ids = female_director_ids
else:
    raise ValueError("GENDER must be 'male' or 'female'.")

# ---------------------
# Determine the Network Source based on SLURM_ARRAY_TASK_ID
# ---------------------
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
# Map job IDs to network sources
source_dict = {0: "ce", 1: "pe", 2: "edu", 3: "oa", 4: "poa"}
if job_id not in source_dict:
    raise ValueError("Invalid SLURM_ARRAY_TASK_ID. Expected 0-4 for network sources.")
source = source_dict[job_id]

# Query network relationships based on the source
query3 = f"SELECT * FROM dir_{source}_rela_all WHERE Head in {tuple(directors_df.DirectorID.unique())} AND Tail in {tuple(directors_df.DirectorID.unique())}"
df_source = pd.read_sql_query(query3, conn)

# Convert relevant columns to int if necessary
if source in ["ce", "pe", "oa", "poa"]:
    df_source['OverlapYearStart'] = df_source['OverlapYearStart'].astype(int)
    if 'OverlapYearEnd' in df_source.columns:
        df_source['OverlapYearEnd'] = df_source['OverlapYearEnd'].astype(int)
elif source == "edu":
    df_source['Year'] = df_source['Year'].astype(int)

# ---------------------
# Compute Network Scores for the Selected Source and Propagate Based on Gender
# ---------------------
for year in range(2000, 2023):
    if source in ["ce", "oa", "poa"]:
        year_table = df_source[(df_source['OverlapYearStart'] <= year) & (df_source['OverlapYearEnd'] >= year)]
        year_table_lag = df_source[(df_source['OverlapYearStart'] <= year-1) & (df_source['OverlapYearEnd'] >= year-1)]
    elif source == "pe":
        year_table = df_source[(df_source['OverlapYearStart'] <= year)]
        year_table_lag = df_source[(df_source['OverlapYearStart'] <= year-1)]
    elif source == "edu":
        year_table = df_source[(df_source['Year'] <= year)]
        year_table_lag = df_source[(df_source['Year'] <= year-1)]
    # Compute personalized PageRank using propagation_ids based on GENDER
    person_pageranks = calculate_person_page_rank(year_table, year_table_lag, filtered_b, propagation_ids)
    # Save results for each year in a temporary DataFrame (accumulate results)
    if year == 2000:
        out_df = pd.DataFrame({"Year": [year], "Node": list(person_pageranks.keys()), 
                               "Person_PageRank": list(person_pageranks.values())})
    else:
        temp_df = pd.DataFrame({"Year": [year]*len(person_pageranks), "Node": list(person_pageranks.keys()), 
                                "Person_PageRank": list(person_pageranks.values())})
        out_df = pd.concat([out_df, temp_df], ignore_index=True)

# (Optionally, you can also compute other scores here, but per your request only personalized PageRank is updated.)

# ---------------------
# Save the Output CSV
# ---------------------
output_file = os.path.join(OUTPUT_FOLDER, f"{source}_new_output_person_to_{GENDER}.csv")
out_df.to_csv(output_file, index=False)
print(f"Saved updated scores to {output_file}")
