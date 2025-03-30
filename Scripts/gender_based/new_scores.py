import multiprocessing
import sqlite3
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os

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
    page_ranks = nx.pagerank(G)
    return page_ranks

def calculate_closeness(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    closeness = nx.closeness_centrality(G)
    return closeness
    
def calculate_betweeness(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    betweeness = nx.betweenness_centrality(G)
    return betweeness    
    
def calculate_degree(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    degree = nx.degree_centrality(G)
    return degree

def calculate_abs_degree(year_table, filtered_b):
    G = nx.Graph()
    for _, row in year_table.iterrows():
        head = row['Head']
        tail = row['Tail']
        G.add_edge(head, tail)
    abs_degree = dict(G.degree())
    return abs_degree
        
# if __name__ == '__main__':
conn = sqlite3.connect('/home/jetzhou/projects/def-cbravo/jetzhou/base.db')
cursor = conn.cursor()

query1 = """
SELECT BoardID FROM company_profile
WHERE HOCountryName='Canada' AND CountryOfQuote='CANADA' AND CCCountryName='Canada'
"""
board_ids = [x[0] for x in cursor.execute(query1).fetchall()]

query2 = """
SELECT *
FROM org_composition_all
WHERE CompanyID IN {}
""".format(tuple(board_ids))

directors_df = pd.read_sql_query(query2, conn)
#filtered_b = directors_df[directors_df['RoleName'].str.contains('CEO')]
filtered_b = directors_df.loc[directors_df['Seniority'].isin(['Executive Director', 'Supervisory Director'])]

query4 = """
    SELECT *
    FROM dir_profile
    """
dir_profile = pd.read_sql_query(query4, conn)
male_director_ids = dir_profile[dir_profile['Gender'] == 'M']['DirectorID'].unique()
female_director_ids = dir_profile[dir_profile['Gender'] == 'F']['DirectorID'].unique()
# sources = ['ce','pe','edu','oa','poa']
# sources = ['ce']
persons = ['male','female']

# Check the job array ID environment variable
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

if job_id == 0:
    source = 'ce'
    query3 = """
        SELECT * FROM dir_ce_rela_all WHERE Head in {} AND Tail in {}
        """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
    # query3 = """
    #     SELECT * FROM tab_cps_relation_ce WHERE Head in {} AND Tail in {}
    #     """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
    df_ce_canadian = pd.read_sql_query(query3, conn)
    df_ce_canadian['OverlapYearStart'] = df_ce_canadian['OverlapYearStart'].astype(int)
    df_ce_canadian['OverlapYearEnd'] = df_ce_canadian['OverlapYearEnd'].astype(int)
    for person in persons:
        print(person)
        num_cpus = 32
        yearly_tables = []
        pool = multiprocessing.Pool(processes=num_cpus)
        results = []
        results2 = []
        results3 = []
        results4 = []
        results5 = []
        for year in range(2000, 2023):
            year_table = df_ce_canadian[(df_ce_canadian['OverlapYearStart'] <= year) & (df_ce_canadian['OverlapYearEnd'] >= year)]
            year_table_lag = df_ce_canadian[(df_ce_canadian['OverlapYearStart'] <= year-1) & (df_ce_canadian['OverlapYearEnd'] >= year-1)]
            if person == 'male':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, male_director_ids)))
            elif person == 'female':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, female_director_ids)))
            results2.append(pool.apply_async(calculate_page_rank, (year_table, filtered_b)))
            results3.append(pool.apply_async(calculate_closeness, (year_table, filtered_b)))
            results4.append(pool.apply_async(calculate_betweeness, (year_table, filtered_b)))
            results5.append(pool.apply_async(calculate_degree, (year_table, filtered_b)))
        pool.close()
        pool.join()

# Initialize an empty DataFrame
        df = pd.DataFrame()
        print('start person pagerank')
        for year, result in enumerate(results):
            person_pageranks = result.get()
            for node, rank in person_pageranks.items():
            # Append row to the DataFrame
                df = pd.concat([df,pd.DataFrame({"Year": [year+2000], "Node": [node], "Person_PageRank": [rank]})], ignore_index=True)
        print('start pagerank')
        for year, result in enumerate(results2):
            pageranks = result.get()
            for node, rank in pageranks.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "PageRank"] = rank
        print('start closeness')
        for year, result in enumerate(results3):
            closeness = result.get()
            for node, rank in closeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "closeness"] = rank
        print('start betweeness')
        for year, result in enumerate(results4):
            betweeness = result.get()
            for node, rank in betweeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "betweeness"] = rank
        print('start degree')
        for year, result in enumerate(results5):
            degree = result.get()
            for node, rank in degree.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "degree"] = abs(rank)
        
    # Write the DataFrame to a CSV file
        df.to_csv(f"/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/{source}_new_output_person_to_{person}.csv", index=False)
        print('------')
        
elif job_id == 1:
    source = 'pe'
    query3 = """
        SELECT * FROM dir_pe_rela_all WHERE Head in {} AND Tail in {}
        """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
        # query3 = """
    #     SELECT * FROM tab_cps_relation_pe WHERE Head in {} AND Tail in {}
    #     """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
    df_pe_canadian = pd.read_sql_query(query3, conn)
    df_pe_canadian['OverlapYearStart'] = df_pe_canadian['OverlapYearStart'].astype(int)
    for person in persons:
        print(person)
        num_cpus = 32
        yearly_tables = []
        pool = multiprocessing.Pool(processes=num_cpus)
        results = []
        results2 = []
        results3 = []
        results4 = []
        results5 = []

        for year in range(2000, 2023):
            year_table = df_pe_canadian[(df_pe_canadian['OverlapYearStart'] <= year)]
            year_table_lag = df_pe_canadian[(df_pe_canadian['OverlapYearStart'] <= year-1)]
            if person == 'male':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, male_director_ids)))
            elif person == 'female':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, female_director_ids)))
            results2.append(pool.apply_async(calculate_page_rank, (year_table, filtered_b)))
            results3.append(pool.apply_async(calculate_closeness, (year_table, filtered_b)))
            results4.append(pool.apply_async(calculate_betweeness, (year_table, filtered_b)))
            results5.append(pool.apply_async(calculate_degree, (year_table, filtered_b)))
        pool.close()
        pool.join()

# Initialize an empty DataFrame
        df = pd.DataFrame()
        print('start person pagerank')
        for year, result in enumerate(results):
            person_pageranks = result.get()
            for node, rank in person_pageranks.items():
            # Append row to the DataFrame
                df = pd.concat([df,pd.DataFrame({"Year": [year+2000], "Node": [node], "Person_PageRank": [rank]})], ignore_index=True)
        print('start pagerank')
        for year, result in enumerate(results2):
            pageranks = result.get()
            for node, rank in pageranks.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "PageRank"] = rank
        print('start closeness')
        for year, result in enumerate(results3):
            closeness = result.get()
            for node, rank in closeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "closeness"] = rank
        print('start betweeness')
        for year, result in enumerate(results4):
            betweeness = result.get()
            for node, rank in betweeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "betweeness"] = rank
        print('start degree')
        for year, result in enumerate(results5):
            degree = result.get()
            for node, rank in degree.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "degree"] = abs(rank)
        
    # Write the DataFrame to a CSV file
        df.to_csv(f"/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/{source}_new_output_person_to_{person}.csv", index=False)
        print('------')  
if job_id == 2:
    source = 'edu'
    query3 = """
        SELECT * FROM dir_edu_rela_all WHERE Head in {} AND Tail in {}
        """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
    df_edu_canadian = pd.read_sql_query(query3, conn)
    df_edu_canadian['Year'] = df_edu_canadian['Year'].astype(int)
    # df_ce_canadian['OverlapYearEnd'] = df_ce_canadian['OverlapYearEnd'].astype(int)
    for person in persons:
        print(person)
        num_cpus = 32
        yearly_tables = []
        pool = multiprocessing.Pool(processes=num_cpus)
        results = []
        results2 = []
        results3 = []
        results4 = []
        results5 = []

        for year in range(2000, 2023):
            # year_table = df_ce_canadian[(df_ce_canadian['OverlapYearStart'] <= year) & (df_ce_canadian['OverlapYearEnd'] >= year)]
            # year_table_lag = df_ce_canadian[(df_ce_canadian['OverlapYearStart'] <= year-1) & (df_ce_canadian['OverlapYearEnd'] >= year-1)]
            year_table = df_edu_canadian[(df_edu_canadian['Year'] <= year)]
            year_table_lag = df_edu_canadian[(df_edu_canadian['Year'] <= year-1)]
            if person == 'male':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, male_director_ids)))
            elif person == 'female':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, female_director_ids)))
            results2.append(pool.apply_async(calculate_page_rank, (year_table, filtered_b)))
            results3.append(pool.apply_async(calculate_closeness, (year_table, filtered_b)))
            results4.append(pool.apply_async(calculate_betweeness, (year_table, filtered_b)))
            results5.append(pool.apply_async(calculate_degree, (year_table, filtered_b)))
            # results6.append(pool.apply_async(calculate_abs_degree, (year_table, filtered_b)))
        pool.close()
        pool.join()

# Initialize an empty DataFrame
        df = pd.DataFrame()
        print('start person pagerank')
        for year, result in enumerate(results):
            person_pageranks = result.get()
            for node, rank in person_pageranks.items():
            # Append row to the DataFrame
                df = pd.concat([df,pd.DataFrame({"Year": [year+2000], "Node": [node], "Person_PageRank": [rank]})], ignore_index=True)
        print('start pagerank')
        for year, result in enumerate(results2):
            pageranks = result.get()
            for node, rank in pageranks.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "PageRank"] = rank
        print('start closeness')
        for year, result in enumerate(results3):
            closeness = result.get()
            for node, rank in closeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "closeness"] = rank
        print('start betweeness')
        for year, result in enumerate(results4):
            betweeness = result.get()
            for node, rank in betweeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "betweeness"] = rank
        print('start degree')
        for year, result in enumerate(results5):
            degree = result.get()
            for node, rank in degree.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "degree"] = abs(rank)

    # Write the DataFrame to a CSV file
        df.to_csv(f"/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/{source}_new_output_person_to_{person}.csv", index=False)
        print('------')
        
if job_id == 3:
    source = 'oa'
    query3 = """
        SELECT * FROM dir_oa_rela_all WHERE Head in {} AND Tail in {}
        """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
    df_oa_canadian  = pd.read_sql_query(query3, conn)
    df_oa_canadian['OverlapYearStart'] = df_oa_canadian['OverlapYearStart'].astype(int)
    df_oa_canadian['OverlapYearEnd'] = df_oa_canadian['OverlapYearEnd'].astype(int)
    for person in persons:
        print(person)
        num_cpus = 32
        yearly_tables = []
        pool = multiprocessing.Pool(processes=num_cpus)
        results = []
        results2 = []
        results3 = []
        results4 = []
        results5 = []

        for year in range(2000, 2023):
            year_table = df_oa_canadian[(df_oa_canadian['OverlapYearStart'] <= year) & (df_oa_canadian['OverlapYearEnd'] >= year)]
            year_table_lag = df_oa_canadian[(df_oa_canadian['OverlapYearStart'] <= year-1) & (df_oa_canadian['OverlapYearEnd'] >= year-1)]
            if person == 'male':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, male_director_ids)))
            elif person == 'female':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, female_director_ids)))
            results2.append(pool.apply_async(calculate_page_rank, (year_table, filtered_b)))
            results3.append(pool.apply_async(calculate_closeness, (year_table, filtered_b)))
            results4.append(pool.apply_async(calculate_betweeness, (year_table, filtered_b)))
            results5.append(pool.apply_async(calculate_degree, (year_table, filtered_b)))

        pool.close()
        pool.join()

# Initialize an empty DataFrame
        df = pd.DataFrame()
        print('start person pagerank')
        for year, result in enumerate(results):
            person_pageranks = result.get()
            for node, rank in person_pageranks.items():
            # Append row to the DataFrame
                df = pd.concat([df,pd.DataFrame({"Year": [year+2000], "Node": [node], "Person_PageRank": [rank]})], ignore_index=True)
        print('start pagerank')
        for year, result in enumerate(results2):
            pageranks = result.get()
            for node, rank in pageranks.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "PageRank"] = rank
        print('start closeness')
        for year, result in enumerate(results3):
            closeness = result.get()
            for node, rank in closeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "closeness"] = rank
        print('start betweeness')
        for year, result in enumerate(results4):
            betweeness = result.get()
            for node, rank in betweeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "betweeness"] = rank
        print('start degree')
        for year, result in enumerate(results5):
            degree = result.get()
            for node, rank in degree.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "degree"] = abs(rank)

    # Write the DataFrame to a CSV file
        df.to_csv(f"/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/{source}_new_output_person_to_{person}.csv", index=False)
        print('------')
        
if job_id == 4:
    source = 'poa'
    query3 = """
        SELECT * FROM dir_poa_rela_all WHERE Head in {} AND Tail in {}
        """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
    df_poa_canadian = pd.read_sql_query(query3, conn)
    df_poa_canadian['OverlapYearStart'] = df_poa_canadian['OverlapYearStart'].astype(int)
    # df_ce_canadian['OverlapYearEnd'] = df_ce_canadian['OverlapYearEnd'].astype(int)
    for person in persons:
        print(person)
        num_cpus = 32
        yearly_tables = []
        pool = multiprocessing.Pool(processes=num_cpus)
        results = []
        results2 = []
        results3 = []
        results4 = []
        results5 = []

        for year in range(2000, 2023):
            year_table = df_poa_canadian[(df_poa_canadian['OverlapYearStart'] <= year)]
            year_table_lag = df_poa_canadian[(df_poa_canadian['OverlapYearStart'] <= year-1)]
            if person == 'male':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, male_director_ids)))
            elif person == 'female':
                results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, female_director_ids)))
            results2.append(pool.apply_async(calculate_page_rank, (year_table, filtered_b)))
            results3.append(pool.apply_async(calculate_closeness, (year_table, filtered_b)))
            results4.append(pool.apply_async(calculate_betweeness, (year_table, filtered_b)))
            results5.append(pool.apply_async(calculate_degree, (year_table, filtered_b)))

        pool.close()
        pool.join()

# Initialize an empty DataFrame
        df = pd.DataFrame()
        print('start person pagerank')
        for year, result in enumerate(results):
            person_pageranks = result.get()
            for node, rank in person_pageranks.items():
            # Append row to the DataFrame
                df = pd.concat([df,pd.DataFrame({"Year": [year+2000], "Node": [node], "Person_PageRank": [rank]})], ignore_index=True)
        print('start pagerank')
        for year, result in enumerate(results2):
            pageranks = result.get()
            for node, rank in pageranks.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "PageRank"] = rank
        print('start closeness')
        for year, result in enumerate(results3):
            closeness = result.get()
            for node, rank in closeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "closeness"] = rank
        print('start betweeness')
        for year, result in enumerate(results4):
            betweeness = result.get()
            for node, rank in betweeness.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "betweeness"] = rank
        print('start degree')
        for year, result in enumerate(results5):
            degree = result.get()
            for node, rank in degree.items():
            # Update the DataFrame
                df.loc[(df["Year"] == year+2000) & (df["Node"] == node), "degree"] = abs(rank)
        
    # Write the DataFrame to a CSV file
        df.to_csv(f"/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/{source}_new_output_person_to_{person}.csv", index=False)
        print('------')
# for source in sources:
#     print(source)
#     if source == 'ce':
#         query3 = """
#         SELECT * FROM dir_ce_rela_all WHERE Head in {} AND Tail in {}
#         """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
#     elif source == 'pe':
#         query3 = """
#         SELECT * FROM dir_pe_rela_all WHERE Head in {} AND Tail in {}
#         """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
#     elif source == 'edu':
#         query3 = """
#         SELECT * FROM dir_edu_rela_all WHERE Head in {} AND Tail in {}
#         """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
#     elif source == 'oa':
#         query3 = """
#         SELECT * FROM dir_oa_rela_all WHERE Head in {} AND Tail in {}
#         """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))
#     elif source == 'poa':
#         query3 = """
#         SELECT * FROM dir_poa_rela_all WHERE Head in {} AND Tail in {}
#         """.format(tuple(directors_df.DirectorID.unique()), tuple(directors_df.DirectorID.unique()))

#     df_ce_canadian = pd.read_sql_query(query3, conn)
#     df_ce_canadian['OverlapYearStart'] = df_ce_canadian['OverlapYearStart'].astype(int)
#     df_ce_canadian['OverlapYearEnd'] = df_ce_canadian['OverlapYearEnd'].astype(int)
#     for person in persons:
#         print(person)
#         num_cpus = 32
#         yearly_tables = []
#         pool = multiprocessing.Pool(processes=num_cpus)
#         results = []
#         results2 = []
#         results3 = []
#         results4 = []
#         results5 = []
#         # results6 = []
#         for year in range(df_ce_canadian['OverlapYearStart'].min(), 2023):
#             year_table = df_ce_canadian[(df_ce_canadian['OverlapYearStart'] <= year) & (df_ce_canadian['OverlapYearEnd'] >= year)]
#             year_table_lag = df_ce_canadian[(df_ce_canadian['OverlapYearStart'] <= year-1) & (df_ce_canadian['OverlapYearEnd'] >= year-1)]
#             if person == 'male':
#                 results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, male_director_ids)))
#             elif person == 'female':
#                 results.append(pool.apply_async(calculate_person_page_rank, (year_table, year_table_lag, filtered_b, female_director_ids)))
#             results2.append(pool.apply_async(calculate_page_rank, (year_table, filtered_b)))
#             results3.append(pool.apply_async(calculate_closeness, (year_table, filtered_b)))
#             results4.append(pool.apply_async(calculate_betweeness, (year_table, filtered_b)))
#             results5.append(pool.apply_async(calculate_degree, (year_table, filtered_b)))
#             # results6.append(pool.apply_async(calculate_abs_degree, (year_table, filtered_b)))
#         pool.close()
#         pool.join()

# # Initialize an empty DataFrame
#         df = pd.DataFrame()
#         print('start person pagerank')
#         for year, result in enumerate(results):
#             person_pageranks = result.get()
#             for node, rank in person_pageranks.items():
#             # Append row to the DataFrame
#                 df = pd.concat([df,pd.DataFrame({"Year": [year+df_ce_canadian['OverlapYearStart'].min()], "Node": [node], "Person_PageRank": [rank]})], ignore_index=True)
#         print('start pagerank')
#         for year, result in enumerate(results2):
#             pageranks = result.get()
#             for node, rank in pageranks.items():
#             # Update the DataFrame
#                 df.loc[(df["Year"] == year+df_ce_canadian['OverlapYearStart'].min()) & (df["Node"] == node), "PageRank"] = rank
#         print('start closeness')
#         for year, result in enumerate(results3):
#             closeness = result.get()
#             for node, rank in closeness.items():
#             # Update the DataFrame
#                 df.loc[(df["Year"] == year+df_ce_canadian['OverlapYearStart'].min()) & (df["Node"] == node), "closeness"] = rank
#         print('start betweeness')
#         for year, result in enumerate(results4):
#             betweeness = result.get()
#             for node, rank in betweeness.items():
#             # Update the DataFrame
#                 df.loc[(df["Year"] == year+df_ce_canadian['OverlapYearStart'].min()) & (df["Node"] == node), "betweeness"] = rank
#         print('start degree')
#         for year, result in enumerate(results5):
#             degree = result.get()
#             for node, rank in degree.items():
#             # Update the DataFrame
#                 df.loc[(df["Year"] == year+df_ce_canadian['OverlapYearStart'].min()) & (df["Node"] == node), "degree"] = abs(rank)
        
#     #     for year, result in enumerate(results6):
#     #         abs_degree = result.get()
#     #         for node, rank in abs_degree.items():
#     #         # Update the DataFrame
#     #             df.loc[(df["Year"] == year+df_ce_canadian['OverlapYearStart'].min()) & (df["Node"] == node), "abs_degree"] = rank
#     # Write the DataFrame to a CSV file
#         df.to_csv(f"/home/jetzhou/projects/def-cbravo/jetzhou/SocialNetworkData/new_person_pagerank/{source}_new_output_person_to_{person}.csv", index=False)
#         print('------')