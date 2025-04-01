#!/usr/bin/env python3
"""
parquet2db.py

This script reads data from various Parquet files and loads the processed data into a SQLite database.
It is used to construct the underlying data structure for the study.
All file paths are defined in a configuration dictionary and should be updated to match your local environment.
"""

import pandas as pd
import sqlite3 as db
import db_operate as dbop  # Ensure this module is in your repository

# -----------------------------------------------------------------------------
# Configuration Dictionary (Update these paths for your local environment)
# -----------------------------------------------------------------------------
FILE_PATHS = {
    # Education records (e.g., individual educational background)
    "EDU": "<YOUR_PATH_HERE>/na_dir_profile_education.snappy.parquet",
    # Other social activity data for individuals
    "OTHER": "<YOUR_PATH_HERE>/na_dir_profile_other_activ.snappy.parquet",
    "COMPANY_NAMES": "<YOUR_PATH_HERE>/na_wrds_company_names.snappy.parquet",
    "COMPANY_PROFILE": "<YOUR_PATH_HERE>/na_wrds_company_profile.snappy.parquet",
    # Employment history records
    "EMP": "<YOUR_PATH_HERE>/na_wrds_dir_profile_emp.snappy.parquet",
    # Board composition data (company board members and executives)
    "COMPOSITION": "<YOUR_PATH_HERE>/na_wrds_org_composition.snappy.parquet",
    "COMPANY_NETWORKS": "<YOUR_PATH_HERE>/na_wrds_company_networks.snappy.parquet",
    # Database file
    "DB_PATH": "<YOUR_PATH_HERE>/base.db",

    # Table names for the database (destination tables)
    "TB_SCHOOL": "tab_edu_school_id",
    "TB_DIRECTOR": "tab_edu_director_id",
    "TB_EDU": "tab_edu",
    "TB_EDU_RELATION": "tab_edu_relation",
    "TB_CPS_COMPANY": "tab_cps_company_id",
    "TB_CPS_DIRECTOR": "tab_cps_director_id",
    "TB_CPS": "tab_cps",
    "TB_CPS_RELATION_CE": "tab_cps_relation_ce",
    "TB_CPS_RELATION_PE": "tab_cps_relation_pe",
    "TB_OTHER": "tab_other",

    # Example SQL queries (if needed)
    "SQL_QUERY_EDU": "select * from tab_edu",
    "SQL_QUERY_CPS": "select * from tab_cps",
}

# -----------------------------------------------------------------------------
# Functions for Database Construction
# -----------------------------------------------------------------------------

def SchoolNameID2DB():
    """
    Reads the education Parquet file to extract company names and IDs.
    Constructs a mapping from CompanyID to CompanyName and writes this mapping
    to the database table defined by TB_SCHOOL.
    """
    conn = dbop.ConnectDB(FILE_PATHS["DB_PATH"])
    df = pd.read_parquet(FILE_PATHS["EDU"])
    re_df = df.loc[:, ["CompanyName", "CompanyID"]].copy()
    re_df['CompanyID'] = re_df['CompanyID'].astype(int)
    re_df = re_df.loc[:, ["CompanyID", "CompanyName"]]
    
    dic_result = {}
    for row in re_df.index:
        print(f"{row}/{len(df.index)}")
        name = re_df.loc[row, "CompanyName"]
        comp_id = re_df.loc[row, "CompanyID"]
        if comp_id not in dic_result:
            dic_result[comp_id] = name
    df2 = pd.DataFrame(list(dic_result.items()), columns=['CompanyId', 'CompanyName'])
    print(df2.count())
    
    df2.to_sql(FILE_PATHS["TB_SCHOOL"], if_exists="append", con=conn, index=False)
    conn.close()


def DirectorNameID2DB():
    """
    Reads the education Parquet file to extract director names and IDs.
    Constructs a mapping from DirectorID to DirectorName and writes this mapping
    to the database table defined by TB_DIRECTOR.
    """
    conn = dbop.ConnectDB(FILE_PATHS["DB_PATH"])
    df = pd.read_parquet(FILE_PATHS["EDU"])
    re_df = df.loc[:, ["DirectorName", "DirectorID"]].copy()
    re_df['DirectorID'] = re_df['DirectorID'].astype(int)
    
    dic_result = {}
    for row in re_df.index:
        print(f"{row}/{len(df.index)}")
        name = re_df.loc[row, "DirectorName"]
        dir_id = re_df.loc[row, "DirectorID"]
        if dir_id not in dic_result:
            dic_result[dir_id] = name
    df2 = pd.DataFrame(list(dic_result.items()), columns=['DirectorID', 'DirectorName'])
    print(df2.count())
    df2.to_sql(FILE_PATHS["TB_DIRECTOR"], if_exists="append", con=conn, index=False)
    conn.close()


def TableEDU2DB():
    """
    Reads the education Parquet file and processes educational qualification records.
    Extracts the columns: Qualification, DirectorID, CompanyID, and AwardDate.
    Writes the cleaned education records into the database table defined by TB_EDU.
    """
    conn = dbop.ConnectDB(FILE_PATHS["DB_PATH"])
    df = pd.read_parquet(FILE_PATHS["EDU"])
    df = df.loc[:, ["Qualification", "DirectorID", "CompanyID", "AwardDate"]].copy()
    df['DirectorID'] = df['DirectorID'].astype(int)
    df['CompanyID'] = df['CompanyID'].astype(int)
    
    lst_result = []
    for row in df.index:
        print(f"{row}/{len(df.index)}")
        award_date = df.loc[row, "AwardDate"]
        director_id = df.loc[row, "DirectorID"]
        company_id = df.loc[row, "CompanyID"]
        qualification = df.loc[row, "Qualification"]
        if not pd.isna(award_date) and company_id != 0:
            dic_tmp = {
                "DirectorID": director_id,
                "CompanyID": company_id,
                "AwardDate": award_date,
                "Qualification": qualification
            }
            lst_result.append(dic_tmp)
    df2 = pd.DataFrame(lst_result, columns=['DirectorID', 'CompanyID', 'AwardDate', 'Qualification'])
    df2.to_sql(FILE_PATHS["TB_EDU"], if_exists="append", con=conn, index=False)
    conn.close()


def CompositionCompanyDirectorNameID2DB():
    """
    Reads the composition Parquet file to extract company and director names with their IDs.
    Removes duplicate records and writes company data to TB_CPS_COMPANY and director data to TB_CPS_DIRECTOR.
    """
    conn = dbop.ConnectDB(FILE_PATHS["DB_PATH"])
    df = pd.read_parquet(FILE_PATHS["COMPOSITION"])
    # Process company names
    df_company = df.loc[:, ["CompanyName", "CompanyID"]].copy()
    df_company['CompanyID'] = df_company['CompanyID'].astype(int)
    df_company = df_company.drop_duplicates(subset=['CompanyID', 'CompanyName'])
    
    # Process director names
    df_director = pd.read_parquet(FILE_PATHS["COMPOSITION"])
    df_director = df_director.loc[:, ["DirectorName", "DirectorID"]].copy()
    df_director['DirectorID'] = df_director['DirectorID'].astype(int)
    df_director = df_director.drop_duplicates(subset=['DirectorID', 'DirectorName'])
    
    df_company.to_sql(FILE_PATHS["TB_CPS_COMPANY"], if_exists="append", con=conn, index=False)
    df_director.to_sql(FILE_PATHS["TB_CPS_DIRECTOR"], if_exists="append", con=conn, index=False)
    conn.close()


def TableCps2DB():
    """
    Reads the composition Parquet file and extracts employment history data
    (CompanyID, DirectorID, DateStartRole, DateEndRole, and Seniority).
    Filters the records for valid start dates and relevant seniority levels,
    and writes the cleaned employment history into the database table defined by TB_CPS.
    """
    conn = dbop.ConnectDB(FILE_PATHS["DB_PATH"])
    df = pd.read_parquet(FILE_PATHS["COMPOSITION"])
    df = df.loc[:, ["CompanyID", "DirectorID", "DateStartRole", "DateEndRole", "Seniority"]].copy()
    df['DirectorID'] = df['DirectorID'].astype(int)
    df['CompanyID'] = df['CompanyID'].astype(int)
    
    lst_result = []
    for row in df.index:
        print(f"{row}/{len(df.index)}")
        director_id = df.loc[row, "DirectorID"]
        company_id = df.loc[row, "CompanyID"]
        start_role = df.loc[row, "DateStartRole"]
        end_role = df.loc[row, "DateEndRole"]
        seniority = df.loc[row, "Seniority"]
        
        if pd.isna(start_role):
            continue
        if seniority not in ["Senior Manager", "Executive Director", "Supervisory Director"]:
            continue
        if pd.isna(end_role):
            end_role = "2022/12/31 0:00"
        
        dic_tmp = {
            "DirectorID": director_id,
            "CompanyID": company_id,
            "DateStartRole": start_role,
            "DateEndRole": end_role,
            "Seniority": seniority
        }
        lst_result.append(dic_tmp)
    df2 = pd.DataFrame(lst_result, columns=["CompanyID", "DirectorID", "DateStartRole", "DateEndRole", "Seniority"])
    df2.to_sql(FILE_PATHS["TB_CPS"], if_exists="append", con=conn, index=False)
    conn.close()


# -----------------------------------------------------------------------------
# Main Execution (Uncomment the desired function calls)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Uncomment the following lines to run specific data-loading tasks.
    # Make sure to update FILE_PATHS with your actual paths.
    
    # SchoolNameID2DB()
    # DirectorNameID2DB()
    # TableEDU2DB()
    # CompositionCompanyDirectorNameID2DB()
    # TableCps2DB()
