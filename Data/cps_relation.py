#!/usr/bin/env python3
"""
cps_relation.py

This script processes board composition data stored in a database and computes director relationships.
It creates two types of relationships:
    - Current Employment (CE): When two directors’ tenures overlap.
    - Prior Employment (PE): When one director’s tenure ends before another begins.

The resulting relationships are stored in database tables for further analysis.
"""

import pandas as pd
import sqlite3 as db
import db_operate as dbop
from parquet2db import file_path  # file_path contains DB path and SQL query strings (update with your own paths)
import datetime

# =============================================================================
# Class Definitions
# =============================================================================

class Unit:
    """
    A class representing a director's record from the board composition data.
    
    Attributes:
        director_id (int): Unique ID of the director.
        company_id (int): Company ID where the director works.
        start_role (str): Start date of the role (format '%Y-%m-%d %H:%M:%S').
        end_role (str): End date of the role (format '%Y-%m-%d %H:%M:%S').
        seniority (str): The seniority level of the director.
    """
    def __init__(self, director_id, company_id, start_role, end_role, seniority) -> None:
        self.director_id = director_id
        self.company_id = company_id
        self.start_role = start_role
        self.end_role = end_role
        self.seniority = seniority

# =============================================================================
# Utility Functions for Date Comparison
# =============================================================================

def DateCompareReSmaller(dateA, dateB):
    """
    Compare two date strings and return the smaller (earlier) date.
    
    Args:
        dateA (str): Date string in the format '%Y-%m-%d %H:%M:%S'.
        dateB (str): Date string in the format '%Y-%m-%d %H:%M:%S'.
    
    Returns:
        str: The earlier date string.
    """
    timestampA = datetime.datetime.strptime(dateA, '%Y-%m-%d %H:%M:%S')
    timestampB = datetime.datetime.strptime(dateB, '%Y-%m-%d %H:%M:%S')
    return dateA if timestampA <= timestampB else dateB

def DateCompareReBigger(dateA, dateB):
    """
    Compare two date strings and return the larger (later) date.
    
    Args:
        dateA (str): Date string in the format '%Y-%m-%d %H:%M:%S'.
        dateB (str): Date string in the format '%Y-%m-%d %H:%M:%S'.
    
    Returns:
        str: The later date string.
    """
    timestampA = datetime.datetime.strptime(dateA, '%Y-%m-%d %H:%M:%S')
    timestampB = datetime.datetime.strptime(dateB, '%Y-%m-%d %H:%M:%S')
    return dateA if timestampA >= timestampB else dateB

# =============================================================================
# Functions to Compute Relationships Between Director Units
# =============================================================================

def DateCompare(unitA, unitB):
    """
    Compares the date ranges of two director units and generates both current (CE) and prior (PE) employment relationships.
    
    Args:
        unitA (Unit): A director unit.
        unitB (Unit): Another director unit.
    
    Returns:
        list: A list containing two dictionaries:
            - First dictionary: Represents the current employment (CE) relationship.
            - Second dictionary: Represents the prior employment (PE) relationship.
            If no overlap is found, returns an empty list.
    """
    ls_ret = []
    dic_ce = {}
    
    timestampA_start = datetime.datetime.strptime(unitA.start_role, '%Y-%m-%d %H:%M:%S')
    timestampB_start = datetime.datetime.strptime(unitB.start_role, '%Y-%m-%d %H:%M:%S')
    timestampA_end = datetime.datetime.strptime(unitA.end_role, '%Y-%m-%d %H:%M:%S')
    timestampB_end = datetime.datetime.strptime(unitB.end_role, '%Y-%m-%d %H:%M:%S')
    
    # Check for overlapping tenure (CE relationship)
    if timestampB_start < timestampA_end and timestampB_end > timestampA_start:
        dic_ce["Head"] = unitA.director_id
        dic_ce["Tail"] = unitB.director_id
        dic_ce["Label"] = "CE"
        dic_ce["Company"] = unitA.company_id
        dic_ce["SeniorityHead"] = unitA.seniority
        dic_ce["SeniorityTail"] = unitB.seniority
        
        # Determine the overlap start date
        dic_ce["Start"] = timestampA_start if timestampA_start >= timestampB_start else timestampB_start
        
        # Determine the overlap end date, with a small adjustment if the date is Jan 1
        if timestampA_end >= timestampB_end:
            dic_ce["End"] = timestampB_end - datetime.timedelta(days=1) if (timestampB_end.month == 1 and timestampB_end.day == 1) else timestampB_end
        else:
            dic_ce["End"] = timestampA_end - datetime.timedelta(days=1) if (timestampA_end.month == 1 and timestampA_end.day == 1) else timestampA_end
        
        ls_ret.append(dic_ce)
        
        # Generate PE relationship by shifting the start date to one day after the CE end date
        dic_pe = dic_ce.copy()
        dic_pe["Label"] = "PE"
        dic_pe["Start"] = dic_ce["End"] + datetime.timedelta(days=1)
        del dic_pe["End"]
        ls_ret.append(dic_pe)
    
    return ls_ret

def ProduceResult(lst, conn):
    """
    Processes a list of director units and produces the current and prior employment relationships.
    Merges records for the same director (based on director_id and seniority) and writes the relationships to the database.
    
    Args:
        lst (list): List of Unit objects for a single company.
        conn (sqlite3.Connection): Active connection to the SQLite database.
    """
    if len(lst) == 1:
        return

    lst_relation_ce = []
    lst_relation_pe = []
    dic_director_id_tmp = {}
    lst_unit = []
    
    # Merge records for the same director (by director_id and seniority)
    for i in lst:
        key = f"{i.director_id};{i.seniority}"
        if key in dic_director_id_tmp:
            dic_director_id_tmp[key].start_role = DateCompareReSmaller(dic_director_id_tmp[key].start_role, i.start_role)
            dic_director_id_tmp[key].end_role = DateCompareReBigger(dic_director_id_tmp[key].end_role, i.end_role)
        else:
            dic_director_id_tmp[key] = i

    for key in dic_director_id_tmp:
        lst_unit.append(dic_director_id_tmp[key])
        
    # Compare each pair of units to generate relationships
    for i in range(len(lst_unit)):
        for j in range(len(lst_unit) - i):
            if lst_unit[i].director_id == lst_unit[i+j].director_id:
                continue
            ret = DateCompare(lst_unit[i], lst_unit[i+j])
            if ret:
                lst_relation_ce.append(ret[0])
                lst_relation_pe.append(ret[1])
                
    if lst_relation_ce and lst_relation_pe:
        df_ce = pd.DataFrame(lst_relation_ce, columns=["Head", "Tail", "Label", "Company", "SeniorityHead", "SeniorityTail", "Start", "End"])
        df_pe = pd.DataFrame(lst_relation_pe, columns=["Head", "Tail", "Label", "Company", "SeniorityHead", "SeniorityTail", "Start"])
        # Write the results to the corresponding database tables.
        df_ce.to_sql(file_path["TB_CPS_RELATION_CE"], if_exists="append", con=conn, index=False)
        df_pe.to_sql(file_path["TB_CPS_RELATION_PE"], if_exists="append", con=conn, index=False)

# =============================================================================
# Main Grouping Function
# =============================================================================

def Group():
    """
    Main function to process the board composition data and produce director relationships.
    
    Steps:
        1. Initialize the board composition dataset using the query from file_path.
        2. Connect to the database.
        3. For each company, group director units and produce relationships using ProduceResult.
        4. Close the database connection.
    """
    df = InitCPS()
    conn = dbop.ConnectDB(file_path["DB_PATH"])
    cp_id = 2  # Initialize current company ID (update as needed)
    lst_relation = []
    leng = len(df)
    
    for row in df.index:
        one = Unit(
            df.loc[row, "DirectorID"],
            df.loc[row, "CompanyID"],
            df.loc[row, "DateStartRole"],
            df.loc[row, "DateEndRole"],
            df.loc[row, "Seniority"]
        )
        print(f"{row}/{leng}")
        if one.company_id == cp_id:
            lst_relation.append(one)
        else:
            # Process the list of director units for the previous company
            ProduceResult(lst_relation, conn)
            cp_id = one.company_id
            lst_relation = [one]
        if row == len(df) - 1 and lst_relation:
            ProduceResult(lst_relation, conn)
    conn.close()

if __name__ == "__main__":
    Group()
