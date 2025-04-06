"""
========================================================
Script Name: all_rela_2.py
Author: [Your Name or Initials]
Last Modified: YYYY-MM-DD

Purpose:
    Demonstrates how to process multiple relationship tables
    (e.g., dir_oa_rela, dir_pe_rela, dir_poa_rela) in a
    multiprocessing environment on a supercomputer. The script
    queries a local SQLite database, analyzes overlapping years,
    and writes outputs to CSV.

Usage:
    1. Update the 'dic_string' dictionary to point to the desired
       table (e.g., dir_oa_rela, dir_pe_rela, dir_poa_rela).
    2. Run this script with Python; it will process data in chunks,
       spawn multiple processes, and produce CSV output files.
========================================================
"""

import multiprocessing
import pandas as pd
import sqlite3
from datetime import datetime

import db_operate as dbop  # Adjust if you want to rename or remove

# -------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------
# To change which table you process, you can switch the "QUERY_EDU" value
# in dic_string to something like:
#   "SELECT * FROM dir_oa_rela"  or
#   "SELECT * FROM dir_pe_rela"  or
#   "SELECT * FROM dir_poa_rela"
# For example:
#
# dic_string = {
#     "DB_PATH": "./data/base.db",
#     "QUERY_EDU": "SELECT * FROM dir_oa_rela"
# }
#

dic_string = {
    "DB_PATH": "./data/base.db",
    "QUERY_EDU": "SELECT * FROM dir_edu_rela WHERE SchoolType='Education'"
}


# -------------------------------------------------------------------
# 2. DATABASE ACCESS CLASS
# -------------------------------------------------------------------
class DBOP:
    """
    Wrapper for SQLite operations, including connection handling
    and query execution.

    Attributes
    ----------
    conn : sqlite3.Connection
        Active connection to the SQLite database.
    cur : sqlite3.Cursor
        Cursor for executing SQL statements.
    """

    def __init__(self) -> None:
        """
        Initialize the database connection and cursor using
        the path in dic_string["DB_PATH"].
        """
        self.conn = sqlite3.connect(dic_string["DB_PATH"])
        self.cur = self.conn.cursor()

    def select_table(self, query_str: str) -> pd.DataFrame:
        """
        Execute a SELECT query and return results as a DataFrame.

        Parameters
        ----------
        query_str : str
            The SQL SELECT statement to execute.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the query result.
        """
        return pd.read_sql_query(query_str, self.conn)

    def close_db(self) -> None:
        """
        Close the cursor and the database connection.
        """
        self.cur.close()
        self.conn.close()


# -------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------
def GenerativeOverlapYear(year_key: int,
                          A_year_start: int, A_year_end: int,
                          B_year_start: int, B_year_end: int):
    """
    Determine whether there is an overlapping time window for two
    individuals and a 'relation' year.

    Parameters
    ----------
    year_key : int
        A 'trigger' year to check for possible overlap (e.g. from edu/pe/oa).
    A_year_start : int
        Start year for individual A.
    A_year_end : int
        End year for individual A.
    B_year_start : int
        Start year for individual B.
    B_year_end : int
        End year for individual B.

    Returns
    -------
    list or bool
        If there is an overlap, returns [overlap_start, overlap_end].
        If no overlap, returns False or None.
    """
    overlap_satrt = None
    overlap_end = None

    # If the 'relation year' is after either individual's end year, no overlap
    if year_key > A_year_end or year_key > B_year_end:
        return False

    # Case 1: B's start is within A's range
    if B_year_start >= A_year_start and B_year_start <= A_year_end:
        if B_year_end <= A_year_end:
            overlap_satrt = B_year_start
            overlap_end = B_year_end
        if B_year_end >= A_year_end:
            overlap_satrt = B_year_start
            overlap_end = A_year_end

    # Case 2: A's start is within B's range
    if A_year_start >= B_year_start and A_year_start <= B_year_end:
        if A_year_end <= B_year_end:
            overlap_satrt = A_year_start
            overlap_end = A_year_end
        if A_year_end >= B_year_end:
            overlap_satrt = A_year_start
            overlap_end = B_year_end

    # If there's a valid overlap, compare with the 'year_key'
    if overlap_satrt and overlap_end:
        if year_key >= overlap_satrt and year_key <= overlap_end:
            overlap_satrt = year_key
        return [overlap_satrt, overlap_end]
    else:
        return None


def DealWithDF(index_processes: int, df: pd.DataFrame):
    """
    Process a chunk of rows from the main query in a separate process.

    Parameters
    ----------
    index_processes : int
        ID of the current process (for logging).
    df : pd.DataFrame
        Subset of data to analyze.

    Notes
    -----
    1. For each row in df, we retrieve org_composition data for both 'Head' and 'Tail'.
    2. We compute any year overlap using GenerativeOverlapYear.
    3. Results are written out to a CSV named with a timestamp.
    """
    leng = len(df)
    dbop_local = DBOP()
    # Prepare columns for results
    df_result = pd.DataFrame(columns=[
        "Board_Head", "Board_Tail",
        "Director_Head", "Director_Tail",
        "Head_Position", "Tail_Position",
        "OverlapYearStart", "OverlapYearEnd",
        "Relationship"
    ])

    for row_i in df.index:
        print(f"Process {index_processes}, row {row_i} / {leng}")
        head_edu = df.loc[row_i, "Head"]
        tail_edu = df.loc[row_i, "Tail"]
        year_edu = int(df.loc[row_i, "Year"])

        # Query for each individual's org data
        query_head = f"SELECT * FROM org_composition WHERE DirectorID={head_edu}"
        query_tail = f"SELECT * FROM org_composition WHERE DirectorID={tail_edu}"
        df_head = dbop_local.select_table(query_head)
        df_tail = dbop_local.select_table(query_tail)

        # Drop rows with NaN in YearStartRole or YearEndRole
        df_head.dropna(subset=["YearStartRole", "YearEndRole"], inplace=True)
        df_tail.dropna(subset=["YearStartRole", "YearEndRole"], inplace=True)

        # Compare each row of df_head vs df_tail
        for row_head in df_head.index:
            cmy_id_head = df_head.loc[row_head, "CompanyID"]
            seniority_head = df_head.loc[row_head, "Seniority"]
            year_start_head = int(df_head.loc[row_head, "YearStartRole"])
            year_end_head = int(df_head.loc[row_head, "YearEndRole"])

            for row_tail in df_tail.index:
                cmy_id_tail = df_tail.loc[row_tail, "CompanyID"]
                seniority_tail = df_tail.loc[row_tail, "Seniority"]
                year_start_tail = int(df_tail.loc[row_tail, "YearStartRole"])
                year_end_tail = int(df_tail.loc[row_tail, "YearEndRole"]

                )
                # Determine overlap
                ret = GenerativeOverlapYear(
                    year_edu,
                    year_start_head, year_end_head,
                    year_start_tail, year_end_tail
                )
                if ret:
                    df_result.loc[len(df_result)] = [
                        cmy_id_head, cmy_id_tail,
                        head_edu, tail_edu,
                        seniority_head, seniority_tail,
                        ret[0], ret[1],
                        "EDU"  # or "PE"/"OA"/"POA" as needed
                    ]

    # Write out results once chunk is processed
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'./res/output_{timestamp}.csv'
    df_result.to_csv(filename, index=False)


def error_callback(error):
    """
    Callback function used to capture any exceptions thrown
    inside the multiprocessing pool.
    """
    print(f"[ERROR] {error}")


def DealRelationEdu():
    """
    Main function to split a large query into chunks, spawn processes,
    and coordinate the analysis. The default query is in dic_string["QUERY_EDU"].

    Steps
    -----
    1. Connect to the SQLite database.
    2. Read data in chunks from the query, until no more rows.
    3. For each chunk, split it among multiple processes.
    4. Each process calls DealWithDF(...) on its chunk.
    5. Wait for all processes to finish, then move on to next chunk.
    """
    conn = sqlite3.connect(dic_string["DB_PATH"])
    num_processes = 5  # Adjust as needed

    # The base query (e.g., dir_edu_rela, dir_oa_rela, etc.)
    query = dic_string["QUERY_EDU"]
    chunk_size = 50000
    offset = 0
    total_handled = 0

    # Read data in a loop
    while True:
        query_with_offset = query + f" LIMIT {chunk_size} OFFSET {offset}"
        df_chunk = pd.read_sql_query(query_with_offset, conn)
        length = len(df_chunk)

        # If no more data, break out
        if length == 0:
            break

        # Split chunk among processes
        num_per_process = (length + num_processes - 1) // num_processes
        small_dfs = [group[1] for group in df_chunk.groupby(df_chunk.index // num_per_process)]

        # Create a pool and dispatch tasks
        pool = multiprocessing.Pool(num_processes)
        for index_processes in range(num_processes):
            pool.apply_async(
                DealWithDF,
                args=(index_processes, small_dfs[index_processes]),
                error_callback=error_callback
            )

        pool.close()
        pool.join()

        total_handled += length
        offset += chunk_size

    conn.close()
    print(f"Total rows processed: {total_handled}")


# -------------------------------------------------------------------
# 4. MAIN ENTRY POINT
# -------------------------------------------------------------------
if __name__ == '__main__':
    """
    Adjust the dictionary or function call below if you want to
    switch from 'dir_edu_rela' to 'dir_pe_rela', 'dir_oa_rela',
    or 'dir_poa_rela'. Then run this script on your HPC environment.
    """
    DealRelationEdu()



