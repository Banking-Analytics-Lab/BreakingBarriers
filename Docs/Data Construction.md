# Data Construction

This document outlines how we build the **network database** and associated tables from BoardEx data, in the correct order of scripts. It also highlights the resource requirements on a high-performance computing (HPC) environment (32 cores running for two weeks).

---

## 1. Converting SAS Files to Parquet

- **Why?**  
  BoardEx data often arrives as `.sas7bdat` files. Converting them to Parquet lets us efficiently read them in Python (e.g., with `pandas`).

- **Method:**  
  Use a PySpark environment and a tool like **spark-sas7bdat**. This step can take substantial time for large files.

- **Where?**  
  The conversion is done outside this repository or in your own Spark job. No sample code is included here—refer to the spark-sas7bdat documentation or your PySpark setup.

- **BoardEx Data (SAS Files):**  
  - `na_dir_profile_education`  
  - `na_dir_profile_other_activ`  
  - `na_wrds_company_names`  
  - `na_wrds_company_profile`  
  - `na_wrds_dir_profile_emp`  
  - `na_wrds_org_composition`  
  - `na_wrds_company_networks`

---

## 2. `parquet2db.py`

- **File Location:** `scripts/database/parquet2db.py`
- **What It Does:**  
  - Reads multiple Parquet files (e.g., education records, employment history).  
  - Loads them into a local SQLite database named `base.db`.  
  - Creates tables (like `tab_edu`, `tab_cps`) for subsequent scripts.

- **Key Outputs:**  
  - A SQLite file named `base.db` containing relevant data tables.

---

## 3. `cps_relation.py`

- **File Location:** `scripts/database/cps_relation.py`
- **What It Does:**  
  - Reads from `base.db` (particularly a table like `tab_cps`).  
  - Detects director relationships within each company, based on overlapping or sequential role dates:
    - **Current Employment (CE)** for overlapping roles.
    - **Prior Employment (PE)** for sequential roles.
  - Stores these relationships in the database (e.g., `TB_CPS_RELATION_CE` and `TB_CPS_RELATION_PE`).

- **Purpose:**  
  - Establish direct linkages between directors—who worked together and when.

---

## 4. `all_rela.py`

- **File Location:** `scripts/database/all_rela.py`
- **What It Does:**  
  - Processes large relationship tables (`dir_oa_rela`, `dir_pe_rela`, `dir_poa_rela`, etc.).  
  - Uses heavy multiprocessing to handle potentially hundreds of millions of rows, analyzing overlap at a yearly level.  
  - Writes final overlap data to CSV files (or inserts back to the database, depending on your code).

- **Resource Needs:**  
  - **Extremely large** data. In our setup, this required **32 CPU cores** continuously for **two weeks** on an HPC cluster.  
  - If your environment is smaller, you might restrict the date range or limit data size.

---

## Summary of Steps

1. **Convert BoardEx SAS to Parquet**  
   - Via PySpark + spark-sas7bdat.  
   - Not included in this repository, but essential before running the scripts below.

2. **Run `parquet2db.py`**  
   - Creates `base.db`, loading each Parquet file into structured tables.

3. **Run `cps_relation.py`**  
   - Builds CE/PE director relationships in the database.

4. **Run `all_rela.py`**  
   - (On HPC) Processes the large `dir_..._rela` tables in parallel, deriving final overlap records.

---

## Notes

- **Data Privacy & Licensing**: Ensure compliance with BoardEx usage terms.  
- **Spot-Checking**: Verify row counts or run small samples before the full HPC job.  
- **Integration**: The final data serve as input to modeling scripts described elsewhere in this repository.
