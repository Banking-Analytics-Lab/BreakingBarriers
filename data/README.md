# Data

No data is included in this repository, and no data should be committed to it.

## Source and access terms

The study uses BoardEx, North American tables, accessed through WRDS. The paper's data availability
statement says that BoardEx is subscription based, that its terms of service prohibit redistribution of
the raw data, and that subscribers with valid access credentials can obtain the full dataset from the
provider.

The raw tables, listed in `Docs/Data Construction.md`, arrive as SAS files (`.sas7bdat`):

- `na_dir_profile_education`
- `na_dir_profile_other_activ`
- `na_wrds_company_names`
- `na_wrds_company_profile`
- `na_wrds_dir_profile_emp`
- `na_wrds_org_composition`
- `na_wrds_company_networks`

The conversion from SAS to Parquet (PySpark with spark-sas7bdat) is not part of this repository.

## Where the code expects the files

| Script | Inputs | Outputs |
|---|---|---|
| `Scripts/database/parquet2db.py` | The Parquet files above, at the paths in `FILE_PATHS` (`<YOUR_PATH_HERE>/<table>.snappy.parquet`) | SQLite tables in `<YOUR_PATH_HERE>/base.db` |
| `Scripts/database/cps_relation.py` | `base.db` (`DB_PATH` from the `parquet2db.py` configuration) | Tables `tab_cps_relation_ce` and `tab_cps_relation_pe` |
| `Scripts/database/all_rela.py` | `./data/base.db`, tables `dir_edu_rela` (or `dir_oa_rela`, `dir_pe_rela`, `dir_poa_rela`) and `org_composition` | `./res/output_<timestamp>.csv` |
| `Scripts/matching/MatchingPairs.py` | In `DATA_DIR` (default `data`): `static_features.csv`, `background_director_after_2010.csv`, `directors_career_path_trail_with_first_appointment.csv`, `target_value.csv`, `multi_indexed_df_reverse_non_10_years_lagged_byhand.csv`. In `DB_PATH` (default `data/base.db`): table `org_summary_brdlevel` | `matched_pairs.pkl` in the working folder |
| `Scripts/modelling/FineTune_MatchedPairs.py` | In `DATA_DIR`: `multi_indexed_df_reverse_non_10_years_lagged_byhand.csv`, `DirectorTimeRecord_updated.csv` | `results_match_<group>_<experiment>.pkl` in `RESULT_DIR` |
| `Scripts/modelling/Feature_Importance.py` | In `DATA_DIR`: `final_df_all_match.pkl`, `final_df_female_match.pkl`, `final_df_male_match.pkl` | `match_result_<group>_<task>.pkl` in `RESULT_DIR` |
| `Scripts/modelling/retention analysis.py` | The same three `final_df_*_match.pkl` files | `RII_*.csv` and `dAUC_channel_*.csv` in the chosen output folder |
| `Scripts/gender_based/Gender_base_update_scores.py` | In `DB_PATH`: tables `company_profile`, `org_composition_all`, `dir_profile`, `dir_<source>_rela_all` | `<source>_new_output_person_to_<gender>.csv` in `OUTPUT_DIR/<gender>_results_match/` |
| `Scripts/gender_based/Gender_base_Match_and_Fine_tune.py` | In `DATA_DIR`: `multi_indexed_df_reverse_non_10_years_lagged_byhand_person_female.csv` or `..._person_male.csv`, `static_features.csv`, `background_director_after_2010_update_v2.csv`, `directors_career_path_trail_with_first_appointment.csv`, `target_value.csv`, `DirectorTimeRecord_updated.csv`. In `DB_PATH`: table `org_summary_brdlevel` | `results_match_nearest_*.pkl` in `OUTPUT_DIR/<gender>_results_match/` |

## Files that no script here produces

Not documented: the repository does not contain the code that builds these inputs from the database.

- The CSV files read from `DATA_DIR` by the matching, training and gender-based scripts, including the
  network-score tables `multi_indexed_df_reverse_non_10_years_lagged_byhand*.csv`.
- The three `final_df_*_match.pkl` files.
- The database tables `org_summary_brdlevel`, `company_profile`, `org_composition`,
  `org_composition_all`, `dir_profile` and the `dir_*_rela` and `dir_*_rela_all` tables.

No synthetic sample is provided. `Weight/Sample_of_one.ipynb` runs without data: it draws random
features for one synthetic director.
