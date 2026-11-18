# BreakingBarriers

Code for the paper:

> Yuhao Zhou, Wenhao Chen, María Óskarsdóttir, Matt Davison, Cristián Bravo (2026). Unveiling gender
> disparities in corporate board career paths using deep learning. *Patterns* 7(4), 101495.
> <https://doi.org/10.1016/j.patter.2026.101495>

The working title of the paper was "Breaking Barriers: Unveiling Gender Disparities in Corporate Board
Career Paths Using Deep Learning", hence the name of this repository. The study uses BoardEx data on
more than 19,000 senior managers and board members of over 700 publicly traded Canadian firms from 2000
to 2022. The code works with director networks from five sources (current employment, prior employment,
education, and current and prior social engagement), describes each director by five centrality
measures per network and year (degree, closeness, betweenness, PageRank and personalized PageRank),
and matches female to male
directors on career sequences (optimal matching with TraMineR) and static features. It then trains LSTM
models with out-of-time cross-validation to predict a board appointment in the next year, measures the
contribution of each network source by ablation, analyses the retain-forget behaviour of the LSTM gates,
and recomputes personalized PageRank within each gender. The code archived with the paper on Zenodo
(<https://doi.org/10.5281/zenodo.17675341>, version v1.0.0) was taken from a fork of this repository at
commit `71dbf14`, the last commit on `main` here before this release.

## Requirements

Python 3. The package versions used for the paper were not recorded. The demonstration notebook
`Weight/Sample_of_one.ipynb` was last saved with Python 3.10.9.

```bash
pip install -r requirements.txt
```

`Scripts/matching/MatchingPairs.py` and `Scripts/gender_based/Gender_base_Match_and_Fine_tune.py` also
need R with the TraMineR package, which they call through `rpy2`. `MatchingPairs.py` sets `R_HOME` at
line 50 to the R module of the Alliance (Compute Canada) software stack; change that line on other
systems. `Gender_base_Match_and_Fine_tune.py` reads `R_HOME` from the environment.

Hardware, from `Docs/`: the matching step needs 64 GB of memory or more, and
`Scripts/database/all_rela.py` ran on 32 CPU cores for two weeks on an HPC cluster. The hardware and run
time of the model training were not recorded.

### Configuration

The scripts read their paths from environment variables. Only `MatchingPairs.py` has relative defaults;
in the other scripts the defaults are placeholders such as `/path/to/your/data`, so set the variables.

| Variable | Used by | Default |
|---|---|---|
| `DATA_DIR` | `MatchingPairs.py`, `FineTune_MatchedPairs.py`, `Feature_Importance.py`, `retention analysis.py`, `Gender_base_Match_and_Fine_tune.py` | `data` in `MatchingPairs.py`, a placeholder elsewhere |
| `DB_PATH` | `MatchingPairs.py`, both `Gender_base_*.py` | `data/base.db` in `MatchingPairs.py`, a placeholder elsewhere |
| `HOME_DIR` | `MatchingPairs.py` (only to build the default `DB_PATH`) | `.` |
| `RESULT_DIR` | `FineTune_MatchedPairs.py`, `Feature_Importance.py`, `retention analysis.py` | placeholder |
| `OUTPUT_DIR` | both `Gender_base_*.py` | placeholder |
| `GENDER` | both `Gender_base_*.py` | `female` (or `male`) |
| `R_HOME` | `Gender_base_Match_and_Fine_tune.py` | placeholder |
| `SLURM_ARRAY_TASK_ID` | the training, importance and gender-based scripts: selects the experiment | `0` |

The database scripts take their paths from a dictionary in the code: `FILE_PATHS` in
`Scripts/database/parquet2db.py` (`<YOUR_PATH_HERE>` placeholders) and `dic_string` in
`Scripts/database/all_rela.py` (`./data/base.db`).

## Data

The study uses BoardEx (North American tables), a subscription database accessed through WRDS. Its terms
of service prohibit redistribution of the raw data, so no data is included here. Subscribers can obtain
the tables from the provider. `data/README.md` lists the tables, where each script expects its inputs,
and which intermediate files are not produced by any script in this repository.
`Docs/Data Construction.md` describes how the network database is built.

## Training

The order of the steps follows `Docs/Data Construction.md` and `Docs/CodeMapping.md`. The figure numbers
in `Docs/CodeMapping.md` follow an earlier version of the manuscript; the Results section below uses the
numbers of the published article.

```bash
# 1. Build the SQLite database (edit FILE_PATHS and uncomment the calls under __main__ first)
python Scripts/database/parquet2db.py
python Scripts/database/cps_relation.py
python Scripts/database/all_rela.py          # writes ./res/output_<timestamp>.csv; create res/ first

# 2. Match female to male directors (writes matched_pairs.pkl to the working folder)
DATA_DIR=data DB_PATH=data/base.db python Scripts/matching/MatchingPairs.py

# 3. LSTM grid search with out-of-time cross-validation (12 validation years)
#    task 0-2: matched sample (all channels, single channels, single measures)
#    task 3-5: male matched sample; task 6-8: female matched sample
SLURM_ARRAY_TASK_ID=0 DATA_DIR=data RESULT_DIR=results python Scripts/modelling/FineTune_MatchedPairs.py

# 4. Gender-specific personalized PageRank, one network source per task (0-4: ce, pe, edu, oa, poa)
GENDER=female SLURM_ARRAY_TASK_ID=0 DB_PATH=data/base.db OUTPUT_DIR=results \
  python Scripts/gender_based/Gender_base_update_scores.py

# 5. Matching and LSTM grid search on the gender-specific scores (tasks 0-21)
GENDER=female SLURM_ARRAY_TASK_ID=0 DATA_DIR=data DB_PATH=data/base.db OUTPUT_DIR=results \
  python Scripts/gender_based/Gender_base_Match_and_Fine_tune.py
```

`MatchingPairs.py` uses a caliper of 1 on the sum of seven normalised dissimilarity matrices;
`Gender_base_Match_and_Fine_tune.py` uses a caliper of 0.15 on their mean. The paper reports a
dissimilarity threshold of 0.15.

### Known gaps

Found by reading the code; the code was not run for this release.

- `Scripts/database/parquet2db.py`, `cps_relation.py` and `all_rela.py` import a module `db_operate`
  that is not in the repository.
- `cps_relation.py` imports `file_path` from `parquet2db.py`, which defines `FILE_PATHS`, and calls
  `InitCPS()`, which is not defined.
- `FineTune_MatchedPairs.py` expects `male_data` and `female_data` from the matching step, but defines
  neither, and `MatchingPairs.py` saves only the matched index pairs.
- `Gender_base_Match_and_Fine_tune.py` uses `OneHotEncoder` without importing it.
- The step that turns the output of `Gender_base_update_scores.py` into the
  `*_person_female.csv` and `*_person_male.csv` files read by `Gender_base_Match_and_Fine_tune.py` is not
  in the repository.

## Evaluation

```bash
# Feature importance by ablation: zeroes one network source at a time, 10 runs
# (task 0, 1, 2: all, female, male matched samples; writes RESULT_DIR/match_result_<group>_<task>.pkl)
SLURM_ARRAY_TASK_ID=0 DATA_DIR=data RESULT_DIR=results python Scripts/modelling/Feature_Importance.py
```

`Scripts/modelling/retention analysis.py` defines `run_model_retain_groups`, which trains a gated LSTM
per validation year and writes the retain and intake indices (`RII_25_<year>.csv`,
`RII_channel_<year>.csv`, `RII_measure_<year>.csv`, and optionally `dAUC_channel_<year>.csv`). The file
has no entry point: uncomment the example at its end (lines 278 to 282) to run it.

`Weight/Sample_of_one.ipynb` loads `Weight/all_all/model_valyear_2019.pth` and predicts the probability
of a board appointment for one synthetic director with random features. Open it from the `Weight/`
folder, since the model path is relative.

## Pretrained models

`Weight/` holds 396 PyTorch state dictionaries (38,088,072 bytes in all), 12 in each of 33 folders.
The folders are named `<group>_<features>`, with group `all`, `female` or `male` and features `all`,
`ce`, `pe`, `edu`, `cse`, `pse`, `degree`, `closeness`, `betweeness`, `pagerank` or `personpagerank`.
Each folder holds `model_valyear_2010.pth` to `model_valyear_2021.pth`, which match the twelve
validation years of the out-of-time splits in the training scripts. Every file is 96,182 bytes and has
the same architecture: a one-layer LSTM with 25 inputs and 64 hidden units, and a linear output layer.
No script in the repository saves these files, and the repository does not document how they were
trained. In particular, the single-feature folders also take 25 inputs, while the training scripts give
a single-source model 5 inputs. Only `Weight/Sample_of_one.ipynb` loads one of them.

## Results

| Table or figure in the paper | Command | Expected result |
|---|---|---|
| Tables 1 and 2, Figures 1 to 5 (descriptive statistics) | Not documented | Not documented |
| Figure 6 (fraction of women matched by threshold) | `Scripts/matching/MatchingPairs.py` computes one matching; the threshold sweep and plot are not in the repository | Not documented |
| Figure 8 (network source AUC by group) | `FineTune_MatchedPairs.py`, tasks 0, 1, 3, 4, 6, 7; the weighted AUC and plot are not in the repository | Not documented |
| Figures 9, 10, 15, 16 (retention heatmaps) | `run_model_retain_groups` in `retention analysis.py` (RII files); the heatmaps are not in the repository | Not documented |
| Figures 11, 12, 13 (network source contribution) | `Feature_Importance.py`; the aggregation and plots are not in the repository | Not documented |
| Figure 14 (network centrality AUC by group) | `FineTune_MatchedPairs.py`, tasks 2, 5, 8; the weighted AUC and plot are not in the repository | Not documented |
| Figures 17, 18, 19 (network centrality contribution) | Not documented: `Feature_Importance.py` ablates network sources only | Not documented |
| Figures 20, 21 (female-to-female and male-to-male support) | `Gender_base_update_scores.py`, then `Gender_base_Match_and_Fine_tune.py` with `GENDER=female` or `male` | Not documented |

The mapping of scripts to published figures follows `Docs/CodeMapping.md` and the code, renumbered to
the published article. No expected values are recorded in the repository.

## Citation

```bibtex
@article{zhou2026unveiling,
  title   = {Unveiling gender disparities in corporate board career paths using deep learning},
  author  = {Zhou, Yuhao and Chen, Wenhao and Óskarsdóttir, María and Davison, Matt and Bravo, Cristián},
  year    = {2026},
  journal = {Patterns},
  volume  = {7},
  number  = {4},
  pages   = {101495},
  doi     = {10.1016/j.patter.2026.101495},
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Licence

GNU Affero General Public License v3.0. See `LICENSE`.

## Acknowledgements

Code by Yuhao (Jet) Zhou (GitHub: JetZhouUWO); the paper credits the software to Yuhao Zhou and Wenhao
Chen. The paper acknowledges the NSERC Discovery Grant program, the SSHRC Insight program, the Icelandic
Research Fund, the Canada Research Chairs program and the Ontario Graduate Scholarship program, and
compute support from Compute Ontario, Calcul Québec and the Digital Research Alliance of Canada.
