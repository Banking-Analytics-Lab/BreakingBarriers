# Code Mapping

This document maps the code files in this repository to the corresponding sections and figures in the paper. It is intended to help readers understand which parts of the code implement specific methods or produce specific results.

## 1. Data Preprocessing and Matching (Paper Section 4.1)

- **File:** `scripts/matching/MatchingPairs.py`
- **What it does:**  
  - Cleans the data (e.g., role names).  
  - Constructs career sequences.  
  - Computes dissimilarity matrices using TraMineR (R package via rpy2).  
  - Matches female directors to male directors based on a threshold.
- **Related Figures:**  
  - Figures 6 (matching ratio and thresholds)

## 2. Model Training and Fine-Tuning (Paper Section 4.2 and Section 5)

- **File:** `scripts/modeling/FineTune_MatchedPairs.py`
- **What it does:**  
  - Trains and fine-tunes LSTM models on matched candidate pairs.  
  - Supports models for all candidates, male-only candidates, and female-only candidates.  
  - Implements cross-validation over time and hyperparameter search.
- **Related Figures:**  
  - Figure 8 (model performance across groups)  
  - Figure 12 (temporal evaluation of model performance)

## 3. Feature Importance Analysis (Paper Section 5)

- **File:** `scripts/modeling/Feature_Importance.py`
- **What it does:**  
  - Perturbs individual network channels and measurements to assess feature importance.  
  - Computes the change in model performance to estimate importance.
- **Related Figures:**  
  - Figures 9, 10, 11 (importance of network source types)  
  - Figures 13, 14, 15 (importance of network centrality measures)
 
## 4. Gender-based Analysis Model Training and Fine-Tuning (Paper Section 4.2 and Section 5)

- **Files:**  
  - `scripts/gender_based/Gender_base_update_scores.py`  
  - `scripts/gender_based/Gender_base_Match_and_Fine_tune.py`

- **What it does:**  
  - `Gender_base_update_scores.py`:  
    - Computes network centrality scores, including **gender-specific Personalized PageRank** that only propagates influence within the same gender group.  
    - Outputs one set of scores for male-to-male propagation and another for female-to-female propagation.  
  - `Gender_base_Match_and_Fine_tune.py`:  
    - Trains LSTM models using matched candidate pairs and the updated gender-specific scores.  
    - Supports evaluation on all, male-only, or female-only candidate groups.  
    - Includes time-based cross-validation and hyperparameter tuning.

- **Related Figures:**  
  - **Figure 16** – Effect of male-to-male support in networks on board appointments  
  - **Figure 17** – Effect of female-to-female support in networks on board appointments
