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

## 2. Model Training and Fine-Tuning (Paper Section 4.2)

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
