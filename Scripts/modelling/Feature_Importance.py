# feature_importance_analysis.py

"""
This unified script runs feature importance analysis by perturbing one channel at a time
for LSTM models trained on matched datasets (all, female, male).

Each dataset is loaded conditionally by job array ID or dataset type string.
Paths are configurable via environment variables.
"""

import os
import pickle
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Configuration Variables
# ---------------------------
DATA_DIR = os.getenv('DATA_DIR', '/path/to/your/data')
RESULT_DIR = os.getenv('RESULT_DIR', '/path/to/results')

MATCHED_DATASETS = {
    'all': os.path.join(DATA_DIR, 'final_df_all_match.pkl'),
    'female': os.path.join(DATA_DIR, 'final_df_female_match.pkl'),
    'male': os.path.join(DATA_DIR, 'final_df_male_match.pkl'),
}

# ---------------------------
# Utility Functions
# ---------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3402)

# Model definition
class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, dropout=0.5, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        out, _ = self.lstm(x)
        out = out * mask.unsqueeze(-1)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# Data helpers
def front_pad_sequences(sequences):
    max_length = max(len(s) for s in sequences)
    feature_size = sequences[0].shape[1]
    padded = torch.zeros((len(sequences), max_length, feature_size))
    mask = torch.zeros((len(sequences), max_length))
    for i, seq in enumerate(sequences):
        padded[i, -len(seq):] = seq
        mask[i, -len(seq):] = 1
    return padded, mask

def prepare_test_data(df, year):
    test_df = df[df['year'] == year]
    X, y = [], []
    for ind in test_df['individual'].unique():
        d = test_df[test_df['individual'] == ind]
        if not d.empty:
            seq = torch.tensor([s for s in d['network_scores']], dtype=torch.float32)
            X.append(seq)
            y.append(d['director_appointment_next_year'].values[-1])
    return front_pad_sequences(X) + (torch.tensor(y, dtype=torch.float32),) if X else (None, None, None)

def perturb_and_evaluate(model, X, mask, y, source_ranges):
    scores = {}
    for k, idx in source_ranges.items():
        X_pert = X.clone()
        X_pert[:, :, idx] = 0
        with torch.no_grad():
            y_pred = torch.sigmoid(model(X_pert, mask)).numpy()
        scores[k] = roc_auc_score(y.numpy(), y_pred)
    return scores

def normalize_importance(year_results, baseline):
    normalized = {}
    for yr, d in year_results.items():
        base = baseline[yr]
        diffs = {k: base - v for k, v in d.items()}
        total = sum(diffs.values())
        normalized[yr] = {k: v / total for k, v in diffs.items()}
    return normalized

# ---------------------------
# Experiment Logic
# ---------------------------

def run_model(df, channels, n_splits=12, window_size=10, hidden_dim=64, lr=0.1, epochs=150):
    years = sorted(df['year'].unique())
    individuals = df['individual'].unique()
    input_dim = len(channels) * 5
    val_auc, val_years = [], []

    for i in tqdm(range(n_splits)):
        start, end = years[0], years[i + window_size - 1]
        val_year = years[i + window_size]
        train_df = df[(df['year'] >= start) & (df['year'] <= end)]
        val_df = df[(df['year'] == val_year) & df['individual'].isin(df[df['year'] == val_year]['individual'].unique())]

        X_train, y_train = [], []
        for ind in individuals:
            d = train_df[train_df['individual'] == ind]
            if not d.empty:
                X_train.append(torch.tensor([s for s in d['network_scores']], dtype=torch.float32))
                y_train.append(d['director_appointment_next_year'].values[-1])
        X_train, m_train = front_pad_sequences(X_train)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        X_val, y_val = [], []
        for ind in individuals:
            d = val_df[val_df['individual'] == ind]
            if not d.empty:
                X_val.append(torch.tensor([s for s in d['network_scores']], dtype=torch.float32))
                y_val.append(d['director_appointment_next_year'].values[-1])
        if not X_val:
            continue
        X_val, m_val = front_pad_sequences(X_val)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        model = LSTMModel(input_dim, hidden_dim)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = crit(model(X_train, m_train), y_train)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = torch.sigmoid(model(X_val, m_val))
            val_auc.append(roc_auc_score(y_val, pred))
            val_years.append(val_year)
    return model, val_years, val_auc

def run_importance(df, channels, source_ranges, n_runs=10):
    all_rounds = []
    for _ in range(n_runs):
        model, val_years, val_auc = run_model(df, channels)
        base_auc = dict(zip(val_years, val_auc))
        feats = {}
        for yr in val_years:
            X, m, y = prepare_test_data(df, yr)
            if X is None:
                continue
            feats[yr] = perturb_and_evaluate(model, X, m, y, source_ranges)
        all_rounds.append(normalize_importance(feats, base_auc))
    return all_rounds

# ---------------------------
# Entry Point
# ---------------------------

channels = ['(?<!P)OA', 'POA', 'CE', 'PE', 'EDU']
source_ranges = {'CE': slice(0, 5), 'EDU': slice(5, 10), 'OA': slice(10, 15), 'PE': slice(15, 20), 'POA': slice(20, 25)}

# Determine which dataset to use
job_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
targets = ['all', 'female', 'male']
dataset_key = targets[job_id % len(targets)]

df = pd.read_pickle(MATCHED_DATASETS[dataset_key])
results = run_importance(df, channels, source_ranges, n_runs=10)

# Save
out_path = os.path.join(RESULT_DIR, f'match_result_{dataset_key}_{job_id}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(results, f)