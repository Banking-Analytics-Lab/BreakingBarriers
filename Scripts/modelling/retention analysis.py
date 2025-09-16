# retention_analysis.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = os.getenv('DATA_DIR', '/path/to/your/data')
RESULT_DIR = os.getenv('RESULT_DIR', '/path/to/results')

MATCHED_DATASETS = {
    'all': os.path.join(DATA_DIR, 'final_df_all_match.pkl'),
    'female': os.path.join(DATA_DIR, 'final_df_female_match.pkl'),
    'male': os.path.join(DATA_DIR, 'final_df_male_match.pkl'),
}

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=3402):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3402)

# ---------------------------
# Group definitions
# ---------------------------
channels_order = ['(?<!P)OA', 'POA', 'CE', 'PE', 'EDU']
measures_order = ['degree', 'closeness', 'betweeness',
                  '(?<!Person_)PageRank', 'Person_PageRank']

# 25 = 5 channels × 5 measures
group_25 = {(ch, meas): [ci*5 + mj]
            for ci, ch in enumerate(channels_order)
            for mj, meas in enumerate(measures_order)}

group_by_channel = {ch: [ci*5 + j for j in range(5)]
                    for ci, ch in enumerate(channels_order)}

group_by_measure = {meas: [ci*5 + mj for ci in range(len(channels_order))]
                    for mj, meas in enumerate(measures_order)}

# ---------------------------
# Model with gate logging
# ---------------------------
class _LSTMCellWithGates(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wx = nn.Linear(input_size, 4*hidden_size, bias=True)
        self.Wh = nn.Linear(hidden_size, 4*hidden_size, bias=True)

    def forward(self, x_t, h_prev, c_prev):
        z = self.Wx(x_t) + self.Wh(h_prev)
        i, f, g, o = z.chunk(4, dim=-1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        o = torch.sigmoid(o); g = torch.tanh(g)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t, {'i': i, 'f': f, 'o': o, 'g': g}

class LSTMModelGated(nn.Module):
    """
    forward(x, mask, return_gates=False) -> logits[, gate_log]
    x: [B,T,25], mask: [B,T] (1=real, 0=pad).
    """
    def __init__(self, input_dim=25, hidden_dim=64, dropout=0.5, use_masked_last=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = _LSTMCellWithGates(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.use_masked_last = use_masked_last

    def forward(self, x, mask, return_gates=False):
        B, T, F = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        H_seq, gate_log = [], {'i': [], 'f': [], 'o': [], 'g': [], 'c_norm': []} if return_gates else None

        for t in range(T):
            h, c, gts = self.cell(x[:, t, :], h, c)
            H_seq.append(h.unsqueeze(1))
            if return_gates:
                gate_log['i'].append(gts['i'].mean(-1))
                gate_log['f'].append(gts['f'].mean(-1))
                gate_log['o'].append(gts['o'].mean(-1))
                gate_log['g'].append(gts['g'].mean(-1).abs())
                gate_log['c_norm'].append(c.norm(dim=-1))

        H = self.dropout(torch.cat(H_seq, dim=1))
        h_last = H[torch.arange(B), mask.sum(dim=1).long() - 1] if self.use_masked_last else H[:, -1, :]
        logits = self.fc(h_last).squeeze(-1)

        if return_gates:
            m = mask.float()
            for k in gate_log:
                gate_log[k] = torch.stack(gate_log[k], dim=1) * m  # [B,T]
            return logits, gate_log
        return logits

# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def front_pad(sequences):
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    F = sequences[0].shape[-1]
    X = torch.zeros((len(sequences), max_len, F), dtype=torch.float32)
    M = torch.zeros((len(sequences), max_len), dtype=torch.float32)
    for i, seq in enumerate(sequences):
        L = len(seq)
        X[i, -L:, :] = seq
        M[i, -L:] = 1.0
    return X, M

@torch.no_grad()
def compute_RII(X, M, gates, group_dict, max_L=10, p=1):
    """
    Retention/Intake Index (RII):
      E[ f_t * ||x_{t,group}||_p ],  E[ i_t * ||x_{t,group}||_p ]
    """
    B, T, F = X.shape
    m = M.float()
    Fgate, Igate = gates['f'], gates['i']

    last_idx = (m.sum(dim=1) - 1).long()
    rows = []

    def group_mag(x_bt, idxs):
        V = x_bt[..., idxs]
        if isinstance(idxs, list) and len(idxs) == 1:
            V = V[..., 0]
            return V.abs() if p == 1 else V.pow(2).sqrt()
        return (V.abs().pow(p).sum(dim=-1)).pow(1.0/p)

    for gname, idxs in group_dict.items():
        Xg = group_mag(X, idxs) * m
        Rf = np.zeros(max_L); Ri = np.zeros(max_L); Cnt = np.zeros(max_L)
        for b in range(B):
            for t in range(T):
                if m[b, t] == 1:
                    lag = int(last_idx[b].item() - t + 1)
                    if 1 <= lag <= max_L:
                        Rf[lag-1] += float(Fgate[b, t] * Xg[b, t])
                        Ri[lag-1] += float(Igate[b, t] * Xg[b, t])
                        Cnt[lag-1] += 1.0
        Cnt[Cnt==0] = 1.0
        for ell in range(max_L):
            rows.append({"group": gname, "lag": ell+1,
                         "RII_f": Rf[ell]/Cnt[ell],
                         "RII_i": Ri[ell]/Cnt[ell],
                         "support": int(Cnt[ell])})
    return pd.DataFrame(rows)

def knockout_group_lagwindow(X, M, group_cols, lag_start, lag_end):
    Xk = X.clone()
    B, T, F = X.shape
    last_idx = (M.sum(dim=1) - 1).long()
    for b in range(B):
        for lag in range(lag_start, lag_end+1):
            t = int(last_idx[b].item() - (lag-1))
            if 0 <= t < T:
                Xk[b, t, group_cols] = 0.0
    return Xk

# ---------------------------
# Main routine
# ---------------------------
def run_model_retain_groups(final_df, channels, hidden_dim=64, lr=1e-3, epochs=50,
                            outdir='results/retain_forget', compute_dauc=False,
                            windows=[(1,1), (1,3), (4,6)]):

    ensure_dir(outdir)
    years = sorted(final_df['year'].unique())
    window_size = 10
    n_splits = len(years) - window_size
    individuals = final_df['individual'].unique()
    input_dim = len(channels) * 5

    per_fold_meta = []

    for i in range(n_splits):
        start, end = years[0], years[i + window_size - 1]
        val_year = years[i + window_size]

        train_df = final_df[(final_df['year'] >= start) & (final_df['year'] <= end)]
        val_df = final_df[final_df['year'] == val_year]
        window_df = final_df[(final_df['year'] >= start) & (final_df['year'] <= val_year)]

        # Train sequences
        X_train, y_train = [], []
        for ind in individuals:
            sub = train_df[train_df['individual'] == ind].sort_values('year')
            if not sub.empty:
                seq = torch.tensor([s for s in sub['network_scores']], dtype=torch.float32)[-10:]
                y = sub['director_appointment_next_year'].values[-1]
                X_train.append(seq); y_train.append(y)
        if not X_train:
            continue
        X_train, M_train = front_pad(X_train)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # Validation sequences
        X_val, y_val = [], []
        for ind in individuals:
            hist = window_df[window_df['individual'] == ind].sort_values('year')
            subv = val_df[val_df['individual'] == ind]
            if not hist.empty and not subv.empty:
                seq = torch.tensor([s for s in hist['network_scores']], dtype=torch.float32)[-10:]
                y = subv['director_appointment_next_year'].values[-1]
                X_val.append(seq); y_val.append(y)
        if not X_val:
            continue
        X_val, M_val = front_pad(X_val)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        # Train model
        model = LSTMModelGated(input_dim, hidden_dim, dropout=0.5, use_masked_last=False)
        crit = nn.BCEWithLogitsLoss()
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            loss = crit(model(X_train, M_train, return_gates=False), y_train)
            loss.backward(); opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits, gates = model(X_val, M_val, return_gates=True)
            auc = float(roc_auc_score(y_val.numpy(), torch.sigmoid(logits).numpy()))

        # RII
        compute_RII(X_val, M_val, gates, group_25).assign(year=val_year) \
            .to_csv(os.path.join(outdir, f"RII_25_{val_year}.csv"), index=False)
        compute_RII(X_val, M_val, gates, group_by_channel).assign(year=val_year) \
            .to_csv(os.path.join(outdir, f"RII_channel_{val_year}.csv"), index=False)
        compute_RII(X_val, M_val, gates, group_by_measure).assign(year=val_year) \
            .to_csv(os.path.join(outdir, f"RII_measure_{val_year}.csv"), index=False)

        # Optional ΔAUC
        if compute_dauc:
            rows = []
            base = float(roc_auc_score(y_val.numpy(),
                        torch.sigmoid(model(X_val, M_val)).numpy()))
            for gname, idxs in group_by_channel.items():
                for (a, b) in windows:
                    Xk = knockout_group_lagwindow(X_val, M_val, idxs, a, b)
                    auc_k = float(roc_auc_score(y_val.numpy(),
                                torch.sigmoid(model(Xk, M_val)).numpy()))
                    rows.append({"group": gname, "window": f"{a}-{b}",
                                 "delta_auc": auc_k - base,
                                 "base_auc": base, "auc_knockout": auc_k,
                                 "year": val_year})
            pd.DataFrame(rows).to_csv(os.path.join(outdir, f"dAUC_channel_{val_year}.csv"), index=False)

        per_fold_meta.append({"year": val_year, "auc": auc})
        print(f"[retain] year={val_year}  AUC={auc:.3f}  (saved RII files)")

    return pd.DataFrame(per_fold_meta)

# ---------------------------
# Entry Point (example)
# ---------------------------
# df = pd.read_pickle(MATCHED_DATASETS['all'])
# channels = ['CE', 'EDU', 'OA', 'PE', 'POA']
# run_model_retain_groups(df, channels, hidden_dim=64, lr=1e-3, epochs=50,
#                         outdir=os.path.join(RESULT_DIR, 'retain_forget'),
#                         compute_dauc=True)
