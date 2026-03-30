from pathlib import Path
import json, os, random, pickle, warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, average_precision_score
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path("/content/drive/MyDrive/ChemDFM")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

for p in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR, RUNS_DIR, RESULTS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def map_split(x: str) -> str:
    x = str(x).lower()
    if "train" in x:
        return "train"
    if "test" in x or "val" in x:
        return "test"
    if "ood" in x:
        return "ood"
    return "drop"

def load_adata(data_path=None, split_col="split_ho_pathway", require_pathway=False):
    data_path = Path(data_path or (RAW_DIR / "sciplex_complete_middle_subset.h5ad"))
    assert data_path.exists(), f"Missing data file: {data_path}"
    adata = ad.read_h5ad(data_path)
    if "dose_val" in adata.obs.columns and "dose" not in adata.obs.columns:
        adata.obs["dose"] = adata.obs["dose_val"]
    req = ["condition", "cell_type", "dose", split_col]
    if require_pathway:
        req.append("pathway")
    for c in req:
        if c not in adata.obs.columns:
            raise ValueError(f"Missing required obs column: {c}")
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    adata.obs["_split3"] = adata.obs[split_col].astype(str).map(map_split)
    keep_mask = adata.obs["_split3"].isin(["train", "test", "ood"]).values
    adata = adata[keep_mask].copy()
    X = X[keep_mask]
    return adata, X

def rowwise_pearson(a, b):
    vals = []
    for i in range(a.shape[0]):
        if np.std(a[i]) < 1e-8 or np.std(b[i]) < 1e-8:
            continue
        vals.append(pearsonr(a[i], b[i])[0])
    return float(np.mean(vals)) if vals else np.nan

def compute_metrics(pred, true, x0, eval_topk=(20, 50)):
    out = {}
    out["r2_full"] = float(r2_score(true.reshape(-1), pred.reshape(-1)))
    out["pearson_rowmean"] = rowwise_pearson(true, pred)
    out["mse"] = float(np.mean((true - pred) ** 2))
    out["collapse_ratio"] = float(np.var(pred) / (np.var(true) + 1e-8))
    out["mean_shift_error"] = float(np.mean(np.abs((pred - x0) - (true - x0))))
    for k in eval_topk:
        vals = []
        for i in range(true.shape[0]):
            idx = np.argsort(-np.abs(true[i] - x0[i]))[:k]
            vals.append(r2_score(true[i, idx], pred[i, idx]))
        out[f"r2_top{k}"] = float(np.mean(vals))
    return out

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class StructuredDoseEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )
    def forward(self, dose):
        return self.net(dose)

class ResidualDoseResponseModel(nn.Module):
    def __init__(self, latent_dim, n_drugs, n_cells, emb_dim=32, hidden=256, dose_hidden=32, dropout=0.1):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.cell_emb = nn.Embedding(n_cells, emb_dim)
        self.dose_enc = StructuredDoseEncoder(out_dim=dose_hidden)
        self.ctrl_enc = MLP(latent_dim, [hidden, hidden], hidden, dropout=dropout)
        fusion_in = hidden + emb_dim + emb_dim + dose_hidden
        self.delta_head = MLP(fusion_in, [hidden, hidden], latent_dim, dropout=dropout)
    def forward(self, x0, drug_idx, cell_idx, dose):
        z0 = self.ctrl_enc(x0)
        zd = self.drug_emb(drug_idx)
        zc = self.cell_emb(cell_idx)
        zz = self.dose_enc(dose)
        z = torch.cat([z0, zd, zc, zz], dim=1)
        delta_hat = self.delta_head(z)
        xhat = x0 + delta_hat
        return delta_hat, xhat

def pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T
    return torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0.0)

def rbf_mmd(x, y, gamma=None):
    if x.shape[0] < 2 or y.shape[0] < 2:
        return x.new_tensor(0.0)
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)
    if gamma is None:
        with torch.no_grad():
            vals = dxy.flatten()
            vals = vals[vals > 0]
            gamma = 1.0 / (vals.median().item() + 1e-6) if vals.numel() > 0 else 1.0
    Kxx = torch.exp(-gamma * dxx)
    Kyy = torch.exp(-gamma * dyy)
    Kxy = torch.exp(-gamma * dxy)
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

def cell_aware_mmd(x_hat, x_true, cell_idx, min_count=8):
    losses = []
    for cid in torch.unique(cell_idx):
        mask = (cell_idx == cid)
        if mask.sum().item() >= min_count:
            losses.append(rbf_mmd(x_hat[mask], x_true[mask]))
    return torch.stack(losses).mean() if len(losses) > 0 else x_hat.new_tensor(0.0)

def topk_mask_from_true(delta_true, k=10):
    idx = torch.topk(delta_true.abs(), k=min(k, delta_true.shape[1]), dim=1).indices
    mask = torch.zeros_like(delta_true)
    mask.scatter_(1, idx, 1.0)
    return mask

class ChemResidualDataset(Dataset):
    def __init__(self, adata, X_pca, X0_pca, DELTA_pca=None, split="train", include_idx=False, require_pathway=False):
        mask = (adata.obs["_split3"].values == split) & (adata.obs["condition"].astype(str).values != "control")
        self.idxs = np.where(mask)[0]
        self.adata = adata
        self.X = X_pca
        self.X0 = X0_pca
        self.D = DELTA_pca
        self.include_idx = include_idx
        self.require_pathway = require_pathway
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        idx = self.idxs[i]
        row = self.adata.obs.iloc[idx]
        dose = np.log1p(max(float(row["dose"]), 0.0))
        out = {
            "x_true": torch.tensor(self.X[idx], dtype=torch.float32),
            "x0": torch.tensor(self.X0[idx], dtype=torch.float32),
            "drug_idx": torch.tensor(int(row["drug_idx"]), dtype=torch.long),
            "cell_idx": torch.tensor(int(row["cell_idx"]), dtype=torch.long),
            "dose": torch.tensor([dose], dtype=torch.float32),
            "condition": str(row["condition"]),
            "cell_type": str(row["cell_type"]),
        }
        if self.D is not None:
            out["delta"] = torch.tensor(self.D[idx], dtype=torch.float32)
        if self.include_idx:
            out["idx"] = int(idx)
        if self.require_pathway and "pathway" in self.adata.obs.columns:
            out["pathway"] = str(row["pathway"])
        return out

def evaluate_loader(model, loader, eval_topk=(20,50), split_name=None):
    model.eval()
    all_pred, all_true, all_x0 = [], [], []
    all_conditions, all_cells = [], []
    with torch.no_grad():
        for batch in loader:
            x0 = batch["x0"].to(DEVICE)
            x_true = batch["x_true"].to(DEVICE)
            drug_idx = batch["drug_idx"].to(DEVICE)
            cell_idx = batch["cell_idx"].to(DEVICE)
            dose = batch["dose"].to(DEVICE)
            _, x_hat = model(x0, drug_idx, cell_idx, dose)
            all_pred.append(x_hat.cpu().numpy())
            all_true.append(x_true.cpu().numpy())
            all_x0.append(x0.cpu().numpy())
            all_conditions.extend(batch["condition"])
            all_cells.extend(batch["cell_type"])
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    x0 = np.concatenate(all_x0, axis=0)
    overall = compute_metrics(pred, true, x0, eval_topk=eval_topk)
    group_df = pd.DataFrame({"condition": all_conditions, "cell_type": all_cells})
    group_df["sqerr"] = ((pred - true) ** 2).mean(axis=1)
    group_df = group_df.groupby(["cell_type", "condition"], as_index=False)["sqerr"].mean()
    per_cell_rows = []
    for cell in sorted(set(all_cells)):
        m = np.array(all_cells) == cell
        per_cell_rows.append({"split": split_name, "cell_type": cell, **compute_metrics(pred[m], true[m], x0[m], eval_topk=eval_topk)})
    return overall, pd.DataFrame(per_cell_rows), group_df, pred, true, x0

def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "lower": np.nan, "upper": np.nan}
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boots.append(np.mean(sample))
    lower = np.quantile(boots, alpha/2)
    upper = np.quantile(boots, 1-alpha/2)
    return {"mean": float(np.mean(values)), "lower": float(lower), "upper": float(upper)}
