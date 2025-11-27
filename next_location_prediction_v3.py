#!/usr/bin/env python3
"""
Hierarchical Transformer Pyramid V3 - Production Ready
Next-Location Prediction on GeoLife Dataset

Advanced Features:
- Focal loss for class imbalance
- Mixup augmentation
- Recency-aware attention bias
- Multi-scale pooling
- Stochastic depth
- Warm restarts scheduler

Target: >50% Acc@1, <1M parameters
Author: Claude
"""

import os
import pickle
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SEED
# ============================================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# ============================================================================
# METRICS (PROVIDED)
# ============================================================================
def get_mrr(prediction, targets):
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    return torch.sum(torch.reciprocal(ranks)).cpu().numpy()

def get_ndcg(prediction, targets, k=10):
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float().cpu().numpy()
    ndcg = 1 / np.log2(ranks + 1)
    ndcg[ranks > k] = 0
    return np.sum(ndcg)

def calculate_correct_total_prediction(logits, true_y):
    result_ls = []
    for k in [1, 3, 5, 10]:
        actual_k = min(k, logits.shape[-1])
        pred = torch.topk(logits, k=actual_k, dim=-1).indices
        if k == 1:
            top1 = torch.squeeze(pred).cpu()
        top_k = torch.eq(true_y[:, None], pred).any(dim=1).sum().cpu().numpy()
        result_ls.append(top_k)
    result_ls.extend([get_mrr(logits, true_y), get_ndcg(logits, true_y), true_y.shape[0]])
    return np.array(result_ls, dtype=np.float32), true_y.cpu(), top1

def get_performance_dict(d):
    perf = {k: d[k] for k in ["correct@1", "correct@3", "correct@5", "correct@10", "rr", "ndcg", "f1", "total"]}
    perf["acc@1"] = perf["correct@1"] / perf["total"] * 100
    perf["acc@5"] = perf["correct@5"] / perf["total"] * 100
    perf["acc@10"] = perf["correct@10"] / perf["total"] * 100
    perf["mrr"] = perf["rr"] / perf["total"] * 100
    perf["ndcg"] = perf["ndcg"] / perf["total"] * 100
    return perf

# ============================================================================
# DATASET
# ============================================================================
class GeoLifeDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            'X': torch.tensor(d['X'], dtype=torch.long),
            'weekday': torch.tensor(d['weekday_X'], dtype=torch.long),
            'start_min': torch.tensor(d['start_min_X'], dtype=torch.long),
            'dur': torch.tensor(d['dur_X'], dtype=torch.float32),
            'diff': torch.tensor(d['diff'], dtype=torch.long),
            'user': torch.tensor(d['user_X'], dtype=torch.long),
            's2_11': torch.tensor(d['s2_level11_X'], dtype=torch.long),
            's2_13': torch.tensor(d['s2_level13_X'], dtype=torch.long),
            's2_14': torch.tensor(d['s2_level14_X'], dtype=torch.long),
            's2_15': torch.tensor(d['s2_level15_X'], dtype=torch.long),
            'Y': torch.tensor(d['Y'], dtype=torch.long),
            'seq_len': len(d['X'])
        }

def collate_fn(batch):
    max_len = max(b['seq_len'] for b in batch)
    B = len(batch)
    
    out = {
        'X': torch.zeros(B, max_len, dtype=torch.long),
        'weekday': torch.zeros(B, max_len, dtype=torch.long),
        'start_min': torch.zeros(B, max_len, dtype=torch.long),
        'dur': torch.zeros(B, max_len, dtype=torch.float32),
        'diff': torch.zeros(B, max_len, dtype=torch.long),
        'user': torch.zeros(B, max_len, dtype=torch.long),
        's2_11': torch.zeros(B, max_len, dtype=torch.long),
        's2_13': torch.zeros(B, max_len, dtype=torch.long),
        's2_14': torch.zeros(B, max_len, dtype=torch.long),
        's2_15': torch.zeros(B, max_len, dtype=torch.long),
        'mask': torch.zeros(B, max_len, dtype=torch.bool),
        'Y': torch.stack([b['Y'] for b in batch]),
        'seq_lens': torch.tensor([b['seq_len'] for b in batch]),
    }
    
    for i, b in enumerate(batch):
        L = b['seq_len']
        for k in ['X', 'weekday', 'start_min', 'dur', 'diff', 'user', 
                  's2_11', 's2_13', 's2_14', 's2_15']:
            out[k][i, :L] = b[k]
        out['mask'][i, :L] = True
    
    return out

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class RotaryPositionalEncoding(nn.Module):
    """Rotary position encoding for better position awareness."""
    def __init__(self, dim, max_seq_len=100):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos().unsqueeze(0))
        self.register_buffer('sin', emb.sin().unsqueeze(0))
        
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
        
    def forward(self, x):
        seq_len = x.size(1)
        return x * self.cos[:, :seq_len, :] + self.rotate_half(x) * self.sin[:, :seq_len, :]

class RecencyBiasAttention(nn.Module):
    """Multi-head attention with recency bias for recent locations."""
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=60):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable recency bias
        self.recency_bias = nn.Parameter(torch.zeros(1, 1, max_len, max_len))
        nn.init.normal_(self.recency_bias, std=0.01)
        
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, T, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add recency bias
        attn = attn + self.recency_bias[:, :, :T, :T]
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(out)

class CrossAttention(nn.Module):
    """Cross-attention where Q comes from one stream, K/V from another."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, kv_input, mask=None):
        B, T, _ = query.shape
        
        q = self.q(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(kv_input).view(B, -1, 2, self.n_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(out)

class FeedForward(nn.Module):
    """SwiGLU-style FFN."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class TransformerBlock(nn.Module):
    """Pre-norm Transformer with stochastic depth."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RecencyBiasAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop_path = drop_path
        
    def forward(self, x, mask=None):
        if self.training and random.random() < self.drop_path:
            return x
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class HierarchicalCrossAttnBlock(nn.Module):
    """Cross-attention block with residual."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * 2, dropout)
        
    def forward(self, query, kv, mask=None):
        x = query + self.cross_attn(self.norm_q(query), self.norm_kv(kv), mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x

class MultiScalePooling(nn.Module):
    """Combines last position, mean, and attention pooling."""
    def __init__(self, d_model):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        self.combine = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask, seq_lens):
        B, T, D = x.shape
        device = x.device
        
        # Last position pooling
        last_idx = seq_lens - 1
        last_pool = x[torch.arange(B, device=device), last_idx]
        
        # Mean pooling (masked)
        x_masked = x * mask.unsqueeze(-1).float()
        mean_pool = x_masked.sum(1) / mask.sum(1, keepdim=True).float()
        
        # Attention pooling
        scores = self.attn_pool(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1)
        attn_pool = (x * weights.unsqueeze(-1)).sum(1)
        
        # Combine
        combined = torch.cat([last_pool, mean_pool, attn_pool], dim=-1)
        return self.norm(self.combine(combined))

# ============================================================================
# MAIN MODEL
# ============================================================================

class HierarchicalTransformerPyramidV3(nn.Module):
    """
    Production-ready hierarchical Transformer pyramid.
    
    Architecture:
    1. Rich embeddings (location + temporal + duration)
    2. Main Transformer with recency bias
    3. Parallel cross-attention: Q from main, K/V from S2 hierarchies
    4. Weighted fusion with learnable gates
    5. Fusion Transformer
    6. Multi-scale pooling
    7. Classification head
    """
    def __init__(self, num_locs, num_s2_11, num_s2_13, num_s2_14, num_s2_15, 
                 num_users=50, d_model=64, n_heads=4, n_layers=2, dropout=0.15):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locs + 1, d_model, padding_idx=0)
        self.s2_11_emb = nn.Embedding(num_s2_11 + 1, d_model, padding_idx=0)
        self.s2_13_emb = nn.Embedding(num_s2_13 + 1, d_model, padding_idx=0)
        self.s2_14_emb = nn.Embedding(num_s2_14 + 1, d_model, padding_idx=0)
        self.s2_15_emb = nn.Embedding(num_s2_15 + 1, d_model, padding_idx=0)
        
        # Compact temporal embeddings
        d_t = d_model // 4
        self.weekday_emb = nn.Embedding(8, d_t)
        self.hour_emb = nn.Embedding(25, d_t)
        self.diff_emb = nn.Embedding(9, d_t)
        self.user_emb = nn.Embedding(num_users + 1, d_t)
        self.temporal_proj = nn.Linear(d_t * 4, d_model)
        
        # Duration projection
        self.dur_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Rotary position encoding
        self.rope = RotaryPositionalEncoding(d_model)
        
        # Main Transformer
        drop_rates = [0.1 * i / (n_layers - 1) if n_layers > 1 else 0 for i in range(n_layers)]
        self.main_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout, dr)
            for dr in drop_rates
        ])
        
        # Cross-attention blocks
        self.cross_main = HierarchicalCrossAttnBlock(d_model, n_heads, dropout)
        self.cross_s2_11 = HierarchicalCrossAttnBlock(d_model, n_heads, dropout)
        self.cross_s2_13 = HierarchicalCrossAttnBlock(d_model, n_heads, dropout)
        self.cross_s2_14 = HierarchicalCrossAttnBlock(d_model, n_heads, dropout)
        self.cross_s2_15 = HierarchicalCrossAttnBlock(d_model, n_heads, dropout)
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(5) / 5)
        
        # Fusion projection
        self.fusion_proj = nn.Linear(d_model * 5, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # Fusion Transformer
        self.fusion_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Pooling
        self.pool = MultiScalePooling(d_model)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locs)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
    
    def forward(self, batch):
        X = batch['X']
        mask = batch['mask']
        seq_lens = batch['seq_lens']
        
        # Temporal features
        weekday = batch['weekday']
        hour = torch.clamp(batch['start_min'] // 60, 0, 23)
        diff = torch.clamp(batch['diff'], 0, 7)
        user = batch['user']
        dur = batch['dur'].unsqueeze(-1) / 1440.0
        
        # Build main embedding
        loc = self.loc_emb(X)
        temp = self.temporal_proj(torch.cat([
            self.weekday_emb(weekday), self.hour_emb(hour),
            self.diff_emb(diff), self.user_emb(user)
        ], dim=-1))
        dur_feat = self.dur_proj(dur)
        
        main = self.rope(loc + temp + dur_feat)
        
        # Main Transformer
        for layer in self.main_layers:
            main = layer(main, mask)
        
        # S2 embeddings
        s2_11 = self.rope(self.s2_11_emb(batch['s2_11']))
        s2_13 = self.rope(self.s2_13_emb(batch['s2_13']))
        s2_14 = self.rope(self.s2_14_emb(batch['s2_14']))
        s2_15 = self.rope(self.s2_15_emb(batch['s2_15']))
        
        # Cross-attention: Q from main, K/V from each stream
        out_main = self.cross_main(main, main, mask)
        out_11 = self.cross_s2_11(main, s2_11, mask)
        out_13 = self.cross_s2_13(main, s2_13, mask)
        out_14 = self.cross_s2_14(main, s2_14, mask)
        out_15 = self.cross_s2_15(main, s2_15, mask)
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        streams = [out_main, out_11, out_13, out_14, out_15]
        
        # Concatenate for projection
        concat = torch.cat(streams, dim=-1)
        fused = self.fusion_proj(concat)
        fused = self.fusion_norm(fused)
        
        # Also add weighted combination for residual
        weighted = sum(w * s for w, s in zip(weights, streams))
        fused = fused + weighted
        
        # Fusion Transformer
        for layer in self.fusion_layers:
            fused = layer(fused, mask)
        
        # Pool
        pooled = self.pool(fused, mask, seq_lens)
        
        # Classify
        return self.classifier(pooled)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing
        
    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # One-hot with smoothing
        targets_oh = torch.zeros_like(log_probs)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_oh * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        loss = -focal_weight * targets_smooth * log_probs
        return loss.sum(dim=-1).mean()

# ============================================================================
# TRAINING
# ============================================================================

def mixup_data(batch, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    B = batch['X'].size(0)
    idx = torch.randperm(B, device=batch['X'].device)
    
    return lam, idx

def train_epoch(model, loader, opt, sched, criterion, device, mixup_alpha=0.1):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        opt.zero_grad()
        
        # Optional mixup
        if mixup_alpha > 0 and random.random() < 0.5:
            lam, idx = mixup_data(batch, mixup_alpha)
            logits = model(batch)
            loss = lam * criterion(logits, batch['Y']) + \
                   (1 - lam) * criterion(logits, batch['Y'][idx])
        else:
            logits = model(batch)
            loss = criterion(logits, batch['Y'])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    results = np.zeros(7, dtype=np.float32)
    all_true, all_pred = [], []
    
    for batch in tqdm(loader, desc="Eval"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        
        res, true_y, pred = calculate_correct_total_prediction(logits, batch['Y'])
        results += res
        all_true.extend(true_y.numpy())
        all_pred.extend(pred.numpy())
    
    f1 = f1_score(all_true, all_pred, average='weighted')
    
    return get_performance_dict({
        "correct@1": results[0], "correct@3": results[1],
        "correct@5": results[2], "correct@10": results[3],
        "rr": results[4], "ndcg": results[5],
        "f1": f1, "total": results[6]
    })

def get_vocab(paths):
    vocab = {'locs': set(), 's2_11': set(), 's2_13': set(), 
             's2_14': set(), 's2_15': set(), 'users': set()}
    
    for p in paths:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        for d in data:
            vocab['locs'].update(d['X'].tolist())
            vocab['locs'].add(d['Y'])
            vocab['s2_11'].update(d['s2_level11_X'].tolist())
            vocab['s2_13'].update(d['s2_level13_X'].tolist())
            vocab['s2_14'].update(d['s2_level14_X'].tolist())
            vocab['s2_15'].update(d['s2_level15_X'].tolist())
            vocab['users'].update(d['user_X'].tolist())
    
    return {
        'num_locs': max(vocab['locs']) + 1,
        'num_s2_11': max(vocab['s2_11']) + 1,
        'num_s2_13': max(vocab['s2_13']) + 1,
        'num_s2_14': max(vocab['s2_14']) + 1,
        'num_s2_15': max(vocab['s2_15']) + 1,
        'num_users': max(vocab['users']) + 1
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    DATA_DIR = "/content/expr_hrcl_next_pred_av3/data/temp"
    TRAIN = os.path.join(DATA_DIR, "geolife_transformer_7_train.pk")
    VAL = os.path.join(DATA_DIR, "geolife_transformer_7_validation.pk")
    TEST = os.path.join(DATA_DIR, "geolife_transformer_7_test.pk")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Vocab
    vocab = get_vocab([TRAIN, VAL, TEST])
    print(f"Vocab: {vocab}")
    
    # Config
    cfg = {
        'batch': 64,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.15,
        'lr': 2e-3,
        'wd': 0.01,
        'epochs': 100,
        'patience': 15,
        'mixup': 0.1,
        'focal_gamma': 1.5,
        'label_smooth': 0.1
    }
    
    # Data
    train_ds = GeoLifeDataset(TRAIN)
    val_ds = GeoLifeDataset(VAL)
    test_ds = GeoLifeDataset(TEST)
    
    train_loader = DataLoader(train_ds, cfg['batch'], shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, cfg['batch'], shuffle=False, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, cfg['batch'], shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Model
    model = HierarchicalTransformerPyramidV3(
        num_locs=vocab['num_locs'],
        num_s2_11=vocab['num_s2_11'],
        num_s2_13=vocab['num_s2_13'],
        num_s2_14=vocab['num_s2_14'],
        num_s2_15=vocab['num_s2_15'],
        num_users=vocab['num_users'],
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    assert n_params < 1_000_000, f"Too many params: {n_params}"
    
    # Training setup
    opt = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    criterion = FocalLoss(gamma=cfg['focal_gamma'], label_smoothing=cfg['label_smooth'])
    
    # Train
    best_acc = 0.0
    best_state = None
    patience = 0
    
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    for epoch in range(1, cfg['epochs'] + 1):
        loss = train_epoch(model, train_loader, opt, sched, criterion, device, cfg['mixup'])
        metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}: Loss={loss:.4f}, Acc@1={metrics['acc@1']:.2f}%, "
              f"Acc@5={metrics['acc@5']:.2f}%, MRR={metrics['mrr']:.2f}%")
        
        if metrics['acc@1'] > best_acc:
            best_acc = metrics['acc@1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            print(f"  ★ New best: {best_acc:.2f}%")
        else:
            patience += 1
            if patience >= cfg['patience']:
                print(f"\nEarly stop at epoch {epoch}")
                break
    
    # Test
    if best_state:
        model.load_state_dict(best_state)
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n  Acc@1:  {test_metrics['acc@1']:.2f}%")
    print(f"  Acc@5:  {test_metrics['acc@5']:.2f}%")
    print(f"  Acc@10: {test_metrics['acc@10']:.2f}%")
    print(f"  MRR:    {test_metrics['mrr']:.2f}%")
    print(f"  NDCG:   {test_metrics['ndcg']:.2f}%")
    print(f"  F1:     {test_metrics['f1']:.4f}")
    
    if test_metrics['acc@1'] >= 50:
        print("\n✓ TARGET ACHIEVED: Acc@1 >= 50%!")
    elif test_metrics['acc@1'] >= 45:
        print("\n~ Acceptable: Acc@1 >= 45%")
    else:
        print("\n✗ Below threshold")
    
    return test_metrics

if __name__ == "__main__":
    main()
