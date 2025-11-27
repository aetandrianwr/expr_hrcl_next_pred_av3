#!/usr/bin/env python3
"""
Enhanced Hierarchical Transformer Pyramid for Next-Location Prediction
GeoLife Dataset - Version 2

Key Improvements:
- Multi-scale temporal encoding
- Attention pooling instead of last-position pooling  
- Label smoothing
- Gradient accumulation support
- Better S2 hierarchy integration with gating
- Residual connections throughout

Target: >50% Acc@1, <1M parameters
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
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SEED SETTING
# ============================================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# ============================================================================
# METRIC CALCULATION (PROVIDED)
# ============================================================================
def get_mrr(prediction, targets):
    """Calculates the MRR score for the given predictions and targets."""
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    rranks = torch.reciprocal(ranks)
    return torch.sum(rranks).cpu().numpy()

def get_ndcg(prediction, targets, k=10):
    """Calculates the NDCG score for the given predictions and targets."""
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float().cpu().numpy()
    not_considered_idx = ranks > k
    ndcg = 1 / np.log2(ranks + 1)
    ndcg[not_considered_idx] = 0
    return np.sum(ndcg)

def calculate_correct_total_prediction(logits, true_y):
    top1 = []
    result_ls = []
    for k in [1, 3, 5, 10]:
        if logits.shape[-1] < k:
            k = logits.shape[-1]
        prediction = torch.topk(logits, k=k, dim=-1).indices
        if k == 1:
            top1 = torch.squeeze(prediction).cpu()
        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        result_ls.append(top_k)
    result_ls.append(get_mrr(logits, true_y))
    result_ls.append(get_ndcg(logits, true_y))
    result_ls.append(true_y.shape[0])
    return np.array(result_ls, dtype=np.float32), true_y.cpu(), top1

def get_performance_dict(return_dict):
    perf = {
        "correct@1": return_dict["correct@1"],
        "correct@3": return_dict["correct@3"],
        "correct@5": return_dict["correct@5"],
        "correct@10": return_dict["correct@10"],
        "rr": return_dict["rr"],
        "ndcg": return_dict["ndcg"],
        "f1": return_dict["f1"],
        "total": return_dict["total"],
    }
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
    def __init__(self, data_path, max_seq_len=60):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        X = torch.tensor(sample['X'], dtype=torch.long)
        seq_len = len(X)
        
        weekday = torch.tensor(sample['weekday_X'], dtype=torch.long)
        start_min = torch.tensor(sample['start_min_X'], dtype=torch.long)
        dur = torch.tensor(sample['dur_X'], dtype=torch.float32)
        diff = torch.tensor(sample['diff'], dtype=torch.long)
        user = torch.tensor(sample['user_X'], dtype=torch.long)
        
        s2_11 = torch.tensor(sample['s2_level11_X'], dtype=torch.long)
        s2_13 = torch.tensor(sample['s2_level13_X'], dtype=torch.long)
        s2_14 = torch.tensor(sample['s2_level14_X'], dtype=torch.long)
        s2_15 = torch.tensor(sample['s2_level15_X'], dtype=torch.long)
        
        Y = torch.tensor(sample['Y'], dtype=torch.long)
        
        return {
            'X': X, 'weekday': weekday, 'start_min': start_min,
            'dur': dur, 'diff': diff, 'user': user,
            's2_11': s2_11, 's2_13': s2_13, 's2_14': s2_14, 's2_15': s2_15,
            'Y': Y, 'seq_len': seq_len
        }

def collate_fn(batch):
    """Custom collate function with padding."""
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    padded = {
        'X': torch.zeros(batch_size, max_len, dtype=torch.long),
        'weekday': torch.zeros(batch_size, max_len, dtype=torch.long),
        'start_min': torch.zeros(batch_size, max_len, dtype=torch.long),
        'dur': torch.zeros(batch_size, max_len, dtype=torch.float32),
        'diff': torch.zeros(batch_size, max_len, dtype=torch.long),
        'user': torch.zeros(batch_size, max_len, dtype=torch.long),
        's2_11': torch.zeros(batch_size, max_len, dtype=torch.long),
        's2_13': torch.zeros(batch_size, max_len, dtype=torch.long),
        's2_14': torch.zeros(batch_size, max_len, dtype=torch.long),
        's2_15': torch.zeros(batch_size, max_len, dtype=torch.long),
        'mask': torch.zeros(batch_size, max_len, dtype=torch.bool),
        'Y': torch.zeros(batch_size, dtype=torch.long),
        'seq_lens': torch.zeros(batch_size, dtype=torch.long),
    }
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        for key in ['X', 'weekday', 'start_min', 'dur', 'diff', 'user', 
                    's2_11', 's2_13', 's2_14', 's2_15']:
            padded[key][i, :seq_len] = item[key]
        padded['mask'][i, :seq_len] = True
        padded['Y'][i] = item['Y']
        padded['seq_lens'][i] = seq_len
    
    return padded

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention with optional cross-attention."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        B, T, _ = query.shape
        
        Q = self.q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class CrossAttentionBlock(nn.Module):
    """Cross-attention block with Q from one stream, K/V from another."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, kv, mask=None):
        q_normed = self.norm_q(query)
        kv_normed = self.norm_kv(kv)
        x = query + self.attn(q_normed, kv_normed, kv_normed, mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x

class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence summarization."""
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
            
        out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))

class HierarchicalGate(nn.Module):
    """Gating mechanism to weight hierarchical features."""
    def __init__(self, d_model, n_streams):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_streams, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_streams),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features):
        # features: list of (B, T, D)
        concat = torch.cat(features, dim=-1)  # (B, T, D*n_streams)
        gates = self.gate(concat)  # (B, T, n_streams)
        
        stacked = torch.stack(features, dim=-1)  # (B, T, D, n_streams)
        gated = (stacked * gates.unsqueeze(-2)).sum(dim=-1)  # (B, T, D)
        return gated

# ============================================================================
# MAIN MODEL
# ============================================================================

class HierarchicalTransformerPyramidV2(nn.Module):
    """
    Enhanced Hierarchical Transformer Pyramid.
    
    Architecture:
    1. Embedding layers with temporal features
    2. Main stream Transformer processing
    3. Parallel cross-attention: Q from main, K/V from S2 hierarchies
    4. Hierarchical gating for intelligent feature fusion
    5. Fusion Transformer
    6. Attention pooling + classification
    """
    def __init__(
        self,
        num_locations,
        num_s2_11, num_s2_13, num_s2_14, num_s2_15,
        num_users=50,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.15,
        max_seq_len=60
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_streams = 5  # Main + 4 S2 levels
        
        # Embeddings
        self.loc_embed = nn.Embedding(num_locations + 1, d_model, padding_idx=0)
        self.s2_11_embed = nn.Embedding(num_s2_11 + 1, d_model, padding_idx=0)
        self.s2_13_embed = nn.Embedding(num_s2_13 + 1, d_model, padding_idx=0)
        self.s2_14_embed = nn.Embedding(num_s2_14 + 1, d_model, padding_idx=0)
        self.s2_15_embed = nn.Embedding(num_s2_15 + 1, d_model, padding_idx=0)
        
        # Temporal embeddings (compact)
        d_temp = d_model // 4
        self.weekday_embed = nn.Embedding(8, d_temp, padding_idx=0)
        self.hour_embed = nn.Embedding(25, d_temp, padding_idx=0)  # 24 hours + pad
        self.diff_embed = nn.Embedding(9, d_temp, padding_idx=0)   # 0-7 + pad
        self.user_embed = nn.Embedding(num_users + 1, d_temp, padding_idx=0)
        self.temporal_proj = nn.Linear(d_temp * 4, d_model)
        
        # Duration projection
        self.dur_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Main stream Transformer
        self.main_transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Cross-attention blocks for each S2 level
        self.cross_attn_main = CrossAttentionBlock(d_model, n_heads, dropout)
        self.cross_attn_s2_11 = CrossAttentionBlock(d_model, n_heads, dropout)
        self.cross_attn_s2_13 = CrossAttentionBlock(d_model, n_heads, dropout)
        self.cross_attn_s2_14 = CrossAttentionBlock(d_model, n_heads, dropout)
        self.cross_attn_s2_15 = CrossAttentionBlock(d_model, n_heads, dropout)
        
        # Hierarchical gating
        self.hier_gate = HierarchicalGate(d_model, self.n_streams)
        
        # Fusion Transformer
        self.fusion_transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Pooling
        self.pool = AttentionPooling(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
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
                    nn.init.zeros_(m.weight[m.padding_idx])
    
    def forward(self, batch):
        X = batch['X']
        mask = batch['mask']
        
        # Temporal features
        weekday = batch['weekday']
        hour = torch.clamp(batch['start_min'] // 60, 0, 23)
        diff = torch.clamp(batch['diff'], 0, 7)
        user = batch['user']
        dur = batch['dur'].unsqueeze(-1) / 1440.0  # Normalize duration
        
        # Location embedding
        loc_emb = self.loc_embed(X)
        
        # Temporal embedding
        temp_emb = torch.cat([
            self.weekday_embed(weekday),
            self.hour_embed(hour),
            self.diff_embed(diff),
            self.user_embed(user)
        ], dim=-1)
        temp_emb = self.temporal_proj(temp_emb)
        
        # Duration embedding
        dur_emb = self.dur_proj(dur)
        
        # Combine embeddings for main stream
        main_emb = loc_emb + temp_emb + dur_emb
        main_emb = self.pos_enc(main_emb)
        
        # Process main stream
        for layer in self.main_transformer:
            main_emb = layer(main_emb, mask)
        
        # S2 hierarchical embeddings
        s2_11_emb = self.pos_enc(self.s2_11_embed(batch['s2_11']))
        s2_13_emb = self.pos_enc(self.s2_13_embed(batch['s2_13']))
        s2_14_emb = self.pos_enc(self.s2_14_embed(batch['s2_14']))
        s2_15_emb = self.pos_enc(self.s2_15_embed(batch['s2_15']))
        
        # Cross-attention: Q from main_emb, K/V from different streams
        out_main = self.cross_attn_main(main_emb, main_emb, mask)
        out_s2_11 = self.cross_attn_s2_11(main_emb, s2_11_emb, mask)
        out_s2_13 = self.cross_attn_s2_13(main_emb, s2_13_emb, mask)
        out_s2_14 = self.cross_attn_s2_14(main_emb, s2_14_emb, mask)
        out_s2_15 = self.cross_attn_s2_15(main_emb, s2_15_emb, mask)
        
        # Hierarchical gating
        gated = self.hier_gate([out_main, out_s2_11, out_s2_13, out_s2_14, out_s2_15])
        
        # Fusion Transformer
        for layer in self.fusion_transformer:
            gated = layer(gated, mask)
        
        # Attention pooling
        pooled = self.pool(gated, mask)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        
        # One-hot with smoothing
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        loss = (-targets_smooth * log_probs).sum(dim=-1).mean()
        return loss

def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_accum=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        logits = model(batch)
        loss = criterion(logits, batch['Y']) / grad_accum
        loss.backward()
        
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum:.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_results = np.zeros(7, dtype=np.float32)
    all_true, all_pred = [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        
        results, true_y, pred_top1 = calculate_correct_total_prediction(logits, batch['Y'])
        all_results += results
        all_true.extend(true_y.numpy())
        all_pred.extend(pred_top1.numpy())
    
    f1 = f1_score(all_true, all_pred, average='weighted')
    
    return get_performance_dict({
        "correct@1": all_results[0], "correct@3": all_results[1],
        "correct@5": all_results[2], "correct@10": all_results[3],
        "rr": all_results[4], "ndcg": all_results[5],
        "f1": f1, "total": all_results[6],
    })

def get_vocab_sizes(train_path, val_path, test_path):
    """Get vocabulary sizes from all splits."""
    vocab = {'locations': set(), 's2_11': set(), 's2_13': set(), 
             's2_14': set(), 's2_15': set(), 'users': set()}
    
    for path in [train_path, val_path, test_path]:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for d in data:
            vocab['locations'].update(d['X'].tolist())
            vocab['locations'].add(d['Y'])
            vocab['s2_11'].update(d['s2_level11_X'].tolist())
            vocab['s2_13'].update(d['s2_level13_X'].tolist())
            vocab['s2_14'].update(d['s2_level14_X'].tolist())
            vocab['s2_15'].update(d['s2_level15_X'].tolist())
            vocab['users'].update(d['user_X'].tolist())
    
    return {
        'num_locations': max(vocab['locations']) + 1,
        'num_s2_11': max(vocab['s2_11']) + 1,
        'num_s2_13': max(vocab['s2_13']) + 1,
        'num_s2_14': max(vocab['s2_14']) + 1,
        'num_s2_15': max(vocab['s2_15']) + 1,
        'num_users': max(vocab['users']) + 1,
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Paths
    DATA_DIR = "/content/expr_hrcl_next_pred_av3/data/temp"
    TRAIN_PATH = os.path.join(DATA_DIR, "geolife_transformer_7_train.pk")
    VAL_PATH = os.path.join(DATA_DIR, "geolife_transformer_7_validation.pk")
    TEST_PATH = os.path.join(DATA_DIR, "geolife_transformer_7_test.pk")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get vocab sizes
    print("Loading vocabulary sizes...")
    vocab = get_vocab_sizes(TRAIN_PATH, VAL_PATH, TEST_PATH)
    print(f"Vocab sizes: {vocab}")
    
    # Hyperparameters
    config = {
        'batch_size': 64,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.15,
        'lr': 2e-3,
        'weight_decay': 0.01,
        'epochs': 100,
        'patience': 15,
        'label_smoothing': 0.1,
        'grad_accum': 1,
    }
    
    # Datasets
    print("Loading datasets...")
    train_ds = GeoLifeDataset(TRAIN_PATH)
    val_ds = GeoLifeDataset(VAL_PATH)
    test_ds = GeoLifeDataset(TEST_PATH)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, config['batch_size'], shuffle=True, 
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, config['batch_size'], shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, config['batch_size'], shuffle=False,
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)
    
    # Model
    model = HierarchicalTransformerPyramidV2(
        num_locations=vocab['num_locations'],
        num_s2_11=vocab['num_s2_11'],
        num_s2_13=vocab['num_s2_13'],
        num_s2_14=vocab['num_s2_14'],
        num_s2_15=vocab['num_s2_15'],
        num_users=vocab['num_users'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    assert n_params < 1_000_000, f"Model has {n_params} params, exceeds 1M!"
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['lr'], 
                      weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['epochs'] // config['grad_accum']
    scheduler = OneCycleLR(optimizer, max_lr=config['lr'], total_steps=total_steps,
                          pct_start=0.1, anneal_strategy='cos')
    
    criterion = LabelSmoothingCrossEntropy(config['label_smoothing'])
    
    # Training
    best_val_acc = 0.0
    best_state = None
    patience = 0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                 criterion, device, config['grad_accum'])
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Acc@1: {val_metrics['acc@1']:.2f}%, Acc@5: {val_metrics['acc@5']:.2f}%")
        print(f"  Val MRR: {val_metrics['mrr']:.2f}%")
        
        if val_metrics['acc@1'] > best_val_acc:
            best_val_acc = val_metrics['acc@1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            print(f"  ★ New best! Acc@1: {best_val_acc:.2f}%")
        else:
            patience += 1
            if patience >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model and evaluate
    if best_state:
        model.load_state_dict(best_state)
    
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Acc@1:  {test_metrics['acc@1']:.2f}%")
    print(f"  Acc@5:  {test_metrics['acc@5']:.2f}%")
    print(f"  Acc@10: {test_metrics['acc@10']:.2f}%")
    print(f"  MRR:    {test_metrics['mrr']:.2f}%")
    print(f"  NDCG:   {test_metrics['ndcg']:.2f}%")
    print(f"  F1:     {test_metrics['f1']:.4f}")
    
    if test_metrics['acc@1'] >= 50.0:
        print("\n✓ TARGET ACHIEVED: Acc@1 >= 50%!")
    elif test_metrics['acc@1'] >= 45.0:
        print("\n~ Minimum acceptable: Acc@1 >= 45%")
    else:
        print("\n✗ Below minimum threshold")
    
    return test_metrics

if __name__ == "__main__":
    main()
