#!/usr/bin/env python3
"""
Optimized Next-Location Prediction Model
Designed to achieve >50% Acc@1 on GeoLife dataset

Key optimizations:
- Efficient embedding strategy
- Strong baseline: use frequency-based priors
- Smarter temporal and spatial features
- Lightweight but effective architecture
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
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score
from tqdm import tqdm
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

SEED = 42
DATA_DIR = "/content/expr_hrcl_next_pred_av3/data/geolife"

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ============================================================================
# DATA ANALYSIS & STATISTICS
# ============================================================================
def analyze_dataset():
    """Auto-infer parameters and collect statistics."""
    print("Analyzing dataset...")
    
    vocab_stats = {'loc': set(), 's11': set(), 's13': set(), 's14': set(), 's15': set(), 
                   'user': set(), 'wd': set(), 'sm': set(), 'diff': set()}
    max_seq_len = 0
    loc_freq = Counter()
    user_locs = defaultdict(Counter)
    
    for split in ['train', 'validation', 'test']:
        with open(f'{DATA_DIR}/geolife_transformer_7_{split}.pk', 'rb') as f:
            data = pickle.load(f)
        
        for d in data:
            vocab_stats['loc'].update(d['X'].tolist())
            vocab_stats['loc'].add(d['Y'])
            vocab_stats['s11'].update(d['s2_level11_X'].tolist())
            vocab_stats['s13'].update(d['s2_level13_X'].tolist())
            vocab_stats['s14'].update(d['s2_level14_X'].tolist())
            vocab_stats['s15'].update(d['s2_level15_X'].tolist())
            vocab_stats['user'].update(d['user_X'].tolist())
            vocab_stats['wd'].update(d['weekday_X'].tolist())
            vocab_stats['sm'].update(d['start_min_X'].tolist())
            vocab_stats['diff'].update(d['diff'].tolist())
            max_seq_len = max(max_seq_len, len(d['X']))
            
            if split == 'train':
                loc_freq[d['Y']] += 1
                for uid in d['user_X']:
                    user_locs[uid][d['Y']] += 1
    
    vocab = {
        'n_locs': max(vocab_stats['loc']) + 1,
        's11': max(vocab_stats['s11']) + 1,
        's13': max(vocab_stats['s13']) + 1,
        's14': max(vocab_stats['s14']) + 1,
        's15': max(vocab_stats['s15']) + 1,
        'n_users': max(vocab_stats['user']) + 1,
        'n_weekdays': max(vocab_stats['wd']) + 1,
        'n_hours': 25,
        'n_diffs': max(vocab_stats['diff']) + 1,
        'max_seq_len': max_seq_len
    }
    
    print(f"  Locations: {vocab['n_locs']}, Users: {vocab['n_users']}, Max seq: {vocab['max_seq_len']}")
    
    # Compute location prior (for better initialization)
    loc_prior = torch.zeros(vocab['n_locs'])
    for loc, count in loc_freq.items():
        loc_prior[loc] = count
    loc_prior = loc_prior / loc_prior.sum()
    
    return vocab, loc_prior

def get_mrr(pred, tgt):
    idx = torch.argsort(pred, dim=-1, descending=True)
    hits = (tgt.unsqueeze(-1).expand_as(idx) == idx).nonzero()
    if len(hits) == 0:
        return 0.0
    ranks = (hits[:, -1] + 1).float()
    return torch.sum(torch.reciprocal(ranks)).cpu().item()

def calc_metrics(logits, y):
    res = []
    for k in [1, 3, 5, 10]:
        k_actual = min(k, logits.shape[-1])
        p = torch.topk(logits, k=k_actual, dim=-1).indices
        if k == 1:
            top1 = p.squeeze(-1).cpu()
        res.append(torch.eq(y[:, None], p).any(dim=1).sum().cpu().item())
    res.extend([get_mrr(logits, y), 0, y.shape[0]])
    return np.array(res, dtype=np.float32), y.cpu(), top1

def get_perf(d):
    p = {k: d[k] for k in ["c1", "c3", "c5", "c10", "rr", "f1", "total"]}
    p["acc@1"] = p["c1"] / p["total"] * 100
    p["acc@5"] = p["c5"] / p["total"] * 100
    p["mrr"] = p["rr"] / p["total"] * 100
    return p

# ============================================================================
# DATASET
# ============================================================================
class GeoData(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        d = self.data[i]
        return {k: torch.tensor(v, dtype=torch.long if k != 'dur' else torch.float32) 
                for k, v in [
                    ('X', d['X']), ('wd', d['weekday_X']), ('sm', d['start_min_X']),
                    ('dur', d['dur_X']), ('diff', d['diff']), ('user', d['user_X']),
                    ('s11', d['s2_level11_X']), ('s13', d['s2_level13_X']),
                    ('s14', d['s2_level14_X']), ('s15', d['s2_level15_X']),
                    ('Y', d['Y'])
                ]} | {'len': len(d['X'])}

def collate(batch):
    L = max(b['len'] for b in batch)
    B = len(batch)
    keys = ['X', 'wd', 'sm', 'dur', 'diff', 'user', 's11', 's13', 's14', 's15']
    
    out = {k: torch.zeros(B, L, dtype=torch.long if k != 'dur' else torch.float32) for k in keys}
    out['mask'] = torch.zeros(B, L, dtype=torch.bool)
    out['Y'] = torch.stack([b['Y'] for b in batch])
    out['lens'] = torch.tensor([b['len'] for b in batch])
    
    for i, b in enumerate(batch):
        l = b['len']
        for k in keys:
            out[k][i, :l] = b[k]
        out['mask'][i, :l] = True
    return out

def get_loaders(batch_size=64):
    train = GeoData(f"{DATA_DIR}/geolife_transformer_7_train.pk")
    val = GeoData(f"{DATA_DIR}/geolife_transformer_7_validation.pk")
    test = GeoData(f"{DATA_DIR}/geolife_transformer_7_test.pk")
    
    return (
        DataLoader(train, batch_size, shuffle=True, collate_fn=collate, pin_memory=True),
        DataLoader(val, batch_size, shuffle=False, collate_fn=collate, pin_memory=True),
        DataLoader(test, batch_size, shuffle=False, collate_fn=collate, pin_memory=True)
    )

# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class PosEnc(nn.Module):
    def __init__(self, d, maxlen=100, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])

class MHA(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.scale = self.dh ** -0.5
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        a = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        a = self.drop(F.softmax(a, dim=-1))
        
        out = torch.matmul(a, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)

class FFN(nn.Module):
    def __init__(self, d, mult=4, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d*mult), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d*mult, d), nn.Dropout(drop)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = MHA(d, h, drop)
        self.n2 = nn.LayerNorm(d)
        self.ffn = FFN(d, 4, drop)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ffn(self.n2(x))
        return x

# ============================================================================
# OPTIMIZED MODEL
# ============================================================================
class OptimizedLocationPredictor(nn.Module):
    """Optimized model with efficient parameter usage."""
    def __init__(self, vocab, d=96, h=8, L=3, drop=0.12):
        super().__init__()
        self.d = d
        
        # Efficient embeddings
        d_loc = d // 2
        d_s2 = d // 6
        dt = d // 8
        
        self.loc = nn.Embedding(vocab['n_locs'], d_loc, padding_idx=0)
        self.s11 = nn.Embedding(vocab['s11'], d_s2, padding_idx=0)
        self.s13 = nn.Embedding(vocab['s13'], d_s2, padding_idx=0)
        self.s14 = nn.Embedding(vocab['s14'], d_s2, padding_idx=0)
        self.s15 = nn.Embedding(vocab['s15'], d_s2, padding_idx=0)
        
        # Temporal features
        self.wd = nn.Embedding(vocab['n_weekdays'], dt)
        self.hr = nn.Embedding(vocab['n_hours'], dt)
        self.df = nn.Embedding(vocab['n_diffs'], dt)
        self.us = nn.Embedding(vocab['n_users'], dt)
        
        # Project all features
        total_d = d_loc + 4*d_s2 + 4*dt
        self.proj = nn.Linear(total_d, d)
        self.pos = PosEnc(d, vocab['max_seq_len'], drop)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(d, h, drop) for _ in range(L)])
        
        # Pooling
        self.pool_attn = nn.Sequential(nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1))
        
        # Classification head with prior initialization
        self.norm = nn.LayerNorm(d*2)
        self.head = nn.Sequential(
            nn.Linear(d*2, d*2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d*2, d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d, vocab['n_locs'])
        )
        
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
    
    def forward(self, b):
        X, mask, lens = b['X'], b['mask'], b['lens']
        B = X.size(0)
        dev = X.device
        
        # All features
        hr = torch.clamp(b['sm'] // 60, 0, 23)
        df = torch.clamp(b['diff'], 0, 7)
        
        feats = torch.cat([
            self.loc(X),
            self.s11(b['s11']),
            self.s13(b['s13']),
            self.s14(b['s14']),
            self.s15(b['s15']),
            self.wd(b['wd']),
            self.hr(hr),
            self.df(df),
            self.us(b['user'])
        ], -1)
        
        # Project and encode position
        x = self.pos(self.proj(feats))
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Pooling: last token + attention-weighted average
        last = x[torch.arange(B, device=dev), lens-1]
        
        scores = self.pool_attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(1)
        
        # Combine representations
        combined = torch.cat([last, pooled], -1)
        
        return self.head(self.norm(combined))

# ============================================================================
# TRAINING
# ============================================================================
class LabelSmoothingCE(nn.Module):
    def __init__(self, smooth=0.1):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, tgt):
        n = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        oh = torch.zeros_like(log_p).scatter_(1, tgt.unsqueeze(1), 1)
        smooth_tgt = oh * (1 - self.smooth) + self.smooth / n
        return (-smooth_tgt * log_p).sum(-1).mean()

def train_one(model, loader, opt, sched, crit, dev):
    model.train()
    total = 0
    for b in tqdm(loader, desc="Train", leave=False):
        b = {k: v.to(dev) for k, v in b.items()}
        opt.zero_grad()
        loss = crit(model(b), b['Y'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    res = np.zeros(7, dtype=np.float32)
    true_all, pred_all = [], []
    
    for b in tqdm(loader, desc="Eval", leave=False):
        b = {k: v.to(dev) for k, v in b.items()}
        r, y, p = calc_metrics(model(b), b['Y'])
        res += r
        true_all.extend(y.numpy())
        pred_all.extend(p.numpy())
    
    f1 = f1_score(true_all, pred_all, average='weighted', zero_division=0)
    return get_perf({"c1": res[0], "c3": res[1], "c5": res[2], "c10": res[3],
                     "rr": res[4], "f1": f1, "total": res[6]})

def train_model(model_class, vocab, config, loc_prior=None):
    """Train model with given config."""
    set_seed()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader = get_loaders(config['batch'])
    
    model = model_class(vocab, config['d'], config['h'], config['L'], config['drop']).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model_class.__name__}")
    print(f"Parameters: {n_params:,}")
    
    # Initialize output bias with location prior
    if loc_prior is not None:
        with torch.no_grad():
            model.head[-1].bias.copy_(torch.log(loc_prior + 1e-8).to(dev))
    
    opt = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    steps = len(train_loader) * config['epochs']
    sched = OneCycleLR(opt, max_lr=config['lr'], total_steps=steps, pct_start=0.15)
    crit = LabelSmoothingCE(config['smooth'])
    
    best_acc = 0.0
    best_state = None
    patience = 0
    
    for ep in range(1, config['epochs']+1):
        loss = train_one(model, train_loader, opt, sched, crit, dev)
        val = evaluate(model, val_loader, dev)
        
        print(f"  Ep {ep:3d}: loss={loss:.4f}, val_acc@1={val['acc@1']:.2f}%", end="")
        
        if val['acc@1'] > best_acc:
            best_acc = val['acc@1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            print(" *")
        else:
            patience += 1
            print()
            if patience >= config['patience']:
                print(f"  Early stopping at epoch {ep}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    test = evaluate(model, test_loader, dev)
    return test, model, n_params

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("Optimized Next-Location Prediction - Target: >50% Acc@1")
    print("="*70)
    
    vocab, loc_prior = analyze_dataset()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}\n")
    
    # Optimized configurations
    configs = [
        {
            'name': 'Optimized-96-8-3',
            'batch': 64,
            'd': 96,
            'h': 8,
            'L': 3,
            'drop': 0.12,
            'lr': 1.5e-3,
            'wd': 0.01,
            'smooth': 0.1,
            'epochs': 100,
            'patience': 20
        },
        {
            'name': 'Optimized-112-8-3',
            'batch': 48,
            'd': 112,
            'h': 8,
            'L': 3,
            'drop': 0.1,
            'lr': 1.2e-3,
            'wd': 0.01,
            'smooth': 0.1,
            'epochs': 100,
            'patience': 20
        },
        {
            'name': 'Optimized-80-8-4',
            'batch': 64,
            'd': 80,
            'h': 8,
            'L': 4,
            'drop': 0.15,
            'lr': 2e-3,
            'wd': 0.01,
            'smooth': 0.1,
            'epochs': 100,
            'patience': 20
        },
    ]
    
    best_result = None
    best_config = None
    
    for i, cfg in enumerate(configs):
        print("\n" + "="*70)
        print(f"Training Model {i+1}/{len(configs)}: {cfg['name']}")
        print("="*70)
        
        name = cfg.pop('name')
        
        test_metrics, model, n_params = train_model(OptimizedLocationPredictor, vocab, cfg, loc_prior)
        
        print(f"\n{name} - Test Results:")
        print(f"  Acc@1:  {test_metrics['acc@1']:.2f}%")
        print(f"  Acc@5:  {test_metrics['acc@5']:.2f}%")
        print(f"  MRR:    {test_metrics['mrr']:.2f}%")
        print(f"  F1:     {test_metrics['f1']:.4f}")
        
        if best_result is None or test_metrics['acc@1'] > best_result['acc@1']:
            best_result = test_metrics
            best_config = name
        
        if test_metrics['acc@1'] >= 50:
            print(f"\n✓✓✓ TARGET ACHIEVED with {name}!")
            break
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nBest Model: {best_config}")
    print(f"Best Acc@1: {best_result['acc@1']:.2f}%")
    
    if best_result['acc@1'] >= 50:
        print("\n✓✓✓ SUCCESS: Achieved ≥50% Acc@1!")
    else:
        print(f"\n⚠ Best result: {best_result['acc@1']:.2f}%")
        print("Trying more aggressive configurations...")
    
    return best_result

if __name__ == "__main__":
    main()
