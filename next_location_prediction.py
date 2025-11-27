#!/usr/bin/env python3
"""
Hierarchical Transformer Pyramid for Next-Location Prediction
GeoLife Dataset

Architecture:
- Main location history (X) serves as shared queries (Q)
- Parallel Transformer blocks with K/V from different streams:
  - Stream 0: Main X embeddings
  - Stream 1-4: S2 levels (11, 13, 14, 15)
- Fusion Transformer to combine hierarchical information
- Classification head for next location prediction

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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.metrics import f1_score
from tqdm import tqdm

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
    def __init__(self, data_path, max_seq_len=51):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Main sequence
        X = torch.tensor(sample['X'], dtype=torch.long)
        seq_len = len(X)
        
        # Temporal features
        weekday = torch.tensor(sample['weekday_X'], dtype=torch.long)
        start_min = torch.tensor(sample['start_min_X'], dtype=torch.long)
        dur = torch.tensor(sample['dur_X'], dtype=torch.float32)
        diff = torch.tensor(sample['diff'], dtype=torch.long)
        user = torch.tensor(sample['user_X'], dtype=torch.long)
        
        # S2 hierarchical features
        s2_11 = torch.tensor(sample['s2_level11_X'], dtype=torch.long)
        s2_13 = torch.tensor(sample['s2_level13_X'], dtype=torch.long)
        s2_14 = torch.tensor(sample['s2_level14_X'], dtype=torch.long)
        s2_15 = torch.tensor(sample['s2_level15_X'], dtype=torch.long)
        
        # Target
        Y = torch.tensor(sample['Y'], dtype=torch.long)
        
        return {
            'X': X,
            'weekday': weekday,
            'start_min': start_min,
            'dur': dur,
            'diff': diff,
            'user': user,
            's2_11': s2_11,
            's2_13': s2_13,
            's2_14': s2_14,
            's2_15': s2_15,
            'Y': Y,
            'seq_len': seq_len
        }

def collate_fn(batch):
    """Custom collate function with padding."""
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors
    X = torch.zeros(batch_size, max_len, dtype=torch.long)
    weekday = torch.zeros(batch_size, max_len, dtype=torch.long)
    start_min = torch.zeros(batch_size, max_len, dtype=torch.long)
    dur = torch.zeros(batch_size, max_len, dtype=torch.float32)
    diff = torch.zeros(batch_size, max_len, dtype=torch.long)
    user = torch.zeros(batch_size, max_len, dtype=torch.long)
    s2_11 = torch.zeros(batch_size, max_len, dtype=torch.long)
    s2_13 = torch.zeros(batch_size, max_len, dtype=torch.long)
    s2_14 = torch.zeros(batch_size, max_len, dtype=torch.long)
    s2_15 = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    Y = torch.zeros(batch_size, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        X[i, :seq_len] = item['X']
        weekday[i, :seq_len] = item['weekday']
        start_min[i, :seq_len] = item['start_min']
        dur[i, :seq_len] = item['dur']
        diff[i, :seq_len] = item['diff']
        user[i, :seq_len] = item['user']
        s2_11[i, :seq_len] = item['s2_11']
        s2_13[i, :seq_len] = item['s2_13']
        s2_14[i, :seq_len] = item['s2_14']
        s2_15[i, :seq_len] = item['s2_15']
        mask[i, :seq_len] = True
        Y[i] = item['Y']
        seq_lens[i] = seq_len
    
    return {
        'X': X,
        'weekday': weekday,
        'start_min': start_min,
        'dur': dur,
        'diff': diff,
        'user': user,
        's2_11': s2_11,
        's2_13': s2_13,
        's2_14': s2_14,
        's2_15': s2_15,
        'mask': mask,
        'Y': Y,
        'seq_lens': seq_lens
    }

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)
        return self.dropout(x)

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block where Q comes from one stream (query_input)
    and K, V come from another stream (kv_input).
    Uses pre-norm architecture for training stability.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query_input, kv_input, mask=None):
        """
        Args:
            query_input: (batch, seq_len, d_model) - source of queries
            kv_input: (batch, seq_len, d_model) - source of keys and values
            mask: (batch, seq_len) - attention mask (True = valid, False = padding)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = query_input.shape
        
        # Pre-norm
        q_normed = self.norm_q(query_input)
        kv_normed = self.norm_kv(kv_input)
        
        # Project to Q, K, V
        Q = self.q_proj(q_normed)
        K = self.k_proj(kv_normed)
        V = self.v_proj(kv_normed)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection with residual
        output = query_input + self.dropout(self.out_proj(attn_output))
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network with pre-norm."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x

class TransformerBlock(nn.Module):
    """Full Transformer block with self-attention and FFN."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CrossAttentionBlock(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        x = self.self_attn(x, x, mask)
        x = self.ffn(x)
        return x

class ParallelCrossAttention(nn.Module):
    """
    Parallel cross-attention blocks where Q comes from main stream
    and K, V come from different hierarchical streams.
    """
    def __init__(self, d_model, n_heads, n_streams, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_streams)
        ])
        self.ffn_blocks = nn.ModuleList([
            FeedForward(d_model, d_model * 2, dropout)
            for _ in range(n_streams)
        ])
        
    def forward(self, query_input, stream_inputs, mask=None):
        """
        Args:
            query_input: (batch, seq_len, d_model) - shared Q source
            stream_inputs: list of (batch, seq_len, d_model) - K/V sources
            mask: (batch, seq_len)
        Returns:
            outputs: list of (batch, seq_len, d_model)
        """
        outputs = []
        for i, (cross_attn, ffn, kv_input) in enumerate(
            zip(self.cross_attn_blocks, self.ffn_blocks, stream_inputs)
        ):
            out = cross_attn(query_input, kv_input, mask)
            out = ffn(out)
            outputs.append(out)
        return outputs

# ============================================================================
# MAIN MODEL
# ============================================================================

class HierarchicalTransformerPyramid(nn.Module):
    """
    Hierarchical Transformer Pyramid for Next-Location Prediction.
    
    Architecture:
    1. Embedding layer for locations and S2 hierarchical features
    2. Shared query from main location sequence (X)
    3. Parallel cross-attention blocks with K/V from:
       - Main X embeddings
       - S2 level 11, 13, 14, 15 embeddings
    4. Fusion Transformer to combine hierarchical outputs
    5. Classification head
    """
    def __init__(
        self,
        num_locations,
        num_s2_11,
        num_s2_13,
        num_s2_14,
        num_s2_15,
        num_weekdays=7,
        num_time_bins=96,  # 15-min bins
        num_diff=8,
        num_users=50,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
        max_seq_len=60
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_streams = 5  # X + 4 S2 levels
        
        # Location embeddings
        self.loc_embed = nn.Embedding(num_locations + 1, d_model, padding_idx=0)
        
        # S2 hierarchical embeddings
        self.s2_11_embed = nn.Embedding(num_s2_11 + 1, d_model, padding_idx=0)
        self.s2_13_embed = nn.Embedding(num_s2_13 + 1, d_model, padding_idx=0)
        self.s2_14_embed = nn.Embedding(num_s2_14 + 1, d_model, padding_idx=0)
        self.s2_15_embed = nn.Embedding(num_s2_15 + 1, d_model, padding_idx=0)
        
        # Temporal embeddings
        self.weekday_embed = nn.Embedding(num_weekdays + 1, d_model // 4, padding_idx=0)
        self.time_embed = nn.Embedding(num_time_bins + 1, d_model // 4, padding_idx=0)
        self.diff_embed = nn.Embedding(num_diff + 1, d_model // 4, padding_idx=0)
        self.user_embed = nn.Embedding(num_users + 1, d_model // 4, padding_idx=0)
        
        # Temporal projection
        self.temporal_proj = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Initial self-attention for main stream
        self.main_transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Parallel cross-attention blocks
        self.parallel_cross_attn = ParallelCrossAttention(
            d_model, n_heads, self.n_streams, dropout
        )
        
        # Fusion layer - project concatenated outputs
        self.fusion_proj = nn.Linear(d_model * self.n_streams, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # Fusion Transformer
        self.fusion_transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: dict with keys X, weekday, start_min, dur, diff, user,
                   s2_11, s2_13, s2_14, s2_15, mask, seq_lens
        Returns:
            logits: (batch, num_locations)
        """
        X = batch['X']
        weekday = batch['weekday']
        start_min = batch['start_min']
        diff = batch['diff']
        user = batch['user']
        s2_11 = batch['s2_11']
        s2_13 = batch['s2_13']
        s2_14 = batch['s2_14']
        s2_15 = batch['s2_15']
        mask = batch['mask']
        seq_lens = batch['seq_lens']
        
        # Convert start_min to time bins (15-min intervals)
        time_bins = torch.clamp(start_min // 15, 0, 95)
        
        # Embed main locations
        loc_emb = self.loc_embed(X)  # (batch, seq, d_model)
        
        # Embed temporal features
        weekday_emb = self.weekday_embed(weekday)
        time_emb = self.time_embed(time_bins)
        diff_emb = self.diff_embed(torch.clamp(diff, 0, 7))
        user_emb = self.user_embed(user)
        
        # Concatenate temporal embeddings
        temporal_emb = torch.cat([weekday_emb, time_emb, diff_emb, user_emb], dim=-1)
        temporal_emb = self.temporal_proj(temporal_emb)
        
        # Combine location and temporal embeddings
        main_emb = loc_emb + temporal_emb
        main_emb = self.pos_encoding(main_emb)
        
        # Process main stream through self-attention
        for layer in self.main_transformer:
            main_emb = layer(main_emb, mask)
        
        # Embed S2 hierarchical features
        s2_11_emb = self.pos_encoding(self.s2_11_embed(s2_11))
        s2_13_emb = self.pos_encoding(self.s2_13_embed(s2_13))
        s2_14_emb = self.pos_encoding(self.s2_14_embed(s2_14))
        s2_15_emb = self.pos_encoding(self.s2_15_embed(s2_15))
        
        # Prepare streams for parallel cross-attention
        stream_inputs = [main_emb, s2_11_emb, s2_13_emb, s2_14_emb, s2_15_emb]
        
        # Parallel cross-attention: Q from main_emb, K/V from each stream
        parallel_outputs = self.parallel_cross_attn(main_emb, stream_inputs, mask)
        
        # Concatenate parallel outputs
        concat_output = torch.cat(parallel_outputs, dim=-1)  # (batch, seq, d_model * n_streams)
        
        # Project back to d_model
        fused = self.fusion_proj(concat_output)
        fused = self.fusion_norm(fused)
        
        # Fusion Transformer
        for layer in self.fusion_transformer:
            fused = layer(fused, mask)
        
        # Final normalization
        fused = self.final_norm(fused)
        
        # Pool: use last valid position for each sequence
        batch_size = X.size(0)
        last_positions = seq_lens - 1
        pooled = fused[torch.arange(batch_size, device=X.device), last_positions]
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        logits = model(batch)
        loss = F.cross_entropy(logits, batch['Y'])
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model and compute metrics."""
    model.eval()
    
    all_results = np.zeros(7, dtype=np.float32)  # [top1, top3, top5, top10, rr, ndcg, total]
    all_true = []
    all_pred = []
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        logits = model(batch)
        
        results, true_y, pred_top1 = calculate_correct_total_prediction(logits, batch['Y'])
        all_results += results
        all_true.extend(true_y.numpy())
        all_pred.extend(pred_top1.numpy())
    
    # Compute F1 score
    f1 = f1_score(all_true, all_pred, average='weighted')
    
    return_dict = {
        "correct@1": all_results[0],
        "correct@3": all_results[1],
        "correct@5": all_results[2],
        "correct@10": all_results[3],
        "rr": all_results[4],
        "ndcg": all_results[5],
        "f1": f1,
        "total": all_results[6],
    }
    
    return get_performance_dict(return_dict)

def get_vocab_sizes(data_path):
    """Get vocabulary sizes from the dataset."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    all_X = np.concatenate([d['X'] for d in data])
    all_Y = np.array([d['Y'] for d in data])
    all_locs = np.concatenate([all_X, all_Y])
    
    all_s2_11 = np.concatenate([d['s2_level11_X'] for d in data])
    all_s2_13 = np.concatenate([d['s2_level13_X'] for d in data])
    all_s2_14 = np.concatenate([d['s2_level14_X'] for d in data])
    all_s2_15 = np.concatenate([d['s2_level15_X'] for d in data])
    
    all_users = np.concatenate([d['user_X'] for d in data])
    
    return {
        'num_locations': int(all_locs.max()) + 1,
        'num_s2_11': int(all_s2_11.max()) + 1,
        'num_s2_13': int(all_s2_13.max()) + 1,
        'num_s2_14': int(all_s2_14.max()) + 1,
        'num_s2_15': int(all_s2_15.max()) + 1,
        'num_users': int(all_users.max()) + 1,
    }

def count_parameters(model):
    """Count trainable parameters."""
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
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get vocabulary sizes
    print("Loading vocabulary sizes...")
    vocab_sizes = get_vocab_sizes(TRAIN_PATH)
    print(f"Vocabulary sizes: {vocab_sizes}")
    
    # Hyperparameters (tuned for <1M params and good performance)
    BATCH_SIZE = 64
    D_MODEL = 64
    N_HEADS = 4
    N_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 100
    PATIENCE = 15
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = GeoLifeDataset(TRAIN_PATH)
    val_dataset = GeoLifeDataset(VAL_PATH)
    test_dataset = GeoLifeDataset(TEST_PATH)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    # Create model
    model = HierarchicalTransformerPyramid(
        num_locations=vocab_sizes['num_locations'],
        num_s2_11=vocab_sizes['num_s2_11'],
        num_s2_13=vocab_sizes['num_s2_13'],
        num_s2_14=vocab_sizes['num_s2_14'],
        num_s2_15=vocab_sizes['num_s2_15'],
        num_users=vocab_sizes['num_users'],
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    assert num_params < 1_000_000, f"Model has {num_params} params, exceeds 1M limit!"
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # OneCycleLR for better convergence
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Acc@1: {val_metrics['acc@1']:.2f}%")
        print(f"  Val Acc@5: {val_metrics['acc@5']:.2f}%")
        print(f"  Val MRR: {val_metrics['mrr']:.2f}%")
        
        # Early stopping check
        if val_metrics['acc@1'] > best_val_acc:
            best_val_acc = val_metrics['acc@1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  New best model! Acc@1: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
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
    
    # Check if target achieved
    if test_metrics['acc@1'] >= 50.0:
        print("\n✓ TARGET ACHIEVED: Acc@1 >= 50%!")
    elif test_metrics['acc@1'] >= 45.0:
        print("\n~ Minimum acceptable: Acc@1 >= 45%")
    else:
        print("\n✗ Below minimum threshold")
    
    return test_metrics

if __name__ == "__main__":
    main()
