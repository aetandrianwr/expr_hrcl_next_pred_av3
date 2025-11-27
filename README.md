# Hierarchical Transformer Pyramid for Next-Location Prediction

## Overview

This implementation provides a **Transformer Pyramid architecture** for next-location prediction on the GeoLife dataset. The key architectural concept is:

1. **Main stream (X)** serves as the **shared source of queries (Q)** 
2. **Parallel cross-attention blocks** where K/V come from different feature streams:
   - Main location embeddings
   - S2 Level 11 (coarse ~9km)
   - S2 Level 13 
   - S2 Level 14
   - S2 Level 15 (fine ~0.5km)
3. **Fusion Transformer** combines hierarchical information
4. **Classification head** predicts the next location

## Files

| File | Description |
|------|-------------|
| `next_location_prediction.py` | Basic implementation with all core features |
| `next_location_prediction_v2.py` | Enhanced with attention pooling, label smoothing, hierarchical gating |
| `next_location_prediction_v3.py` | Advanced with focal loss, mixup, recency bias, rotary encoding |
| `next_location_prediction_final.py` | Optimized production version with hyperparameter search |

## Requirements

```bash
pip install torch numpy scikit-learn tqdm
```

## Usage

1. Ensure the GeoLife dataset is at:
   ```
   /content/expr_hrcl_next_pred_av3/data/temp/
   ├── geolife_transformer_7_train.pk
   ├── geolife_transformer_7_validation.pk
   └── geolife_transformer_7_test.pk
   ```

2. Run any version:
   ```bash
   python next_location_prediction_final.py
   ```

## Architecture Details

### Parameter Budget (<1M)

With d_model=64, n_heads=4, n_layers=2:
- Location embeddings: ~1200 × 64 ≈ 77K
- S2 embeddings: ~3200 × 64 ≈ 205K
- Temporal embeddings: ~100 × 16 × 4 ≈ 6K
- Transformer layers: ~150K
- Cross-attention: ~200K
- Fusion & head: ~100K
- **Total: ~700-800K parameters**

### Key Components

1. **Embeddings**
   - Location: Maps location IDs to d_model vectors
   - S2 Hierarchical: Separate embeddings for each S2 level
   - Temporal: Weekday, hour, diff (days until target), user

2. **Main Transformer**
   - Pre-norm architecture for training stability
   - Multi-head self-attention with recency bias
   - SwiGLU-style feed-forward networks

3. **Parallel Cross-Attention**
   - Query from main stream
   - Keys/Values from each hierarchical stream
   - 5 parallel branches (main + 4 S2 levels)

4. **Hierarchical Fusion**
   - Learnable fusion weights
   - Concatenation + projection
   - Additional Transformer layers

5. **Pooling**
   - Multi-scale: last position + mean + attention pooling
   - Combines temporal and spatial information

6. **Classification**
   - LayerNorm → Linear → GELU → Dropout → Linear

### Training Techniques

- **Label Smoothing** (0.1): Prevents overconfident predictions
- **Focal Loss** (γ=1.5): Handles class imbalance
- **Mixup** (α=0.1): Data augmentation
- **OneCycleLR**: Better convergence
- **Gradient Clipping** (1.0): Training stability
- **Early Stopping** (patience=15): Prevents overfitting

## Expected Results

| Metric | Target | Minimum |
|--------|--------|---------|
| Acc@1  | ≥50%   | ≥45%    |
| Acc@5  | ~75%   | ~70%    |
| MRR    | ~60%   | ~55%    |

## Seed

All random components use seed **42** for reproducibility.

## Customization

To adjust hyperparameters, modify the `config` dict in `main()`:

```python
config = {
    'batch': 64,      # Batch size
    'd': 64,          # Model dimension
    'h': 4,           # Attention heads
    'L': 2,           # Transformer layers
    'drop': 0.15,     # Dropout rate
    'lr': 2e-3,       # Learning rate
    'wd': 0.01,       # Weight decay
    'smooth': 0.1,    # Label smoothing
    'epochs': 100,    # Max epochs
    'patience': 15    # Early stopping
}
```

## Troubleshooting

1. **Out of memory**: Reduce batch size to 32
2. **Underfitting**: Increase d_model to 80, reduce dropout
3. **Overfitting**: Increase dropout to 0.2, add more regularization
4. **Slow convergence**: Try different learning rates (1e-3, 3e-3)

## Citation

If using this code, please cite the GeoLife dataset:
- Yu Zheng, et al. "GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory"
