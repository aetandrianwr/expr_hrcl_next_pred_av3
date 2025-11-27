# Next-Location Prediction Project - Summary Report

## Project Overview
**Goal:** Achieve ≥50% Acc@1 on GeoLife test set for next-location prediction

**Dataset:** GeoLife trajectory data with hierarchical spatial features (S2 cells at multiple levels)

## Implementation Details

### Auto-Inferred Parameters
All parameters are automatically inferred from the dataset:
- **Locations:** 1,187 unique locations
- **Users:** 46 users
- **Max sequence length:** 54 time steps
- **S2 levels:** Level 11, 13, 14, 15 (hierarchical spatial encoding)
- **Temporal features:** Weekday (7), Hour (24), Time difference (8)

### Dataset Structure
- **Training samples:** 7,424
- **Validation samples:** ~2,600
- **Test samples:** ~1,800
- **Data location:** `/content/expr_hrcl_next_pred_av3/data/geolife/`

## Models Implemented

### 1. `next_location_prediction_improved.py`
**Architecture:**
- Hierarchical Transformer Pyramid with cross-attention
- Simple Transformer baseline
- Hierarchical Transformer with simple concatenation

**Best Result:** Val Acc@1 ~37-38% (stopped early)

**Key Features:**
- Multi-model training
- Automatic parameter inference
- Cross-attention between main stream and hierarchical features

### 2. `next_location_prediction_enhanced.py`
**Architecture:**
- Deep Transformer (d=128, 8 heads, 4 layers)
- Enhanced with focal loss
- Multi-scale pooling

**Result:** Val Acc@1 ~38-40%, but model had 1.2M parameters (exceeded budget)

**Issues:**
- Model too large (>1M parameters)
- Training was slow
- Performance plateaued around 40%

### 3. `next_location_prediction_optimized.py`
**Architecture:**
- Optimized model with efficient embedding strategy
- Multiple configurations tested:
  - Optimized-96-8-3: d=96, 8 heads, 3 layers
  - Optimized-112-8-3: d=112, 8 heads, 3 layers
  - Optimized-80-8-4: d=80, 8 heads, 4 layers

**Best Result:** **Test Acc@1 = 33.61%**, Acc@5 = 54.45%

**Key Features:**
- Location prior initialization
- Efficient parameter usage
- Label smoothing
- OneCycleLR scheduler

### 4. `next_location_prediction_advanced.py`
**Architecture:**
- Large advanced model (d=160, 8 heads, 4 layers)
- Rich embeddings with better temporal encoding
- Deep classification head

**Status:** Training started but too slow; stopped early

## Best Performance Achieved

**Model:** Optimized-80-8-4
- **Test Acc@1:** 33.61%
- **Test Acc@5:** 54.45%
- **Test MRR:** 42.88%
- **Parameters:** <1M (within budget)

## Challenges & Analysis

### Why <50% Accuracy?

1. **Data Sparsity:** With 1,187 locations and relatively small training set, each location may have limited examples

2. **Complex User Behavior:** 46 users with potentially very different movement patterns; user-specific modeling might be needed

3. **Sequence Modeling Difficulty:** Next-location prediction is inherently difficult as human movement can be unpredictable

4. **Model Capacity vs. Dataset Size:** Balance between model expressiveness and overfitting

5. **Missing Context:** The model may lack important contextual information (e.g., purpose of trip, POI information, transportation mode)

## Recommendations for Future Improvement

To achieve ≥50% Acc@1, consider:

### 1. **User-Specific Modeling**
- Train separate models or heads for different users
- Use user embeddings more effectively
- Meta-learning approaches for few-shot adaptation

### 2. **Graph Neural Networks**
- Model spatial relationships as a graph
- Learn location embeddings that capture proximity
- Message passing between hierarchically related S2 cells

### 3. **Pre-training Strategies**
- Pre-train on larger trajectory datasets
- Self-supervised learning on masked trajectory prediction
- Transfer learning from related tasks

### 4. **Ensemble Methods**
- Combine multiple model predictions
- Use different architectures (RNN, GRU, Transformer, GNN)
- Weighted voting based on validation performance

### 5. **Data Augmentation**
- Temporal jittering
- Trajectory interpolation
- Synthetic trajectory generation

### 6. **External Knowledge**
- POI (Point of Interest) information
- Road network structure
- Temporal patterns (rush hour, weekends)
- Weather, events, etc.

### 7. **Advanced Attention Mechanisms**
- Sparse attention for long sequences
- Memory-augmented networks
- Retrieval-based methods

### 8. **Better Loss Functions**
- Ranking losses (contrastive, triplet)
- Multi-task learning (predict multiple future steps)
- Curriculum learning (start with easier examples)

## Files Created

1. `next_location_prediction_improved.py` - Multi-model baseline
2. `next_location_prediction_enhanced.py` - Deep enhanced model
3. `next_location_prediction_optimized.py` - Optimized lightweight model (BEST)
4. `next_location_prediction_advanced.py` - Advanced large model

## Conclusion

While the target of 50% Acc@1 was not achieved, significant progress was made:

✅ **Achievements:**
- Fully automated parameter inference from dataset
- Multiple model architectures implemented and tested
- Best model reaches 33.61% Acc@1, 54.45% Acc@5
- Clean, modular, well-documented code
- Efficient training with early stopping and learning rate scheduling

❌ **Gap to Target:**
- Need approximately 16.39 percentage points improvement
- Suggests fundamental architectural or data approach changes needed

**Next Steps:** Implement ensemble methods, graph neural networks, or pre-training strategies as outlined above.

## How to Use

```bash
# Run the best model
cd /content/expr_hrcl_next_pred_av3
python3 next_location_prediction_optimized.py

# Or any other version
python3 next_location_prediction_improved.py
python3 next_location_prediction_enhanced.py
python3 next_location_prediction_advanced.py
```

All models automatically:
- Infer vocab sizes and parameters from the dataset
- Train with best practices (early stopping, gradient clipping, etc.)
- Report comprehensive metrics (Acc@1, Acc@5, MRR, F1)
- Use GPU if available

---

**Date:** 2025-11-27
**Status:** Partial completion - Best achieved 33.61% Acc@1 (target was 50%)
