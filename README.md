# Next-Location Prediction on GeoLife Dataset

## 🎯 Project Goal
Achieve **≥50% Acc@1** on the GeoLife test set for next-location prediction using transformer-based models with <1M parameters.

## 📊 Best Result Achieved
- **Test Acc@1:** 33.61%
- **Test Acc@5:** 54.45%
- **Test MRR:** 42.88%
- **Model:** Optimized-80-8-4 (d=80, 8 heads, 4 layers)

## 🏗️ Architecture Overview

This implementation provides multiple **Transformer-based architectures** for next-location prediction on the GeoLife dataset, with automatic parameter inference from the dataset.

### Key Features
- **Automatic parameter inference** - No hardcoded values
- **Hierarchical spatial encoding** - S2 cells at levels 11, 13, 14, 15
- **Rich temporal features** - Weekday, hour, time-difference
- **Multi-scale pooling** - Last position + attention-weighted average
- **Advanced training** - Label smoothing, gradient clipping, early stopping, OneCycleLR

## 📁 Files

| File | Description | Status |
|------|-------------|--------|
| `next_location_prediction.py` | Original basic implementation | Legacy |
| `next_location_prediction_v2.py` | Enhanced with attention pooling, label smoothing | Legacy |
| `next_location_prediction_v3.py` | Advanced with focal loss, mixup | Legacy |
| `next_location_prediction_final.py` | Original optimized version | Legacy |
| `next_location_prediction_improved.py` | Multi-model baseline | ✅ Tested |
| `next_location_prediction_enhanced.py` | Deep model with focal loss | ✅ Tested |
| `next_location_prediction_optimized.py` | **Best performing model** | ✅ **BEST** |
| `next_location_prediction_advanced.py` | Large capacity model | ⚠️ Slow |
| `PROJECT_SUMMARY.md` | Detailed project analysis | 📄 Report |

## 💾 Dataset

### Location
```
/content/expr_hrcl_next_pred_av3/data/geolife/
├── geolife_transformer_7_train.pk (7,424 samples)
├── geolife_transformer_7_validation.pk (~2,600 samples)
└── geolife_transformer_7_test.pk (~1,800 samples)
```

### Auto-Inferred Parameters
All parameters are automatically detected from the dataset:
- **Locations:** 1,187 unique locations
- **Users:** 46 users
- **Max sequence length:** 54 time steps
- **S2 levels:** Level 11 (314 cells), 13 (673), 14 (926), 15 (1,249)
- **Temporal:** Weekday (7), Hour (24), Time-diff (8)

## 🚀 Usage

### Quick Start
```bash
# Run the best model
python next_location_prediction_optimized.py
```

### Run Other Models
```bash
python next_location_prediction_improved.py   # Multi-model baseline
python next_location_prediction_enhanced.py   # Deep model
python next_location_prediction_advanced.py   # Large model (slow)
```

All models automatically:
- ✅ Infer all parameters from dataset
- ✅ Train with best practices (early stopping, gradient clipping)
- ✅ Report metrics (Acc@1, Acc@5, Acc@10, MRR, F1)
- ✅ Use GPU if available

## 🏛️ Architecture Details

### Optimized Model (Best: 33.61% Acc@1)

**Parameters:** <1M (budget compliant)

**Configuration:**
```python
{
    'batch': 64,
    'd': 80,           # Model dimension
    'h': 8,            # Attention heads  
    'L': 4,            # Transformer layers
    'drop': 0.15,      # Dropout rate
    'lr': 2e-3,        # Learning rate
    'wd': 0.01,        # Weight decay
    'smooth': 0.1,     # Label smoothing
    'epochs': 100,
    'patience': 20
}
```

**Embeddings:**
- Location: d/2 = 40
- S2 cells (×4): d/6 = 13 each
- Temporal (×4): d/8 = 10 each
- Total features → Project to d=80

**Architecture:**
1. Feature concatenation + projection
2. Positional encoding
3. 4× Transformer blocks (8 heads, d=80)
4. Multi-scale pooling (last + attention)
5. Classification head (d×2 → d → n_locs)

## 📈 Performance Analysis

### Best Model Results
| Dataset | Acc@1 | Acc@5 | Acc@10 | MRR | F1 |
|---------|-------|-------|--------|-----|-----|
| Validation | ~40% | ~64% | ~72% | ~48% | ~0.30 |
| **Test** | **33.61%** | **54.45%** | ~64% | **42.88%** | **0.24** |

### Gap Analysis
**Target:** 50% Acc@1  
**Achieved:** 33.61% Acc@1  
**Gap:** -16.39 percentage points

### Why Below Target?

1. **Data Sparsity**
   - 1,187 locations with only 7,424 training samples
   - Many locations have very few examples
   - Long-tail distribution of location frequencies

2. **Complex User Behavior**
   - 46 users with diverse movement patterns
   - Limited personalization in current models
   - User-specific patterns not fully captured

3. **Model Limitations**
   - Transformer models may not fully capture spatial relationships
   - Lack of explicit graph structure for locations
   - Missing external knowledge (POIs, roads, semantics)

4. **Sequence Modeling Challenge**
   - Human movement is inherently unpredictable
   - Limited historical context (max 54 steps)
   - Multimodal destinations possible

## 🔧 Requirements

```bash
pip install torch numpy scikit-learn tqdm
```

**Tested with:**
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (GPU recommended)

## ⚡ Future Improvements

To achieve ≥50% Acc@1, consider:

### 1. Graph Neural Networks
- Model spatial relationships explicitly
- Message passing between nearby locations
- Hierarchical graph structure (S2 cells)

### 2. User-Specific Modeling
- Meta-learning for user adaptation
- User-specific embeddings or parameters
- Few-shot learning approaches

### 3. Pre-training Strategies
- Self-supervised pre-training on larger datasets
- Masked trajectory prediction
- Contrastive learning for location representations

### 4. Ensemble Methods
- Combine multiple model architectures
- Weighted voting or stacking
- Diverse models (RNN, GRU, Transformer, GNN)

### 5. External Knowledge
- POI (Point of Interest) features
- Road network topology
- Temporal patterns (events, weather)
- Semantic location categories

### 6. Advanced Techniques
- Memory-augmented networks
- Retrieval-based methods
- Multi-task learning
- Curriculum learning

## 📝 Training Techniques Used

- ✅ **Label Smoothing** (0.08-0.1): Prevents overconfidence
- ✅ **OneCycleLR**: Cosine annealing with warmup
- ✅ **Gradient Clipping** (1.0): Training stability
- ✅ **Early Stopping** (15-25 patience): Prevents overfitting
- ✅ **Dropout** (0.1-0.15): Regularization
- ✅ **Weight Decay** (0.01): L2 regularization
- ✅ **Multi-scale Pooling**: Last + attention-weighted average

## 🐛 Troubleshooting

1. **Out of memory**: Reduce batch size to 32 or 16
2. **Slow training**: Reduce model dimension or layers
3. **Underfitting**: Increase model capacity, reduce dropout
4. **Overfitting**: Increase dropout, add more regularization
5. **Poor convergence**: Adjust learning rate (try 1e-3, 3e-3)

## 📄 Citation

If using this code or the GeoLife dataset:
```
Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. 
"Mining interesting locations and travel sequences from GPS trajectories"
In Proceedings of International Conference on World Wild Web (WWW 2009), 
Madrid, Spain. ACM Press: 791-800.
```

## 🎓 Project Status

**Status:** ⚠️ Partial Completion  
**Best Achieved:** 33.61% Acc@1 (Target: 50%)  
**Models Trained:** 4+ different architectures  
**Parameter Budget:** ✅ Compliant (<1M parameters)  
**Code Quality:** ✅ Clean, modular, well-documented  

### What Works
✅ Automatic parameter inference from dataset  
✅ Multiple transformer architectures implemented  
✅ Proper training techniques (early stopping, LR scheduling)  
✅ Comprehensive evaluation metrics  
✅ GPU acceleration support  

### What's Missing
❌ 16.39 percentage points to reach 50% Acc@1 target  
❌ Advanced techniques (GNN, meta-learning, pre-training)  
❌ Ensemble methods  
❌ External knowledge integration  

## 📊 Repository Structure

```
expr_hrcl_next_pred_av3/
├── README.md                                  # This file
├── PROJECT_SUMMARY.md                         # Detailed analysis
├── next_location_prediction_optimized.py      # ⭐ Best model (33.61%)
├── next_location_prediction_improved.py       # Multi-model baseline
├── next_location_prediction_enhanced.py       # Deep model attempt
├── next_location_prediction_advanced.py       # Large model (slow)
├── next_location_prediction_final.py          # Original optimized
├── next_location_prediction_v3.py             # Advanced features
├── next_location_prediction_v2.py             # Enhanced version
├── next_location_prediction.py                # Basic version
└── data/geolife/                              # Dataset (not in repo)
    ├── geolife_transformer_7_train.pk
    ├── geolife_transformer_7_validation.pk
    └── geolife_transformer_7_test.pk
```

## 🤝 Contributing

This is a research project. Contributions to improve the model performance are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

Focus areas for contribution:
- Graph Neural Networks for spatial modeling
- User-specific modeling approaches
- Pre-training strategies
- Ensemble methods
- Novel attention mechanisms

---

**Last Updated:** November 27, 2025  
**Author:** GitHub Copilot CLI  
**License:** MIT
