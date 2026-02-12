# Deepfake Detection Project

## Project Structure

### Preprocessing Pipeline (Run in order)
1. `model.py` - Extracts frames from videos → `processed_frames/`
2. `face_crop_yolo.py` - Crops faces using YOLO → `dataset_clean/`
3. `extract_features.py` - (Optional) Extract CNN features → `X.npy`, `y.npy`

### Training Scripts

#### ⭐ **RECOMMENDED: Hybrid Model with Attention**
- **File**: `hybrid_improved.py`
- **Approach**: End-to-end CNN-LSTM with attention mechanism
- **Advantages**: Learns deepfake-specific features, handles class imbalance
- **Usage**: `python hybrid_improved.py`

#### Alternative: LSTM on Pre-extracted Features
- **File**: `train_lstm_improved.py`
- **Approach**: LSTM on EfficientNet features
- **Note**: Limited by feature quality (see data analysis)
- **Usage**: `python train_lstm_improved.py`

### Utility Scripts
- `analyze_data.py` - Analyze feature quality and class distribution

## Current Issue
- Pre-extracted EfficientNet features show minimal difference between real/fake
- Solution: Use `hybrid_improved.py` for end-to-end learning

## Dataset Stats
- Total videos: 5,949
- Real: 992 (16.7%)
- Fake: 4,957 (83.3%)
- Imbalance ratio: 1:5

## Files to Clean Up
Old training scripts (superseded by improved versions):
- `train_lstm.py` → use `train_lstm_improved.py` instead
- `finetune_cnn.py` → similar to old LSTM approach
- `hybrid_cnn_lstm.py` → use `hybrid_improved.py` instead

Possible duplicate data files:
- Check if `X_features.npy` == `X.npy`
- Check if `y_labels.npy` == `y.npy`
