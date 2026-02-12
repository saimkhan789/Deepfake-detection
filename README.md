# 🎭 DeepFake Video Detection (Hybrid CNN–BiLSTM + Attention + Grad-CAM)

A complete end-to-end DeepFake video detection system combining spatial and temporal learning with explainable AI and real-time web deployment.

---

## 📌 Problem Statement

DeepFake technology has evolved rapidly, posing serious risks to digital trust, misinformation control, and identity security.

Most traditional detection methods rely on static CNN-based frame analysis. However, DeepFake videos contain **temporal inconsistencies** such as:

- Unnatural facial motion
- Irregular blinking patterns
- Frame-to-frame artifacts

This project addresses these limitations using a **Hybrid CNN–BiLSTM architecture with Attention**, combined with **Grad-CAM explainability**, and deployed through a **Streamlit web interface**.

---

## 🏗️ System Architecture

### 🔹 Overall Pipeline

1. **Video Upload**
2. Frame Extraction (10 evenly spaced frames)
3. Face Detection using YOLOv8
4. Feature Extraction using EfficientNet-B0
5. Temporal Modeling with Bi-LSTM
6. Multi-Head Attention
7. Final Classification (Real / Fake)
8. Grad-CAM Visualization
9. Web-Based Prediction Output

---

---

## 🧠 Model Architecture

### 🔹 Spatial Feature Extractor
- EfficientNet-B0 (ImageNet pretrained)
- Lower layers frozen
- Upper layers fine-tuned
- Output: 1280-D feature vector

### 🔹 Temporal Modeling
- 2-layer Bi-LSTM
- Hidden size: 256
- Dropout: 0.4–0.5

### 🔹 Attention Mechanism
- Multi-Head Attention (8 heads)
- Captures important temporal cues

### 🔹 Classification
- Fully Connected Layers
- Weighted Cross-Entropy Loss
- Weighted Random Sampler for class imbalance

---

## 📊 Dataset

- **Primary Dataset:** FaceForensics++
- Dataset link: https://www.kaggle.com/datasets/xdxd003/ff-c23
-  10 frames per video
- Balanced Real/Fake classes
- Normalized using ImageNet statistics

---

## 📈 Results

- Validation Accuracy: ~52%
- Limited by:
  - Small dataset (~2000 samples)
  - Only 30 training epochs

### Planned Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 🔍 Explainable AI

Grad-CAM is used to:
- Highlight suspicious facial regions
- Visualize manipulated artifacts
- Improve model transparency

---

## 🌐 Web Deployment (Streamlit)

### Features:
- Upload video (MP4, AVI, MOV, MKV)
- Automatic frame extraction
- Real-time inference
- Confidence score display
- Color-coded results
- Device information display

Run locally:

```bash
streamlit run app.py
```
The weights and processed frames are removed due to their larger storage
## Running Sequence:
- Extract frames → face_crop_yolo.py
- Analyze / check data → analyze_data.py
- Extract features → extract_features.py
- Train LSTM (optional baseline) → train_lstm_improved.py
- Train hybrid CNN–BiLSTM → hybrid_improved.py
- Generate Grad-CAM visualizations → visualize_gradcam.py
- Deploy / run web app → app.py
