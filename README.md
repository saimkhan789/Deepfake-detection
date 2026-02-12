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

## 📂 Project Structure
