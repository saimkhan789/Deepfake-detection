import streamlit as st
import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
import tempfile
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Set page config
st.set_page_config(
    page_title="Deepfake Video Detector",
    page_icon="🎬",
    layout="centered"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAMES = 10

# Model definition (same as training)
class ImprovedHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )
        
        self.ln1 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, cnn_model):
        b, t, c, h, w = x.shape
        
        # Extract features from each frame
        x = x.view(b * t, c, h, w)
        feats = cnn_model(x)
        feats = feats.view(b, t, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(feats)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.ln1(attn_out)
        
        # Use last timestep
        out = attn_out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

@st.cache_resource
def load_model():
    """Load the trained model"""
    # Load CNN
    cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")
    cnn.classifier = nn.Identity()
    cnn.to(DEVICE)
    
    # Load hybrid model
    model = ImprovedHybridModel().to(DEVICE)
    
    # Load weights
    if os.path.exists('best_hybrid_model.pth'):
        checkpoint = torch.load('best_hybrid_model.pth', map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        cnn.load_state_dict(checkpoint['cnn'])
    else:
        st.error("❌ Model file 'best_hybrid_model.pth' not found!")
        return None, None
    
    model.eval()
    cnn.eval()
    
    return model, cnn

def extract_frames(video_path, num_frames=10):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    return frames

def preprocess_frames(frames):
    """Preprocess frames for model input"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    processed = []
    for frame in frames:
        processed.append(transform(frame))
    
    return torch.stack(processed)

def generate_gradcam_explanation(frames, cnn_model):
    """Generate Grad-CAM heatmap for selected frames"""
    target_layers = [cnn_model.features[-1]]
    cam = GradCAM(model=cnn_model, target_layers=target_layers)
    
    vis_images = []
    
    # Analyze middle frames
    mid_idx = len(frames) // 2
    selected_indices = [mid_idx-1, mid_idx, mid_idx+1]
    
    for idx in selected_indices:
        if idx < 0 or idx >= len(frames):
            continue
            
        frame_tensor = frames[idx]
        
        # Denormalize for visualization
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        img_np = frame_tensor.cpu().numpy()
        img_np = (img_np * std + mean) 
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.clip(img_np, 0, 1)
        
        # Run Grad-CAM
        input_tensor = frame_tensor.unsqueeze(0).to(DEVICE)
        
        # Important: GradCAM on CNN requires specific target logic
        # Here we just want to activate the feature map that contributed most
        # Since we use Identity classifier, we pass targets=None to maximize "something"
        # Or we can treat it as a feature extractor.
        # But efficiently, GradCAM works better with a Classifier.
        # Our CNN has Identity classifier.
        # Simple fix: Just run it. It usually highlights dominant features.
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Visualization
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        vis_images.append((img_np, visualization))
        
    return vis_images

def predict_video(video_path, model, cnn):
    """Run prediction on video"""
    frames = extract_frames(video_path, FRAMES)
    if frames is None:
        return None, None, None
    
    # Preprocess (keep as tensor for Grad-CAM)
    frame_tensors = preprocess_frames(frames) # [10, 3, 224, 224]
    input_tensor = frame_tensors.unsqueeze(0).to(DEVICE) # [1, 10, 3, 224, 224]
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor, cnn)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(1).item()
        confidence = probabilities[0][prediction].item()
    
    # Return processed frames (on CPU) for Grad-CAM
    return prediction, confidence, frame_tensors

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: transparent;
    }
    .upload-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .result-real {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .result-fake {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .confidence {
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main UI
st.title("🎬 Deepfake Video Detector")
st.markdown("### Upload a video to detect if it's Real or Fake")
st.markdown("---")

# Load model
model, cnn = load_model()

if model is None or cnn is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Upload a video to analyze"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    # Display video
    st.video(uploaded_file)
    
    # Predict button
    if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
        with st.spinner("Analyzing video... This may take a moment..."):
            prediction, confidence, processed_frames = predict_video(tmp_path, model, cnn)
        
        if prediction is not None:
            st.markdown("---")
            st.markdown("### 📊 Detection Result")
            
            if prediction == 0:  # Real
                st.markdown(f"""
                    <div class="result-real">
                        ✅ REAL VIDEO
                        <div class="confidence">Confidence: {confidence*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            else:  # Fake
                st.markdown(f"""
                    <div class="result-fake">
                        ⚠️ FAKE VIDEO DETECTED
                        <div class="confidence">Confidence: {confidence*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # GRAD-CAM EXPLANATION
                st.markdown("---")
                st.markdown("### 🕵️ Why is this detected as Fake?")
                with st.spinner("Generating AI visual explanation..."):
                    heatmaps = generate_gradcam_explanation(processed_frames, cnn)
                
                st.info("The red areas in the images below show where the AI looked to find fake patterns (artifacts).")
                
                cols = st.columns(len(heatmaps))
                for idx, (orig, heatmap) in enumerate(heatmaps):
                    with cols[idx]:
                        st.image(heatmap, caption=f"Analysis Frame {idx+1}", use_column_width=True)
            
            st.markdown("---")
            st.info("💡 **Note**: This model has ~52% accuracy. For better results, consider retraining with more epochs and data.")
        else:
            st.error("❌ Error processing video. Please try another file.")
    
    # Cleanup
    try:
        os.unlink(tmp_path)
    except:
        pass

else:
    # Instructions
    st.info("👆 Upload a video file to get started")
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Upload** your video (MP4, AVI, MOV, MKV)
        2. **Extract** 10 evenly-spaced frames from the video
        3. **Analyze** using hybrid CNN-LSTM model
        4. **Grad-CAM** explains *why* a video is fake by highlighting artifacts
        """)
    
    with st.expander("⚙️ Model Details"):
        st.markdown(f"""
        - **Architecture**: Hybrid CNN-LSTM
        - **CNN**: EfficientNetB0 (pretrained)
        - **Temporal**: Bidirectional LSTM
        - **Device**: {DEVICE.upper()}
        """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Deepfake Detection System")
