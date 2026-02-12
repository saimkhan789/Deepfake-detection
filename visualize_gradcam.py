import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Same model architecture as training
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

    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # Extract features from each frame
        x = x.view(b * t, c, h, w)
        feats = cnn(x)
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

# Load CNN
cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")
cnn.classifier = nn.Identity()
cnn.to(DEVICE)

# Load the trained model
model = ImprovedHybridModel().to(DEVICE)

checkpoint = torch.load('best_hybrid_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model'])
cnn.load_state_dict(checkpoint['cnn'])

model.eval()
cnn.eval()

print("Model loaded successfully!")

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Target layer for Grad-CAM (last convolutional layer of EfficientNet)
target_layers = [cnn.features[-1]]

# Create output directory
os.makedirs('gradcam_results/real', exist_ok=True)
os.makedirs('gradcam_results/fake', exist_ok=True)

def generate_gradcam_for_video(video_path, label_name, video_id, max_frames=5):
    """Generate Grad-CAM heatmaps for a video's frames"""
    
    frames = sorted(os.listdir(video_path))[:max_frames]
    
    if len(frames) == 0:
        print(f"No frames found in {video_path}")
        return
    
    print(f"\nProcessing {label_name} video {video_id}...")
    
    for frame_idx, frame_name in enumerate(frames):
        frame_path = os.path.join(video_path, frame_name)
        
        # Load and preprocess image
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Transform for model
        input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
        
        # Create Grad-CAM object
        cam = GradCAM(model=cnn, target_layers=target_layers)
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay CAM on image
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        
        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_resized)
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(grayscale_cam, cmap='jet')
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')
        
        # Overlayed
        axes[2].imshow(visualization)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')
        
        # Get prediction
        with torch.no_grad():
            output = cnn(input_tensor)
            # Since we're just looking at CNN features, we can't get full prediction
            # But we can show what the CNN sees
        
        plt.suptitle(f'{label_name.upper()} - Video {video_id} - Frame {frame_idx+1}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = f'gradcam_results/{label_name}/video{video_id}_frame{frame_idx+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")

# Process sample videos from both classes
DATA_DIR = "dataset_clean"
NUM_SAMPLES = 5  # Number of videos per class
FRAMES_PER_VIDEO = 3  # Frames to visualize per video

print(f"\nGenerating Grad-CAM visualizations...")
print(f"Processing {NUM_SAMPLES} videos per class, {FRAMES_PER_VIDEO} frames each")

# Process real videos
real_dir = os.path.join(DATA_DIR, "real")
real_videos = sorted(os.listdir(real_dir))[:NUM_SAMPLES]

for video_name in real_videos:
    video_path = os.path.join(real_dir, video_name)
    generate_gradcam_for_video(video_path, 'real', video_name, FRAMES_PER_VIDEO)

# Process fake videos
fake_dir = os.path.join(DATA_DIR, "fake")
fake_videos = sorted(os.listdir(fake_dir))[:NUM_SAMPLES]

for video_name in fake_videos:
    video_path = os.path.join(fake_dir, video_name)
    generate_gradcam_for_video(video_path, 'fake', video_name, FRAMES_PER_VIDEO)

print("\n" + "="*60)
print("Grad-CAM visualization completed!")
print(f"Results saved in: gradcam_results/")
print(f"  - gradcam_results/real/  ({NUM_SAMPLES} videos × {FRAMES_PER_VIDEO} frames)")
print(f"  - gradcam_results/fake/  ({NUM_SAMPLES} videos × {FRAMES_PER_VIDEO} frames)")
print("="*60)
