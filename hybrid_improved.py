import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DATA_DIR = "dataset_clean"
FRAMES = 10
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# No augmentation for validation
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class VideoDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = sorted(os.listdir(self.paths[idx]))[:FRAMES]

        if len(frames) != FRAMES:
            return None

        imgs = []
        for f in frames:
            img = cv2.imread(os.path.join(self.paths[idx], f))
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        return torch.stack(imgs), self.labels[idx]

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Load paths
paths, labels = [], []

for lbl, cls in [("real", 0), ("fake", 1)]:
    root = os.path.join(DATA_DIR, lbl)
    for v in os.listdir(root):
        paths.append(os.path.join(root, v))
        labels.append(cls)

print(f"Total videos (before sampling): {len(paths)}")
print(f"Real: {labels.count(0)}, Fake: {labels.count(1)}")

# Balance dataset: sample both classes equally to speed up training and fix imbalance
from sklearn.utils import resample

# Separate real and fake samples
real_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
fake_indices = [i for i, lbl in enumerate(labels) if lbl == 1]

real_paths = [paths[i] for i in real_indices]
real_labels = [labels[i] for i in real_indices]

fake_paths = [paths[i] for i in fake_indices]
fake_labels = [labels[i] for i in fake_indices]

# Balance classes: sample equal amounts from both (use the smaller class size)
SAMPLES_PER_CLASS = min(len(real_paths), len(fake_paths))
print(f"Balancing to {SAMPLES_PER_CLASS} samples per class...")

if len(real_paths) > SAMPLES_PER_CLASS:
    real_paths, real_labels = resample(
        real_paths, real_labels, 
        n_samples=SAMPLES_PER_CLASS, 
        random_state=42, 
        replace=False
    )

if len(fake_paths) > SAMPLES_PER_CLASS:
    fake_paths, fake_labels = resample(
        fake_paths, fake_labels, 
        n_samples=SAMPLES_PER_CLASS, 
        random_state=42, 
        replace=False
    )

# Combine balanced real and fake samples
paths = real_paths + fake_paths
labels = real_labels + fake_labels

print(f"\nTotal videos (after balanced sampling): {len(paths)}")
print(f"Real: {labels.count(0)}, Fake: {labels.count(1)}")

X_train, X_test, y_train, y_test = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Calculate sample weights for imbalance
class_counts = np.bincount(y_train)
class_weights_np = 1.0 / class_counts
sample_weights = [class_weights_np[label] for label in y_train]

# Weighted sampler
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    VideoDataset(X_train, y_train, train_transform),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=0
)

test_loader = DataLoader(
    VideoDataset(X_test, y_test, val_transform),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=0
)
#this is extracting the spatial features from teh individual frames
# Load pre-trained CNN (but we'll fine-tune it)
cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")

# IMPORTANT: Fine-tune the last few layers
# Freeze early layers
for param in cnn.features[:5].parameters():
    param.requires_grad = False

# Replace classifier with identity
cnn.classifier = nn.Identity()
cnn.to(DEVICE)

class ImprovedHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Bidirectional LSTM with more layers
        #processes these spatial features of each frame
#sequences to capture temporal consistency.
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=512,  # 256*2 from bidirectional
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
        
        # Self-attention to focus on important frames
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.ln1(attn_out)
        
        # Use last timestep + dropout
        out = attn_out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

model = ImprovedHybridModel().to(DEVICE)

# Calculate class weights for loss
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Use different learning rates for CNN and LSTM
optimizer = torch.optim.AdamW([
    {'params': cnn.parameters(), 'lr': LEARNING_RATE / 10},  # Slower for pre-trained
    {'params': model.lstm.parameters(), 'lr': LEARNING_RATE},
    {'params': model.attention.parameters(), 'lr': LEARNING_RATE},
    {'params': model.fc1.parameters(), 'lr': LEARNING_RATE},
    {'params': model.fc2.parameters(), 'lr': LEARNING_RATE}
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# Training with early stopping
best_val_loss = float('inf')
patience = 7
patience_counter = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    cnn.train()
    total_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for batch in train_pbar:
        if batch is None:
            continue
        xb, yb = batch
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    cnn.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            if batch is None:
                continue
            xb, yb = batch
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
            
            preds = outputs.argmax(1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = val_correct / val_total if val_total > 0 else 0
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'model': model.state_dict(),
            'cnn': cnn.state_dict()
        }, 'best_hybrid_model.pth')
        print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Load best model
checkpoint = torch.load('best_hybrid_model.pth')
model.load_state_dict(checkpoint['model'])
cnn.load_state_dict(checkpoint['cnn'])

# Final evaluation
model.eval()
cnn.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Final Evaluation"):
        if batch is None:
            continue
        xb, yb = batch
        xb = xb.to(DEVICE)
        out = model(xb)
        prob = torch.softmax(out, 1)[:, 1]

        y_true.extend(yb.numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_prob.extend(prob.cpu().numpy())

print("\n" + "="*60)
print("FINAL EVALUATION RESULTS")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(f"              Predicted Real  Predicted Fake")
print(f"Actual Real        {cm[0][0]:<6}         {cm[0][1]:<6}")
print(f"Actual Fake        {cm[1][0]:<6}         {cm[1][1]:<6}")

roc_auc = roc_auc_score(y_true, y_prob)
print(f"\nROC-AUC: {roc_auc:.4f}")

# Per-class accuracy
real_acc = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
fake_acc = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
print(f"\nReal Detection Accuracy: {real_acc:.2%}")
print(f"Fake Detection Accuracy: {fake_acc:.2%}")
