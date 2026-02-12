import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Load features
X = np.load("X.npy")
y = np.load("y.npy")

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: Real={np.sum(y==0)}, Fake={np.sum(y==1)}")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SOLUTION 1: Weighted Random Sampling for class imbalance
class_counts = torch.bincount(y_train)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)  # Increased batch size
test_loader = DataLoader(test_ds, batch_size=32)

# SOLUTION 2: Improved LSTM with Dropout and Layer Normalization
class ImprovedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )
        self.ln1 = nn.LayerNorm(512)  # 256*2 for bidirectional
        
        # Additional LSTM layer for better feature extraction
        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.ln2 = nn.LayerNorm(256)  # 128*2
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # First LSTM layer
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        
        # Second LSTM layer
        out, _ = self.lstm2(out)
        out = self.ln2(out)
        
        # Take last timestep
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = ImprovedLSTM().to(device)

# SOLUTION 3: Weighted Cross Entropy Loss
class_weights_loss = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_loss)

# SOLUTION 4: Better optimizer with learning rate scheduling
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# SOLUTION 5: Early stopping
best_loss = float('inf')
patience = 7
patience_counter = 0

# Training with validation monitoring
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            val_loss += criterion(outputs, yb).item()
    
    avg_val_loss = val_loss / len(test_loader)
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('best_lstm_model.pth'))

# Evaluation
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        y_true.extend(yb.numpy())
        y_pred.extend(outputs.argmax(1).cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

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

# Calculate accuracy per class
real_acc = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
fake_acc = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
print(f"Real Detection Accuracy: {real_acc:.2%}")
print(f"Fake Detection Accuracy: {fake_acc:.2%}")
