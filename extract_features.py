import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm

DATA_DIR = "dataset_clean"
FRAMES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = torch.nn.Identity()
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

X, y = [], []

def extract_from_folder(path):
    features = []
    frames = sorted(os.listdir(path))[:FRAMES]

    for img_name in frames:
        img = cv2.imread(os.path.join(path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model(img).cpu().numpy().squeeze()
        features.append(feat)

    if len(features) == FRAMES:
        return np.stack(features)
    return None

for label, cls in [("real", 0), ("fake", 1)]:
    folder = os.path.join(DATA_DIR, label)
    for video in tqdm(os.listdir(folder), desc=f"Extracting {label}"):
        path = os.path.join(folder, video)
        seq = extract_from_folder(path)
        if seq is not None:
            X.append(seq)
            y.append(cls)

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Feature extraction complete.")
print("X shape:", X.shape)
print("y shape:", y.shape)
