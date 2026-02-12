import os
import cv2
from tqdm import tqdm

DATASET_PATH = "dataset/FaceForensics++_C23"
OUTPUT_PATH = "processed_frames"
FRAMES_PER_VIDEO = 10

REAL_FOLDERS = ["original"]
FAKE_FOLDERS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]

os.makedirs(os.path.join(OUTPUT_PATH, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "fake"), exist_ok=True)

def extract_frames(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < FRAMES_PER_VIDEO:
        cap.release()
        return

    frame_ids = [int(i * total / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]
    count = saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count in frame_ids:
            cv2.imwrite(os.path.join(save_dir, f"frame_{saved:02d}.jpg"), frame)
            saved += 1

        count += 1

    cap.release()

# REAL VIDEOS
for folder in REAL_FOLDERS:
    src = os.path.join(DATASET_PATH, folder)
    for video in tqdm(os.listdir(src), desc="Extracting REAL"):
        if not video.endswith(".mp4"):
            continue

        name = os.path.splitext(video)[0]
        out_dir = os.path.join(OUTPUT_PATH, "real", name)
        os.makedirs(out_dir, exist_ok=True)

        extract_frames(os.path.join(src, video), out_dir)

# FAKE VIDEOS
for folder in FAKE_FOLDERS:
    src = os.path.join(DATASET_PATH, folder)
    for video in tqdm(os.listdir(src), desc=f"Extracting FAKE ({folder})"):
        if not video.endswith(".mp4"):
            continue

        name = f"{os.path.splitext(video)[0]}_{folder}"
        out_dir = os.path.join(OUTPUT_PATH, "fake", name)
        os.makedirs(out_dir, exist_ok=True)

        extract_frames(os.path.join(src, video), out_dir)

print("Frame extraction complete.")
