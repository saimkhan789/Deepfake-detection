import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

INPUT_DIR = "processed_frames"
OUTPUT_DIR = "dataset_clean"
IMG_SIZE = 224

model = YOLO("yolov8n.pt")  # OFFICIAL + STABLE

for label in ["real", "fake"]:
    src_root = os.path.join(INPUT_DIR, label)
    dst_root = os.path.join(OUTPUT_DIR, label)
    os.makedirs(dst_root, exist_ok=True)

    for video_folder in tqdm(os.listdir(src_root), desc=f"Processing {label}"):
        src_path = os.path.join(src_root, video_folder)
        dst_path = os.path.join(dst_root, video_folder)
        os.makedirs(dst_path, exist_ok=True)

        for img_name in os.listdir(src_path):
            img_path = os.path.join(src_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            results = model(img, conf=0.5, verbose=False)

            if len(results[0].boxes) == 0:
                continue

            # take largest detected person box
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            x1, y1, x2, y2 = boxes[areas.argmax()].astype(int)

            # crop upper region (face is top ~40%)
            h = y2 - y1
            face = img[y1:y1 + int(0.4 * h), x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(dst_path, img_name), face)

print("YOLO-based face extraction complete.")
