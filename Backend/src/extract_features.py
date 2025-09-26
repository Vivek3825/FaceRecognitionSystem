# src/extract_features.py

import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
import torch

# Paths (fixed to work from Backend directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)  # Go up from src to Backend
INFO_CSV = os.path.join(backend_dir, "dataset", "face_info.csv")
CSV_OUTPUT = os.path.join(backend_dir, "dataset", "embeddings", "embeddings.csv")
NPZ_OUTPUT = os.path.join(backend_dir, "dataset", "embeddings", "all_embeddings.npz")

# Ensure embeddings directory exists
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Step 1: Prepare new entries
new_rows = []
new_embeddings = {}
sr_no = 1

# Step 2: Read from face_info.csv
with open(INFO_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    info_rows = list(reader)

for row in tqdm(info_rows, desc="🔍 Extracting embeddings"):
    image_path = row["Image Path"]

    try:
        name = row["Name"]
        pid = row["ID"]

        if not os.path.isfile(image_path):
            print(f"[⚠️] Missing file: {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

        # Resize if needed
        if img_tensor.shape[1:] != (160, 160):
            img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(160, 160))[0]

        # Normalize
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = img_tensor.unsqueeze(0)

        # Run on CPU or GPU
        img_tensor = img_tensor.to(next(model.parameters()).device)

        # Extract embedding
        with torch.no_grad():
            embedding = model(img_tensor).squeeze().numpy()

        # Get base filename without extension (used as key)
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # P001_front
        key = base_name

        new_embeddings[key] = embedding
        new_rows.append([sr_no, name, pid, image_path, key])
        sr_no += 1

    except Exception as e:
        print(f"[❌] Error: {image_path}: {e}")
        continue

# Step 3: Save embeddings.csv
with open(CSV_OUTPUT, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Sr No.", "Name", "ID", "Image Path", "Embedding Key"])
    writer.writerows(new_rows)

# Step 4: Save all_embeddings.npz
np.savez(NPZ_OUTPUT, **new_embeddings)

# Final log
print(f"\n✅ Done! {len(new_rows)} embeddings generated.")
print(f"CSV saved to: {CSV_OUTPUT}")
print(f"NPZ saved to: {NPZ_OUTPUT}")
