# src/match_face.py

import os
import csv
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import torch.nn.functional as F

# === Paths ===
EMBEDDING_NPZ = "dataset/embeddings/all_embeddings.npz"
EMBEDDING_CSV = "dataset/embeddings/embeddings.csv"
TEST_IMAGE = "dataset/test_images/test1.jpeg"
THRESHOLD = 0.6

# === Load model and detector ===
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, post_process=True)

# === Load known embeddings ===
data = np.load(EMBEDDING_NPZ)
keys = list(data.keys())
embeddings = np.array([data[k] for k in keys])

# === Load metadata ===
metadata = {}
with open(EMBEDDING_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata[row["Embedding Key"]] = {
            "Name": row["Name"],
            "ID": row["ID"],
            "Image Path": row["Image Path"]
        }

# === Detect face in test image ===
if not os.path.exists(TEST_IMAGE):
    print(f"[❌] Test image not found: {TEST_IMAGE}")
    exit()

img = Image.open(TEST_IMAGE).convert("RGB")
face = mtcnn(img)

if face is None:
    print("[❌] No face detected in test image.")
    exit()

# Normalize
face = face.unsqueeze(0)

# === Get embedding for test face ===
with torch.no_grad():
    test_embedding = model(face).squeeze()

# === Match against known embeddings ===
best_match = None
best_score = -1

for i, emb in enumerate(embeddings):
    emb_tensor = torch.tensor(emb)
    similarity = F.cosine_similarity(test_embedding, emb_tensor, dim=0).item()

    if similarity > best_score:
        best_score = similarity
        best_match = keys[i]

# === Result ===
print(f"\n🔍 Similarity: {best_score:.4f}")

if best_score >= THRESHOLD:
    person = metadata[best_match]
    print(f"[✅] Match found: {person['Name']} (ID: {person['ID']})")
    print(f"📷 Image: {person['Image Path']}")
else:
    print("[❌] Unknown face.")
