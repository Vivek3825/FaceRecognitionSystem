import os
import csv
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import cv2
import ast
import json

# Initialize the face analysis app
app = FaceAnalysis(name='buffalo_l')  # Using a pre-trained model
app.prepare(ctx_id=-1, det_size=(640, 640))

# Define paths
image_folder = "dataset/images"
csv_file = "dataset/info.csv"
embedding_output = "dataset/embeddings.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(embedding_output), exist_ok=True)

print(f"Reading data from: {csv_file}")
print(f"Images folder: {os.path.abspath(image_folder)}")

# Read existing CSV data
metadata = []
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header
    print(f"CSV header: {header}")
    for row in reader:
        metadata.append(row)

print(f"Found {len(metadata)} entries in CSV file")

# Process images and extract embeddings
embeddings_data = []
successful = 0
not_found = 0
no_face = 0

for row in metadata:
    sr_no, name, person_id, image_path, _ = row
    
    # Improved path handling
    basename = os.path.basename(image_path.replace('\\', '/'))
    img_path = os.path.join(image_folder, basename)
    
    print(f"Processing: {name} | ID: {person_id} | Image: {basename}")
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image (corrupt format): {img_path}")
            continue
        
        faces = app.get(img)
        if faces:
            # Debug print to verify embedding extraction
            print(f"Face found! Embedding length: {len(faces[0].normed_embedding)}")
            embedding = json.dumps(faces[0].normed_embedding.tolist())
            embeddings_data.append([sr_no, name, person_id, image_path, embedding])
            successful += 1
        else:
            print(f"No face detected in: {img_path}")
            no_face += 1
    else:
        print(f"Image not found: {img_path}")
        not_found += 1

print(f"\nProcessing summary:")
print(f"- Total images: {len(metadata)}")
print(f"- Successful embeddings: {successful}")
print(f"- Images not found: {not_found}")
print(f"- No face detected: {no_face}")

# Save extracted embeddings
if embeddings_data:
    with open(embedding_output, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Sr No.", "Name", "ID", "Image Path", "Embedding"])
        writer.writerows(embeddings_data)
    
    print(f"\nEmbedding extraction completed successfully.")
    print(f"Saved {len(embeddings_data)} embeddings to {embedding_output}")
else:
    print("\nNo embeddings were extracted! Check the issues above.")
