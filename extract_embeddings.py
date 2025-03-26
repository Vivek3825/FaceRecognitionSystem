import os
import csv
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import cv2
import uuid

# Initialize the face analysis app
app = FaceAnalysis(name='buffalo_l')  # Using a pre-trained model
app.prepare(ctx_id=-1, det_size=(640, 640))

# Define paths
image_folder = "dataset/images"
csv_file = "dataset/info.csv"
embedding_output = "dataset/embeddings.csv"
embedding_folder = "dataset/embeddings"
embedding_npz = "dataset/all_embeddings.npz"  # Optional: for combined storage

# Ensure output directories exist
os.makedirs(os.path.dirname(embedding_output), exist_ok=True)
os.makedirs(embedding_folder, exist_ok=True)

print(f"Reading data from: {csv_file}")
print(f"Images folder: {os.path.abspath(image_folder)}")
print(f"Embeddings will be stored in: {os.path.abspath(embedding_folder)}")

# Read existing CSV data and normalize paths
metadata = []
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header
    for row in reader:
        if len(row) >= 4:  # Ensure we have enough columns
            row[3] = row[3].replace('\\', '/')  # Normalize paths
        metadata.append(row)

print(f"Found {len(metadata)} entries in CSV file")

# Process images and extract embeddings
embeddings_data = []
successful = 0
not_found = 0
no_face = 0
skipped = 0
all_embeddings = {}  # For combined NPZ storage

for row in metadata:
    sr_no, name, person_id, image_path = row
    
    # Improved path handling
    basename = os.path.basename(image_path)
    img_path = os.path.join(image_folder, basename)
    
    # Generate a predictable embedding filename
    embedding_filename = f"{person_id}_{os.path.splitext(basename)[0]}.npy"
    embedding_path = os.path.join(embedding_folder, embedding_filename)
    rel_embedding_path = os.path.join("dataset/embeddings", embedding_filename)
    
    print(f"Processing: {name} | ID: {person_id} | Image: {basename}")
    
    # Check if embedding already exists to avoid duplicate processing
    if os.path.exists(embedding_path):
        print(f"Embedding already exists, skipping extraction...")
        embeddings_data.append([sr_no, name, person_id, image_path, rel_embedding_path])
        skipped += 1
        successful += 1
        continue
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image (corrupt format): {img_path}")
            continue
        
        faces = app.get(img)
        if faces:
            # Extract embedding as NumPy array
            embedding = faces[0].normed_embedding
            print(f"Face found! Embedding length: {len(embedding)}")
            
            # Save individual embedding as NumPy file
            np.save(embedding_path, embedding)
            
            # Also store in combined dictionary for NPZ file (optional)
            embedding_key = f"{person_id}_{os.path.splitext(basename)[0]}"
            all_embeddings[embedding_key] = embedding
            
            # Store metadata and embedding path
            embeddings_data.append([sr_no, name, person_id, image_path, rel_embedding_path])
            successful += 1
        else:
            print(f"No face detected in: {img_path}")
            no_face += 1
    else:
        print(f"Image not found: {img_path}")
        not_found += 1

# Save combined NPZ file (optional)
if all_embeddings:
    np.savez(embedding_npz, **all_embeddings)
    print(f"Saved combined embeddings to {embedding_npz}")

print(f"\nProcessing summary:")
print(f"- Total images: {len(metadata)}")
print(f"- Successful embeddings: {successful}")
print(f"- Skipped (already processed): {skipped}")
print(f"- Images not found: {not_found}")
print(f"- No face detected: {no_face}")

# Save extracted embeddings metadata - with error handling
if embeddings_data:
    try:
        with open(embedding_output, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Sr No.", "Name", "ID", "Image Path", "Embedding Path"])
            writer.writerows(embeddings_data)
        
        print(f"\nEmbedding extraction completed successfully.")
        print(f"Saved {len(embeddings_data)} embedding references to {embedding_output}")
    except PermissionError:
        print(f"\nERROR: Cannot write to {embedding_output}. The file may be open in another program.")
        alt_output = f"dataset/embeddings_{os.path.basename(os.path.splitext(embedding_output)[0])}_new.csv"
        print(f"Trying alternative filename: {alt_output}")
        with open(alt_output, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Sr No.", "Name", "ID", "Image Path", "Embedding Path"])
            writer.writerows(embeddings_data)
        print(f"Successfully wrote data to alternative file: {alt_output}")
else:
    print("\nNo embeddings were extracted! Check the issues above.")
