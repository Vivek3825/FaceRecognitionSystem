import os
import csv
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import cv2

# Initialize the face analysis app
print("Initializing face analysis model...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))

# Define paths
image_folder = "dataset/images"
csv_file = "dataset/info.csv"
embedding_output = "dataset/embeddings.csv"
embedding_npz = "dataset/all_embeddings.npz"  # Main storage file

# Ensure output directory exists
os.makedirs(os.path.dirname(embedding_output), exist_ok=True)

print(f"Reading data from: {csv_file}")
print(f"Images folder: {os.path.abspath(image_folder)}")

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

# Load existing embeddings if available
all_embeddings = {}
if os.path.exists(embedding_npz):
    print(f"Loading existing embeddings from {embedding_npz}")
    try:
        npz_data = np.load(embedding_npz)
        all_embeddings = {key: npz_data[key] for key in npz_data.files}
        print(f"Loaded {len(all_embeddings)} existing embeddings")
    except Exception as e:
        print(f"Error loading existing embeddings: {e}")
        all_embeddings = {}

# Process images and extract embeddings
embeddings_data = []
successful = 0
not_found = 0
no_face = 0
skipped = 0

for row in metadata:
    sr_no, name, person_id, image_path = row
    
    # Improved path handling
    basename = os.path.basename(image_path)
    img_path = os.path.join(image_folder, basename)
    
    # Generate consistent embedding key
    embedding_key = f"{person_id}_{os.path.splitext(basename)[0]}"
    
    print(f"Processing: {name} | ID: {person_id} | Image: {basename}")
    
    # Check if embedding already exists in all_embeddings dictionary
    if embedding_key in all_embeddings:
        print(f"Embedding already exists for {name} ({basename})")
        embedding = all_embeddings[embedding_key]
        embeddings_data.append([sr_no, name, person_id, image_path, embedding_key])
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
            
            # Store in combined dictionary for NPZ file
            all_embeddings[embedding_key] = embedding
            
            # Store metadata and embedding reference
            embeddings_data.append([sr_no, name, person_id, image_path, embedding_key])
            successful += 1
        else:
            print(f"No face detected in: {img_path}")
            no_face += 1
    else:
        print(f"Image not found: {img_path}")
        not_found += 1

# Save combined NPZ file with error handling
if all_embeddings:
    try:
        np.savez(embedding_npz, **all_embeddings)
        print(f"Saved {len(all_embeddings)} embeddings to {embedding_npz}")
    except Exception as e:
        print(f"Error saving NPZ file: {str(e)}")
        # Try with alternative filename
        alt_npz = f"dataset/all_embeddings_new.npz"
        try:
            np.savez(alt_npz, **all_embeddings)
            print(f"Saved embeddings to alternative file: {alt_npz}")
        except Exception as e2:
            print(f"Failed to save NPZ file to alternative location: {str(e2)}")

print(f"\nProcessing summary:")
print(f"- Total images: {len(metadata)}")
print(f"- Successful embeddings: {successful}")
print(f"- Skipped (already processed): {skipped}")
print(f"- Images not found: {not_found}")
print(f"- No face detected: {no_face}")

# Save extracted embeddings metadata
if embeddings_data:
    try:
        with open(embedding_output, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Sr No.", "Name", "ID", "Image Path", "Embedding Key"])
            writer.writerows(embeddings_data)
        
        print(f"\nEmbedding extraction completed successfully.")
        print(f"Saved {len(embeddings_data)} embedding references to {embedding_output}")
    except PermissionError:
        print(f"\nERROR: Cannot write to {embedding_output}.")
        alt_output = f"dataset/embeddings_new.csv"
        with open(alt_output, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Sr No.", "Name", "ID", "Image Path", "Embedding Key"])
            writer.writerows(embeddings_data)
        print(f"Successfully wrote data to alternative file: {alt_output}")
else:
    print("\nNo embeddings were extracted! Check the issues above.")
