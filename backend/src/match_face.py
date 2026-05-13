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
THRESHOLD = 0.6

# === Initialize models ===
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, post_process=True)

# Global variables for embeddings (loaded on demand)
embeddings_data = None
embeddings_keys = None
embeddings_array = None
embeddings_metadata = None

def load_embeddings():
    """Load embeddings data on demand"""
    global embeddings_data, embeddings_keys, embeddings_array, embeddings_metadata
    
    if embeddings_data is not None:
        return  # Already loaded
    
    try:
        # Load embeddings
        if os.path.exists(EMBEDDING_NPZ):
            embeddings_data = np.load(EMBEDDING_NPZ)
            embeddings_keys = list(embeddings_data.keys())
            embeddings_array = np.array([embeddings_data[k] for k in embeddings_keys])
        else:
            embeddings_data = {}
            embeddings_keys = []
            embeddings_array = np.array([])
        
        # Load metadata
        embeddings_metadata = {}
        if os.path.exists(EMBEDDING_CSV):
            with open(EMBEDDING_CSV, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    embeddings_metadata[row["Embedding Key"]] = {
                        "Name": row["Name"],
                        "ID": row["ID"],
                        "Image Path": row["Image Path"]
                    }
    except Exception as e:
        print(f"Warning: Could not load embeddings: {e}")
        embeddings_data = {}
        embeddings_keys = []
        embeddings_array = np.array([])
        embeddings_metadata = {}

def match_face_from_image(image_path):
    """
    Match a face from an image against the known embeddings database.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        dict: Contains match result with person info or unknown status
    """
    if not os.path.exists(image_path):
        return {
            "success": False,
            "error": f"Image not found: {image_path}"
        }

    try:
        img = Image.open(image_path).convert("RGB")
        face = mtcnn(img)

        if face is None:
            return {
                "success": False,
                "error": "No face detected in image"
            }

        # Normalize
        face = face.unsqueeze(0)

        # === Get embedding for face ===
        with torch.no_grad():
            test_embedding = model(face).squeeze()

        # Load embeddings if not already loaded
        load_embeddings()
        
        # === Match against known embeddings ===
        best_match = None
        best_score = -1

        if len(embeddings_array) == 0:
            return {
                "success": True,
                "match_found": False,
                "similarity": 0.0,
                "message": "No embeddings in database"
            }

        for i, emb in enumerate(embeddings_array):
            emb_tensor = torch.tensor(emb)
            similarity = F.cosine_similarity(test_embedding, emb_tensor, dim=0).item()

            if similarity > best_score:
                best_score = similarity
                best_match = embeddings_keys[i]

        # === Result ===
        if best_score >= THRESHOLD:
            person = embeddings_metadata[best_match]
            return {
                "success": True,
                "match_found": True,
                "similarity": best_score,
                "person": {
                    "name": person['Name'],
                    "id": person['ID'],
                    "image_path": person['Image Path']
                }
            }
        else:
            return {
                "success": True,
                "match_found": False,
                "similarity": best_score,
                "message": "Unknown face"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing image: {str(e)}"
        }


def match_face_from_embedding(face_embedding):
    """
    Match a face embedding against the known embeddings database.
    
    Args:
        face_embedding (torch.Tensor): Face embedding tensor
    
    Returns:
        dict: Contains match result with person info or unknown status
    """
    try:
        # Load embeddings if not already loaded
        load_embeddings()
        
        # === Match against known embeddings ===
        best_match = None
        best_score = -1

        if len(embeddings_array) == 0:
            return {
                "success": True,
                "match_found": False,
                "similarity": 0.0,
                "message": "No embeddings in database"
            }

        for i, emb in enumerate(embeddings_array):
            emb_tensor = torch.tensor(emb)
            similarity = F.cosine_similarity(face_embedding, emb_tensor, dim=0).item()

            if similarity > best_score:
                best_score = similarity
                best_match = embeddings_keys[i]

        # === Result ===
        if best_score >= THRESHOLD:
            person = embeddings_metadata[best_match]
            return {
                "success": True,
                "match_found": True,
                "similarity": best_score,
                "person": {
                    "name": person['Name'],
                    "id": person['ID'],
                    "image_path": person['Image Path']
                }
            }
        else:
            return {
                "success": True,
                "match_found": False,
                "similarity": best_score,
                "message": "Unknown face"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Error matching embedding: {str(e)}"
        }


# Example usage (can be removed if not needed)
if __name__ == "__main__":
    # This section can be used for testing with any image
    print("Face matching module loaded successfully!")
    print("Use match_face_from_image(image_path) to match faces.")
