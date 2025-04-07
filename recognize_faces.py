import cv2
import numpy as np
import os
import csv
from insightface.app import FaceAnalysis

# Initialize face recognition
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)

# Load embeddings from NPZ or NPY files
embedding_folder = "dataset/embeddings"
metadata_file = "dataset/embeddings.csv"

known_embeddings = []
metadata = []

print("[INFO] Loading known embeddings...")
with open(metadata_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        sr_no, name, person_id, _, embedding_path = row
        full_path = os.path.join(os.getcwd(), embedding_path)
        if os.path.exists(full_path):
            emb = np.load(full_path)
            known_embeddings.append(emb)
            metadata.append((name, person_id))
print(f"[INFO] Loaded {len(known_embeddings)} embeddings.")

# Compare function
def recognize_face(face_emb):
    max_sim = -1
    best_idx = -1
    for idx, known_emb in enumerate(known_embeddings):
        sim = np.dot(face_emb, known_emb)  # Cosine similarity
        if sim > max_sim:
            max_sim = sim
            best_idx = idx
    return best_idx, max_sim

# Try opening DroidCam or fallback to laptop camera
print("[INFO] Connecting to DroidCam...")

# DroidCam uses this URL format (check in the DroidCam app for your specific IP and port)
# For DroidCam, use port 4747 by default
droidcam_url = "http://100.124.23.75:4747/video"  # Adjust IP and port as needed

cap = cv2.VideoCapture(droidcam_url)
if not cap.isOpened():
    print(f"[ERROR] Could not connect to DroidCam at {droidcam_url}")
    print("[INFO] Falling back to laptop camera...")
    
    # Fallback to laptop camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No camera available. Exiting.")
        exit(1)
    print("[INFO] Using laptop camera.")
else:
    print("[INFO] Successfully connected to DroidCam.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    faces = app.get(frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.normed_embedding
        idx, score = recognize_face(embedding)

        name, pid = metadata[idx]
        confidence = round(score * 100, 1)

        label = f"{name} ({pid}) [{confidence}%]"
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

