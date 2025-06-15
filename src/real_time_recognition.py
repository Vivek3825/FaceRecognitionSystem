# step3_face_recognition.py

import cv2
from PIL import Image
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Load known faces
import csv
import numpy as np

# Load metadata
csv_path = "dataset/embeddings/embeddings.csv"
npz_path = "dataset/embeddings/all_embeddings.npz"

known_names = []
known_embeddings = []

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["Name"]
        key = row["Embedding Key"]
        known_names.append(name)
        # We'll load the embedding later using key

# Load .npz embeddings
npz_data = np.load(npz_path)

# Match embeddings to names using keys
for key in npz_data.files:
    embedding = npz_data[key]
    known_embeddings.append(embedding)

known_embeddings = np.array(known_embeddings)


# Normalize known embeddings
from sklearn.preprocessing import normalize
known_embeddings = normalize(known_embeddings)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Face recognition running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    boxes, _ = mtcnn.detect(pil_img)
    faces = mtcnn(pil_img)

    # ✅ Handle multiple faces
    if isinstance(faces, torch.Tensor) and faces.ndim == 4:
        faces = [f for f in faces]
    elif isinstance(faces, torch.Tensor) and faces.ndim == 3:
        faces = [faces]
    elif faces is None:
        faces = []

    if boxes is not None and len(faces) > 0:
        for box, face in zip(boxes, faces):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Preprocess
            face_input = face.unsqueeze(0).to(DEVICE)
            face_input = (face_input - 0.5) / 0.5

            with torch.no_grad():
                embedding = model(face_input).squeeze().cpu().numpy()

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Correct the function call and variable usage
            def match_face(embedding, known_embeddings, known_names, threshold=0.75):
                embedding = embedding / np.linalg.norm(embedding)

                similarities = cosine_similarity([embedding], known_embeddings)[0]
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]

                # Margin: difference between top-1 and top-2 scores
                sorted_similarities = np.sort(similarities)[::-1]
                margin = sorted_similarities[0] - sorted_similarities[1] if len(sorted_similarities) > 1 else 0.1

                if best_score >= threshold and margin >= 0.05:
                    return known_names[best_idx], best_score, "high_confidence"
                elif best_score >= (threshold - 0.1) and margin >= 0.08:
                    return known_names[best_idx], best_score, "medium_confidence"
                else:
                    return "Unknown", best_score, "low_confidence"

            # Call the function and use the results properly
            name, best_score, confidence = match_face(embedding, known_embeddings, known_names, threshold=0.75)

            # Color coding based on confidence
            if confidence == "high_confidence":
                color = (0, 255, 0)  # Green
            elif confidence == "medium_confidence":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            print(f"✅ Face recognized: {name} with score {best_score:.2f} ({confidence})")

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
