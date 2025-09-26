import os
import csv
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

# Initialize MTCNN face detector
mtcnn = MTCNN(
    keep_all=False,
    device='cpu',
    post_process=True,
    image_size=160
)

# Paths
original_csv = "Backend/dataset/info.csv"
face_output_folder = "Backend/dataset/faces"
new_csv = "Backend/dataset/face_info.csv"

os.makedirs(face_output_folder, exist_ok=True)

face_info_rows = []

with open(original_csv, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header

    for row in reader:
        sr_no, name, pid, img_path = row
        img_path = img_path.replace("\\", "/")

        if not os.path.exists(img_path):
            print(f"[❌] Not found: {img_path}")
            continue

        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[❌] Cannot open image: {img_path} | {e}")
            continue

        face_tensor = mtcnn(pil_img)

        if face_tensor is None:
            print(f"[⚠️] No face detected: {img_path}")
            continue

        try:
            # Convert tensor to numpy
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
            face_np = ((face_np + 1) * 127.5).astype(np.uint8)
            face_np = cv2.resize(face_np, (160, 160))
            face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

            # Save with PID and base name
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_name = f"{pid}_{base_name}.jpg"
            save_path = os.path.join(face_output_folder, save_name)
            cv2.imwrite(save_path, face_bgr)

            print(f"[✅] Saved: {save_path}")

            # Add new row with face image path
            face_info_rows.append([sr_no, name, pid, save_path.replace("\\", "/")])

        except Exception as e:
            print(f"[❌] Error processing {img_path}: {e}")

# Write new face_info.csv
with open(new_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Sr No.", "Name", "ID", "Image Path"])
    writer.writerows(face_info_rows)

print(f"\n✅ face_info.csv created with {len(face_info_rows)} entries at {new_csv}")
