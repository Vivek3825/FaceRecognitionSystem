import cv2
import numpy as np
import os
import csv
import sys
import threading
import queue
import time
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis
print("Initializing FaceAnalysis model...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)

# Confidence threshold for recognition
CONFIDENCE_THRESHOLD = 0.5  # Set for better accuracy
FRAME_WIDTH, FRAME_HEIGHT = 640, 480  # Standard frame size

# Paths
embedding_folder = "dataset/embeddings"
metadata_file = "dataset/embeddings.csv"
embedding_npz = "dataset/all_embeddings.npz"  # Combined NPZ file

# Create thread-safe frame queue
frame_queue = queue.Queue(maxsize=2)  # Limit queue size to avoid memory issues

# Load known embeddings from NPZ file (faster than individual files)
print("[INFO] Loading .npz embeddings...")
try:
    npz_data = np.load(embedding_npz)
    embedding_keys = list(npz_data.keys())
    embedding_values = [npz_data[key] for key in embedding_keys]
    
    # Debug info
    print(f"[DEBUG] Loaded {len(embedding_keys)} embeddings")
    print(f"[DEBUG] First few keys: {embedding_keys[:3]}")

    # Load metadata - IMPORTANT FIX: Match embedding keys directly
    id_name_map = {}
    with open(metadata_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 5:  # Make sure we have all columns
                _, name, person_id, _, emb_key = row
                # Store the direct embedding key
                id_name_map[emb_key] = (name, person_id)
    
    # Debug info
    print(f"[DEBUG] Loaded {len(id_name_map)} metadata entries")
    print(f"[DEBUG] Example metadata: {list(id_name_map.items())[:2]}")
    
    # Verify key matching
    matches = 0
    for key in embedding_keys:
        if key in id_name_map:
            matches += 1
    print(f"[DEBUG] Matching keys: {matches}/{len(embedding_keys)}")
    
    print(f"[INFO] Loaded {len(embedding_values)} embeddings with {matches} matching metadata entries")
except Exception as e:
    print(f"[ERROR] Failed to load embeddings: {e}")
    sys.exit(1)

# Frame capture thread function
def capture_frames(source):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {source}")
        return

    print(f"[INFO] Started video capture thread from source {source}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame. Reconnecting...")
            time.sleep(1)
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            continue
            
        # Clear queue if full and put new frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
                
        frame_queue.put(frame)
    
    cap.release()

# Face matching function
def recognize_face(face_emb):
    max_sim = -1
    best_idx = -1
    best_key = None
    
    for idx, known_emb in enumerate(embedding_values):
        sim = np.dot(face_emb, known_emb)
        if sim > max_sim:
            max_sim = sim
            best_idx = idx
            best_key = embedding_keys[idx]
    
    if max_sim < CONFIDENCE_THRESHOLD:
        return "Unknown", (0, 0, 255), max_sim
    
    # IMPORTANT FIX: Use the direct key
    if best_key in id_name_map:
        name, pid = id_name_map[best_key]
        confidence = round(max_sim * 100, 1)
        return f"{name} ({pid}) [{confidence}%]", (0, 255, 0), max_sim
    else:
        # This case should rarely happen if keys match correctly
        print(f"[WARNING] Found embedding key {best_key} without metadata")
        return f"Unknown (Key: {best_key[:8]}...)", (255, 0, 0), max_sim  # type: ignore

# Camera selection function
def select_camera():
    while True:
        print("\n=== Camera Selection ===")
        print("1: Laptop webcam")
        print("2: USB camera")
        print("3: IP camera (DroidCam)")
        print("q: Quit")
        
        choice = input("Select camera type: ").strip().lower()
        
        if choice == 'q':
            print("Exiting...")
            sys.exit()
        elif choice == '1':
            return 0  # Default webcam
        elif choice == '2':
            # Try common USB camera indices
            for i in range(1, 5):
                print(f"Trying USB camera index {i}...")
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cap.release()
                        return i
                    cap.release()
            print("No USB camera found!")
            continue
        elif choice == '3':
            ip_address = input("Enter IP address (default: 192.168.0.100): ").strip()
            if not ip_address:
                ip_address = "192.168.0.100"
                
            port = input("Enter port (default: 4747): ").strip()
            if not port:
                port = "4747"
                
            # Return DroidCam URL
            return f"http://{ip_address}:{port}/video"
        else:
            print("Invalid selection, please try again.")

# Main application
def main():
    # Get camera source
    video_source = select_camera()
    
    # Start capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(video_source,), daemon=True)
    capture_thread.start()
    
    print("[INFO] Starting face recognition...")
    print("[INFO] Press 'q' to quit")
    
    fps_time = time.time()
    frame_count = 0
    
    while True:
        try:
            # Get frame with timeout
            frame = frame_queue.get(timeout=1.0)
            frame_count += 1
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                fps = frame_count / (current_time - fps_time)
                fps_time = current_time
                frame_count = 0
            else:
                fps = frame_count / (current_time - fps_time + 0.001)
            
            # Process frame for face recognition
            faces = app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.normed_embedding
                label, color, _ = recognize_face(embedding)

                # Draw face rectangle and label
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Show FPS and instructions
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Face Recognition", frame)
            
            # Check for key press - only quit option remains
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        except queue.Empty:
            print("[WARNING] No frames in queue")
            continue
        except Exception as e:
            print(f"[ERROR] {e}")
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
