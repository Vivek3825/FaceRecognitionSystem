# Multi-Camera Face Recognition Manager

import cv2
import threading
import queue
import time
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import csv
import configparser
import os
from typing import Dict, List, Optional, Tuple

class CameraStream:
    """Individual camera handler - captures frames in separate thread"""
    
    def __init__(self, camera_id: int, camera_name: str):
        # Basic camera info
        self.camera_id = camera_id
        self.camera_name = camera_name
        
        # Video capture objects
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Threading control
        self.is_running = False
        self.thread = None
        self.last_frame = None
        
    def start(self) -> bool:
        """Start camera capture in background thread"""
        try:
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return False
            
            # Test camera by reading one frame
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                return False
            
            # Set basic properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Start capture thread
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            print(f"✅ Started: {self.camera_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {self.camera_name}: {e}")
            return False
    
    def _capture_loop(self):
        """Background thread - continuously captures frames"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                # Add to queue (replace old frame if queue full)
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    try:
                        self.frame_queue.get(block=False)
                        self.frame_queue.put(frame, block=False)
                    except queue.Empty:
                        pass
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from camera"""
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return self.last_frame
    
    def stop(self):
        """Stop camera and cleanup"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()


class MultiCameraManager:
    """Main manager for multiple cameras and face recognition"""
    
    def __init__(self):
        print("🔄 Initializing Face Recognition System...")
        
        # Setup AI models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=160, post_process=True, keep_all=True, device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Camera storage
        self.cameras: Dict[str, CameraStream] = {}
        self.recognition_results: Dict[str, List] = {}
        
        # Load face database
        self._load_face_database()
        print("✅ System ready!")
    
    def _load_face_database(self):
        """Load known faces from dataset"""
        try:
            # Load embeddings file
            npz_data = np.load("dataset/embeddings/all_embeddings.npz")
            
            # Load names mapping
            self.name_map = {}
            with open("dataset/embeddings/embeddings.csv", newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.name_map[row["Embedding Key"]] = row["Name"]
            
            # Prepare known faces arrays
            self.known_names = []
            self.known_embeddings = []
            
            for key in npz_data.files:
                if key in self.name_map:
                    self.known_names.append(self.name_map[key])
                    embedding = npz_data[key]
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize
                    self.known_embeddings.append(embedding)
            
            self.known_embeddings = np.array(self.known_embeddings)
            print(f"📚 Loaded {len(self.known_names)} known faces")
            
        except Exception as e:
            print(f"⚠️ Could not load face database: {e}")
            self.known_names = []
            self.known_embeddings = np.array([])
    
    def load_camera_config(self) -> Dict[int, str]:
        """Load camera names from config file"""
        config_file = "backend/camera_config.ini" #File Path
        camera_mapping = {}
        
        if os.path.exists(config_file):
            try:
                config = configparser.ConfigParser()
                config.read(config_file)
                
                if 'Display Names' in config:
                    for camera_id, name in config['Display Names'].items():
                        camera_mapping[int(camera_id)] = name
                        
            except Exception as e:
                print(f"⚠️ Config error: {e}")
        
        return camera_mapping
    
    def add_camera(self, camera_id: int, camera_name: str = None) -> bool:

        for cam in self.cameras.values():
            if cam.camera_id == camera_id:
                print(f"ℹ️ Camera with ID {camera_id} is already active. Skipping.")
                return True

        """Add a camera to the system"""
        # Use config name if not provided
        if camera_name is None:
            config = self.load_camera_config()
            camera_name = config.get(camera_id, f"Camera {camera_id}")
        
        # Check for duplicates
        if camera_name in self.cameras:
            print(f"⚠️ {camera_name} already exists")
            return False
        
        # Start camera
        camera = CameraStream(camera_id, camera_name)
        if camera.start():
            self.cameras[camera_name] = camera
            self.recognition_results[camera_name] = []
            return True
        return False
    
    def start_default_cameras(self) -> int:
        """Start cameras 0 and 1 automatically"""
        started = 0
        for camera_id in [0]:
            if self.add_camera(camera_id):
                started += 1
        return started
    
    def remove_camera(self, camera_name: str):
        """Remove a camera"""
        if camera_name in self.cameras:
            self.cameras[camera_name].stop()
            del self.cameras[camera_name]
            del self.recognition_results[camera_name]
    
    def process_frame_recognition(self, frame: np.ndarray, camera_name: str) -> List[Dict]:
        """Detect and recognize faces in a frame"""
        results = []
        
        try:
            # Convert to PIL for face detection
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Detect faces
            boxes, probs = self.mtcnn.detect(pil_img)
            faces = self.mtcnn(pil_img)
            
            if boxes is not None and faces is not None:
                # Handle single/multiple faces
                if isinstance(faces, torch.Tensor) and faces.ndim == 3:
                    faces = [faces]
                elif isinstance(faces, torch.Tensor) and faces.ndim == 4:
                    faces = [f for f in faces]
                
                # Process each detected face
                for box, face, prob in zip(boxes, faces, probs):
                    if prob < 0.85:  # Skip low confidence detections
                        continue
                    
                    # Get face coordinates
                    x1, y1, x2, y2 = map(int, box)
                    if (x2 - x1 < 60) or (y2 - y1 < 60):  # Skip small faces
                        continue
                    
                    # Get face embedding
                    face_input = face.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.model(face_input).squeeze().cpu().numpy()
                    
                    # Match against known faces
                    name, confidence = self._match_face(embedding)
                    
                    # Store result
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'name': name,
                        'confidence': confidence,
                        'camera': camera_name,
                        'timestamp': time.time()
                    })
        
        except Exception as e:
            print(f"⚠️ Recognition error in {camera_name}: {e}")
        
        return results
    
    def _match_face(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Match face embedding against known faces"""
        if len(self.known_embeddings) == 0:
            return "Unknown", 0.0
        
        # Normalize and compare
        embedding = embedding / np.linalg.norm(embedding)
        similarities = cosine_similarity([embedding], self.known_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Return name if confidence is high enough
        if best_score >= 0.75:
            return self.known_names[best_idx], best_score
        else:
            return "Unknown", best_score
    
    def get_all_camera_names(self) -> List[str]:
        """Get list of active cameras"""
        return list(self.cameras.keys())
    
    def find_person_location(self, person_name: str, time_window: int = 5) -> List[str]:
        """Find which cameras see a specific person"""
        current_time = time.time()
        found_cameras = []
        
        for camera_name, results in self.recognition_results.items():
            for result in results:
                if (current_time - result['timestamp'] <= time_window and 
                    result['name'].lower() == person_name.lower() and
                    result['confidence'] >= 0.70):
                    if camera_name not in found_cameras:
                        found_cameras.append(camera_name)
                    break
        
        return found_cameras
    
    def update_recognition_results(self, camera_name: str, results: List[Dict]):
        """Store recognition results (keep only recent ones)"""
        if camera_name in self.recognition_results:
            self.recognition_results[camera_name].extend(results)
            # Keep only last 50 results to save memory
            if len(self.recognition_results[camera_name]) > 50:
                self.recognition_results[camera_name] = self.recognition_results[camera_name][-50:]
    
    def get_camera_status(self) -> Dict[str, Dict]:
        """Get status of all cameras"""
        status = {}
        for name, camera in self.cameras.items():
            status[name] = {
                'active': camera.is_running,
                'camera_id': camera.camera_id,
                'recent_detections': len([r for r in self.recognition_results[name] 
                                        if time.time() - r['timestamp'] <= 10])
            }
        return status
    
    def cleanup(self):
        """Stop all cameras and cleanup"""
        for camera in self.cameras.values():
            camera.stop()
        self.cameras.clear()
        self.recognition_results.clear()
        print("✅ Cleanup complete")


