# src/multi_camera_manager.py
# NEW FILE: Multi-camera parallel processing manager
# Purpose: Handle multiple camera streams with parallel processing
# Author: Added on August 25, 2025

import cv2
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import csv
import configparser
import os

class CameraStream:
    """
    NEW CLASS: Individual camera stream handler
    Manages single camera capture and processing
    """
    
    def __init__(self, camera_id: int, camera_name: str):
        """
        Initialize camera stream
        
        Args:
            camera_id: Camera index (0 for laptop, 1 for phone, etc.)
            camera_name: Human readable name (e.g., "Laptop Camera", "Phone Camera")
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory issues
        self.is_running = False
        self.thread = None
        self.last_frame = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def start(self) -> bool:
        """
        Start camera capture thread with improved error reporting
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            print(f"🔄 Attempting to start '{self.camera_name}' (ID: {self.camera_id})...")
            
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera '{self.camera_name}' (ID: {self.camera_id}) - Camera not available")
                return False
            
            # Test if camera actually works by trying to read a frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print(f"❌ Camera '{self.camera_name}' (ID: {self.camera_id}) opened but cannot read frames")
                self.cap.release()
                return False
                
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            print(f"✅ Camera '{self.camera_name}' started successfully ({actual_width}x{actual_height} @ {actual_fps}fps)")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera '{self.camera_name}' (ID: {self.camera_id}): {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def _capture_loop(self):
        """
        PRIVATE METHOD: Main capture loop running in separate thread
        Continuously captures frames and puts them in queue
        """
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Update FPS calculation
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Update FPS every 30 frames
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                
                # Store latest frame
                self.last_frame = frame.copy()
                
                # Put frame in queue (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get(block=False)
                        self.frame_queue.put(frame, block=False)
                    except queue.Empty:
                        pass
            else:
                print(f"⚠️ Failed to read frame from {self.camera_name}")
                time.sleep(0.1)  # Small delay before retry
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame from camera
        
        Returns:
            np.ndarray or None: Latest frame if available
        """
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return self.last_frame  # Return last known frame
    
    def stop(self):
        """Stop camera capture and cleanup resources"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print(f"🔌 Camera {self.camera_name} stopped")


class MultiCameraManager:
    """
    NEW CLASS: Main manager for multiple camera streams
    Handles parallel processing and face recognition across all cameras
    """
    
    def __init__(self):
        """Initialize multi-camera manager with face recognition models"""
        print("🔄 Initializing Multi-Camera Face Recognition System...")
        
        # Initialize face recognition models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Using device: {self.device}")
        
        self.mtcnn = MTCNN(
            image_size=160,
            post_process=True,
            keep_all=True,
            device=self.device
        )
        
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Camera management
        self.cameras: Dict[str, CameraStream] = {}
        self.recognition_results: Dict[str, List] = {}  # Store results per camera
        
        # Load known faces database
        self._load_face_database()
        
        # Load camera mapping
        self.camera_mapping = self.load_camera_mapping()
        self.default_cameras = self.get_default_cameras()
        
        print("✅ Multi-Camera Manager initialized successfully!")
    
    def _load_face_database(self):
        """
        PRIVATE METHOD: Load face embeddings and metadata from database
        Same as original system but with better error handling
        """
        try:
            # Load embeddings
            embeddings_path = "dataset/embeddings/all_embeddings.npz"
            csv_path = "dataset/embeddings/embeddings.csv"
            
            npz_data = np.load(embeddings_path)
            
            # Load name mapping
            self.name_map = {}
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.name_map[row["Embedding Key"]] = row["Name"]
            
            # Prepare embeddings array
            self.known_names = []
            self.known_embeddings = []
            
            for key in npz_data.files:
                if key in self.name_map:
                    self.known_names.append(self.name_map[key])
                    embedding = npz_data[key]
                    embedding = embedding / np.linalg.norm(embedding)
                    self.known_embeddings.append(embedding)
            
            self.known_embeddings = np.array(self.known_embeddings)
            
            print(f"📚 Loaded {len(self.known_names)} known faces from database")
            
        except Exception as e:
            print(f"❌ Error loading face database: {e}")
            self.known_names = []
            self.known_embeddings = np.array([])
    
    def add_camera(self, camera_id: int, camera_name: str = None) -> bool:
        """
        Add a new camera to the system with duplicate prevention and auto-naming
        
        Args:
            camera_id: Camera index (0, 1, 2, etc.)
            camera_name: Human readable name (optional - will use mapping if available)
            
        Returns:
            bool: True if camera added successfully
        """
        # Load camera mapping and use it if no name provided
        if camera_name is None:
            camera_mapping = self.load_camera_mapping()
            camera_name = camera_mapping.get(camera_id, f"Camera {camera_id}")
        
        # Check for duplicate camera names
        if camera_name in self.cameras:
            print(f"⚠️ Camera '{camera_name}' already exists")
            return False
        
        # Check for duplicate camera IDs
        for existing_camera in self.cameras.values():
            if existing_camera.camera_id == camera_id:
                print(f"⚠️ Camera ID {camera_id} already in use by '{existing_camera.camera_name}'")
                return False
        
        # Try to create and start the camera
        camera_stream = CameraStream(camera_id, camera_name)
        if camera_stream.start():
            self.cameras[camera_name] = camera_stream
            self.recognition_results[camera_name] = []
            print(f"📹 Added camera: '{camera_name}' (ID: {camera_id})")
            return True
        else:
            print(f"❌ Failed to initialize camera '{camera_name}' (ID: {camera_id})")
            return False
    
    def remove_camera(self, camera_name: str):
        """Remove camera from system"""
        if camera_name in self.cameras:
            self.cameras[camera_name].stop()
            del self.cameras[camera_name]
            del self.recognition_results[camera_name]
            print(f"🗑️ Removed camera: {camera_name}")
    
    def start_default_cameras(self) -> int:
        """
        Start default cameras as defined in the configuration file
        
        Returns:
            Number of cameras successfully started
        """
        default_cameras = self.get_default_cameras()
        started_count = 0
        
        print(f"🚀 Starting default cameras: {default_cameras}")
        
        for camera_id in default_cameras:
            if self.add_camera(camera_id):
                started_count += 1
            else:
                print(f"⚠️ Could not start default camera {camera_id}")
        
        print(f"✅ Started {started_count}/{len(default_cameras)} default cameras")
        return started_count
    
    def process_frame_recognition(self, frame: np.ndarray, camera_name: str) -> List[Dict]:
        """
        NEW METHOD: Process single frame for face recognition
        
        Args:
            frame: Camera frame
            camera_name: Name of camera
            
        Returns:
            List of detected faces with recognition results
        """
        results = []
        
        try:
            # Convert frame to PIL for MTCNN
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Detect faces
            boxes, probs = self.mtcnn.detect(pil_img)
            faces = self.mtcnn(pil_img)
            
            if boxes is not None and faces is not None:
                # Handle single vs multiple faces
                if isinstance(faces, torch.Tensor) and faces.ndim == 4:
                    faces = [f for f in faces]
                elif isinstance(faces, torch.Tensor) and faces.ndim == 3:
                    faces = [faces]
                else:
                    faces = []
                
                for box, face, prob in zip(boxes, faces, probs):
                    if prob is None or prob < 0.85:
                        continue
                    
                    # Skip small faces
                    x1, y1, x2, y2 = map(int, box)
                    if (x2 - x1 < 60) or (y2 - y1 < 60):
                        continue
                    
                    # Extract embedding
                    face_input = face.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.model(face_input).squeeze().cpu().numpy()
                    
                    # Match against known faces
                    name, confidence, confidence_level = self._match_face(embedding)
                    
                    # Store result
                    result = {
                        'bbox': [x1, y1, x2, y2],
                        'name': name,
                        'confidence': confidence,
                        'confidence_level': confidence_level,
                        'camera': camera_name,
                        'timestamp': time.time()
                    }
                    results.append(result)
        
        except Exception as e:
            print(f"⚠️ Error processing frame from {camera_name}: {e}")
        
        return results
    
    def _match_face(self, embedding: np.ndarray) -> Tuple[str, float, str]:
        """
        PRIVATE METHOD: Match face embedding against known database
        Enhanced version of original matching logic
        """
        if len(self.known_embeddings) == 0:
            return "Unknown", 0.0, "no_database"
        
        embedding = embedding / np.linalg.norm(embedding)
        similarities = cosine_similarity([embedding], self.known_embeddings)[0]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Calculate margin (difference between best and second best)
        sorted_similarities = np.sort(similarities)[::-1]
        margin = sorted_similarities[0] - sorted_similarities[1] if len(sorted_similarities) > 1 else 0.1
        
        # Enhanced confidence levels
        if best_score >= 0.5 and margin >= 0.05:
            return self.known_names[best_idx], best_score, "high_confidence"
        elif best_score >= 0.4 and margin >= 0.08:
            return self.known_names[best_idx], best_score, "medium_confidence"
        else:
            return "Unknown", best_score, "low_confidence"
    
    def get_all_camera_names(self) -> List[str]:
        """Get list of all active camera names"""
        return list(self.cameras.keys())
    
    def find_person_location(self, person_name: str, time_window: int = 5) -> List[str]:
        """
        NEW METHOD: Find which cameras currently see a specific person
        
        Args:
            person_name: Name of person to search for
            time_window: How many seconds back to check (default: 5)
            
        Returns:
            List of camera names where person was recently detected
        """
        current_time = time.time()
        found_cameras = []
        
        for camera_name, results in self.recognition_results.items():
            # Check recent results within time window
            for result in results:
                if (current_time - result['timestamp'] <= time_window and 
                    result['name'].lower() == person_name.lower() and
                    result['confidence_level'] in ['high_confidence', 'medium_confidence']):
                    if camera_name not in found_cameras:
                        found_cameras.append(camera_name)
                    break
        
        return found_cameras
    
    def update_recognition_results(self, camera_name: str, results: List[Dict]):
        """
        Update recognition results for a camera
        Keeps only recent results to prevent memory bloat
        """
        if camera_name in self.recognition_results:
            # Add new results
            self.recognition_results[camera_name].extend(results)
            
            # Keep only last 100 results per camera to prevent memory issues
            if len(self.recognition_results[camera_name]) > 100:
                self.recognition_results[camera_name] = self.recognition_results[camera_name][-100:]
    
    def get_camera_status(self) -> Dict[str, Dict]:
        """
        Get status information for all cameras with clear naming
        
        Returns:
            Dict with camera names as keys and status info as values
        """
        status = {}
        for name, camera in self.cameras.items():
            status[name] = {
                'active': camera.is_running,
                'fps': round(camera.fps, 1),
                'camera_id': camera.camera_id,
                'camera_name': camera.camera_name,  # Add explicit camera name
                'recent_detections': len([r for r in self.recognition_results[name] 
                                        if time.time() - r['timestamp'] <= 10])
            }
        return status
    
    def cleanup(self):
        """Cleanup all resources"""
        print("🧹 Cleaning up Multi-Camera Manager...")
        for camera in self.cameras.values():
            camera.stop()
        self.cameras.clear()
        self.recognition_results.clear()
        print("✅ Cleanup completed")
    
    def load_camera_mapping(self, config_file: str = "camera_mapping.ini") -> Dict[int, str]:
        """
        Load camera ID to name mapping from configuration file
        
        Args:
            config_file: Path to the camera mapping configuration file
        
        Returns:
            Dictionary mapping camera IDs to display names
        """
        camera_mapping = {}
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_file)
        
        if os.path.exists(config_path):
            try:
                config = configparser.ConfigParser()
                config.read(config_path)
                
                if 'Display Names' in config:
                    for camera_id, display_name in config['Display Names'].items():
                        try:
                            camera_mapping[int(camera_id)] = display_name
                        except ValueError:
                            print(f"⚠️ Invalid camera ID in config: {camera_id}")
                
                print(f"✅ Loaded camera mapping from {config_file}: {camera_mapping}")
                
            except Exception as e:
                print(f"⚠️ Error reading camera mapping file {config_file}: {e}")
                print("   Using default camera names")
        else:
            print(f"ℹ️ Camera mapping file {config_file} not found, using default names")
        
        return camera_mapping

    def get_default_cameras(self, config_file: str = "camera_mapping.ini") -> List[int]:
        """
        Get default camera IDs from configuration file
        
        Args:
            config_file: Path to the camera mapping configuration file
        
        Returns:
            List of default camera IDs to use
        """
        default_cameras = [0, 1]  # Fallback default
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_file)
        
        if os.path.exists(config_path):
            try:
                config = configparser.ConfigParser()
                config.read(config_path)
                
                if 'Default Configuration' in config and 'default_cameras' in config['Default Configuration']:
                    camera_str = config['Default Configuration']['default_cameras']
                    default_cameras = [int(x.strip()) for x in camera_str.split(',')]
                    print(f"✅ Using default cameras from config: {default_cameras}")
                
            except Exception as e:
                print(f"⚠️ Error reading default cameras from {config_file}: {e}")
                print(f"   Using fallback default: {default_cameras}")
        
        return default_cameras
