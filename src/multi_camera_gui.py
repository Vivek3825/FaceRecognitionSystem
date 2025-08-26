# src/multi_camera_gui.py
# NEW FILE: GUI interface for multi-camera face recognition
# Purpose: Provide user interface for camera management and person location
# Author: Added on August 25, 2025

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import time
from PIL import Image, ImageTk
import numpy as np
from typing import Dict, Optional
from multi_camera_manager import MultiCameraManager

# NEW: Import display configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    import camera_display_config as config
    print("✅ Using custom display configuration")
except ImportError:
    print("⚠️ Using default display configuration")
    # Default fallback values
    class config:
        VIDEO_WIDTH = 480
        VIDEO_HEIGHT = 360
        LABEL_WIDTH = 60
        LABEL_HEIGHT = 25
        WINDOW_WIDTH = 1400
        WINDOW_HEIGHT = 900

class MultiCameraGUI:
    """
    NEW CLASS: GUI application for multi-camera face recognition system
    Provides real-time video display and person location functionality
    """
    
    def __init__(self):
        """Initialize the GUI application"""
        print("🖥️ Initializing Multi-Camera GUI...")
        
        # UPDATED: Configuration from config file - Easy to adjust
        self.VIDEO_WIDTH = config.VIDEO_WIDTH
        self.VIDEO_HEIGHT = config.VIDEO_HEIGHT
        self.LABEL_WIDTH = config.LABEL_WIDTH
        self.LABEL_HEIGHT = config.LABEL_HEIGHT
        
        print(f"📐 Display Settings: {self.VIDEO_WIDTH}x{self.VIDEO_HEIGHT} video, {self.LABEL_WIDTH}x{self.LABEL_HEIGHT} labels")
        
        # Initialize main window - UPDATED: Using configurable window size
        self.root = tk.Tk()
        self.root.title("Multi-Camera Face Recognition System")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize camera manager
        self.camera_manager = MultiCameraManager()
        
        # GUI state variables
        self.is_running = False
        self.video_frames = {}  # Store video frame references
        self.update_thread = None
        
        # Create GUI layout
        self._create_gui_layout()
        
        # Initialize with default cameras
        self._setup_default_cameras()
        
        print("✅ Multi-Camera GUI initialized successfully!")
    
    def _create_gui_layout(self):
        """
        PRIVATE METHOD: Create the main GUI layout
        Organized into sections: controls, video feeds, and status
        """
        # Main container with padding
        main_frame = tk.Frame(self.root, bg='#2b2b2b', padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === TOP SECTION: Controls ===
        self._create_control_panel(main_frame)
        
        # === MIDDLE SECTION: Video Feeds ===
        self._create_video_section(main_frame)
        
        # === BOTTOM SECTION: Status and Person Search ===
        self._create_status_section(main_frame)
    
    def _create_control_panel(self, parent):
        """
        PRIVATE METHOD: Create control panel with start/stop and camera management
        """
        control_frame = tk.LabelFrame(parent, text="Camera Controls", 
                                    bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Control buttons frame
        button_frame = tk.Frame(control_frame, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Start/Stop buttons
        self.start_button = tk.Button(button_frame, text="▶ Start Recognition", 
                                    command=self._start_recognition,
                                    bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                    width=15, height=2)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = tk.Button(button_frame, text="⏹ Stop Recognition", 
                                   command=self._stop_recognition,
                                   bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                   width=15, height=2, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Camera management
        camera_mgmt_frame = tk.Frame(button_frame, bg='#2b2b2b')
        camera_mgmt_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        tk.Label(camera_mgmt_frame, text="Add Camera:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        
        self.camera_id_var = tk.StringVar(value="2")
        tk.Entry(camera_mgmt_frame, textvariable=self.camera_id_var, width=5).pack(side=tk.LEFT, padx=(5, 5))
        
        self.camera_name_var = tk.StringVar(value="External Camera")
        tk.Entry(camera_mgmt_frame, textvariable=self.camera_name_var, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(camera_mgmt_frame, text="Add", command=self._add_camera,
                bg='#2196F3', fg='white', width=8).pack(side=tk.LEFT)
    
    def _create_video_section(self, parent):
        """
        PRIVATE METHOD: Create video display section for multiple camera feeds
        """
        video_frame = tk.LabelFrame(parent, text="Camera Feeds", 
                                  bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create scrollable frame for video feeds
        canvas = tk.Canvas(video_frame, bg='#2b2b2b')
        scrollbar = ttk.Scrollbar(video_frame, orient="vertical", command=canvas.yview)
        self.video_container = tk.Frame(canvas, bg='#2b2b2b')
        
        self.video_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.video_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_status_section(self, parent):
        """
        PRIVATE METHOD: Create status display and person search functionality
        """
        status_frame = tk.LabelFrame(parent, text="System Status & Person Search", 
                                   bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        status_frame.pack(fill=tk.X)
        
        # Create two columns: status and search
        columns_frame = tk.Frame(status_frame, bg='#2b2b2b')
        columns_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Left column: Status
        status_left = tk.Frame(columns_frame, bg='#2b2b2b')
        status_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(status_left, text="Camera Status:", bg='#2b2b2b', fg='white', 
               font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.status_text = tk.Text(status_left, height=4, width=40, bg='#404040', fg='white',
                                 font=('Consolas', 9))
        self.status_text.pack(fill=tk.X, pady=(5, 0))
        
        # Right column: Person Search
        search_right = tk.Frame(columns_frame, bg='#2b2b2b')
        search_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        tk.Label(search_right, text="🔍 Find Person Location:", bg='#2b2b2b', fg='white',
               font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        search_frame = tk.Frame(search_right, bg='#2b2b2b')
        search_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, 
                              font=('Arial', 10), width=20)
        search_entry.pack(side=tk.LEFT, pady=(0, 5))
        search_entry.bind('<Return>', lambda e: self._search_person())
        
        tk.Button(search_frame, text="Search", command=self._search_person,
                bg='#FF9800', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(5, 0))
        
        # Search results
        self.search_result_label = tk.Label(search_right, text="Enter a name and press Search", 
                                          bg='#2b2b2b', fg='#cccccc', font=('Arial', 9),
                                          wraplength=300, justify=tk.LEFT)
        self.search_result_label.pack(anchor=tk.W, pady=(5, 0))
    
    def _setup_default_cameras(self):
        """
        PRIVATE METHOD: Setup default cameras using configuration mapping
        """
        print("🔍 Setting up default cameras from configuration...")
        
        # Use the camera manager's automatic default camera setup
        cameras_added = self.camera_manager.start_default_cameras()
        
        if cameras_added > 0:
            print(f"🎯 Successfully initialized {cameras_added} camera(s)")
            self._update_video_display_layout()
        else:
            print("⚠️ No default cameras could be started!")
            print("   You can add cameras manually using the interface.")
            # Fall back to automatic detection
            self._fallback_camera_detection()
            
    def _detect_and_identify_cameras(self) -> list:
        """
        PRIVATE METHOD: Detect and intelligently identify camera types
        
        Returns:
            List of dictionaries with camera information
        """
        detected_cameras = []
        print("🔎 Scanning camera IDs 0-4...")
        
        for camera_id in range(5):  # Check cameras 0-4
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                # Test if camera actually works
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties for identification
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    # Try to get camera backend info
                    backend = cap.getBackendName()
                    
                    # Smart camera type detection
                    camera_type, camera_name = self._identify_camera_type(camera_id, width, height, backend)
                    
                    camera_info = {
                        'id': camera_id,
                        'name': camera_name,
                        'type': camera_type,
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'backend': backend
                    }
                    
                    detected_cameras.append(camera_info)
                    print(f"� Found Camera ID {camera_id}: {camera_name} ({width}x{height} @ {fps}fps)")
                
                cap.release()
        
        # Sort cameras: Built-in first, then external
        detected_cameras.sort(key=lambda x: (x['type'] != 'Built-in', x['id']))
        
        return detected_cameras
    
    def _identify_camera_type(self, camera_id: int, width: int, height: int, backend: str) -> tuple:
        """
        PRIVATE METHOD: Intelligently identify camera type based on properties
        
        Args:
            camera_id: Camera index
            width: Camera width
            height: Camera height
            backend: Camera backend name
            
        Returns:
            Tuple of (camera_type, camera_name)
        """
        # Common built-in camera resolutions
        builtin_resolutions = [
            (640, 480), (1280, 720), (1920, 1080),  # Common laptop camera resolutions
            (320, 240), (160, 120)  # Lower quality built-in cameras
        ]
        
        # High-quality external camera resolutions
        external_resolutions = [
            (1920, 1080), (2560, 1440), (3840, 2160),  # High-end external cameras
            (1280, 960), (800, 600)  # External webcams
        ]
        
        # Detection logic
        if camera_id == 0:
            # ID 0 is traditionally built-in on most systems
            return "Built-in", "Laptop Camera"
        
        elif (width, height) in builtin_resolutions and camera_id <= 1:
            # Low-medium resolution and low ID suggests built-in
            if camera_id == 1:
                return "Built-in", "Secondary Camera"
            else:
                return "Built-in", "Laptop Camera"
        
        elif (width, height) in external_resolutions or camera_id >= 2:
            # High resolution or high ID suggests external
            if "phone" in backend.lower() or camera_id == 1:
                return "External", "Phone Camera"
            else:
                return "External", f"External Camera {camera_id}"
        
        else:
            # Default classification
            if camera_id <= 1:
                camera_name = "Phone Camera" if camera_id == 1 else "Laptop Camera"
                camera_type = "External" if camera_id == 1 else "Built-in"
            else:
                camera_name = f"External Camera {camera_id}"
                camera_type = "External"
            
            return camera_type, camera_name
    
    def _add_camera(self):
        """
        EVENT HANDLER: Add new camera based on user input
        """
        try:
            camera_id = int(self.camera_id_var.get())
            camera_name = self.camera_name_var.get().strip()
            
            if not camera_name:
                messagebox.showerror("Error", "Please enter a camera name")
                return
            
            if self.camera_manager.add_camera(camera_id, camera_name):
                self._update_video_display_layout()
                messagebox.showinfo("Success", f"Camera '{camera_name}' added successfully!")
                # Reset input fields
                self.camera_id_var.set(str(camera_id + 1))
                self.camera_name_var.set("External Camera")
            else:
                messagebox.showerror("Error", f"Failed to add camera '{camera_name}'")
                
        except ValueError:
            messagebox.showerror("Error", "Camera ID must be a number")
    
    def _update_video_display_layout(self):
        """
        PRIVATE METHOD: Update video display layout based on active cameras
        """
        # Clear existing video frames
        for widget in self.video_container.winfo_children():
            widget.destroy()
        self.video_frames.clear()
        
        # Create frames for each camera
        cameras = self.camera_manager.get_all_camera_names()
        
        for i, camera_name in enumerate(cameras):
            # Create frame for this camera
            camera_frame = tk.LabelFrame(self.video_container, text=camera_name,
                                       bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
            camera_frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            
            # Video display label - UPDATED: Using configurable sizes
            video_label = tk.Label(camera_frame, text="📷 Camera Starting...", 
                                 bg='#404040', fg='white', width=self.LABEL_WIDTH, height=self.LABEL_HEIGHT)
            video_label.pack(padx=10, pady=10)
            
            # Info label for detections
            info_label = tk.Label(camera_frame, text="Ready", bg='#2b2b2b', fg='#cccccc')
            info_label.pack()
            
            self.video_frames[camera_name] = {
                'video_label': video_label,
                'info_label': info_label
            }
    
    def _start_recognition(self):
        """
        EVENT HANDLER: Start the recognition system
        """
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start update thread
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            print("▶️ Recognition system started")
    
    def _stop_recognition(self):
        """
        EVENT HANDLER: Stop the recognition system
        """
        if self.is_running:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            print("⏹️ Recognition system stopped")
    
    def _update_loop(self):
        """
        PRIVATE METHOD: Main update loop running in separate thread
        Continuously processes camera frames and updates GUI
        """
        while self.is_running:
            try:
                # Process each camera
                for camera_name, camera in self.camera_manager.cameras.items():
                    if not camera.is_running:
                        continue
                    
                    # Get latest frame
                    frame = camera.get_frame()
                    if frame is None:
                        continue
                    
                    # Process for face recognition
                    recognition_results = self.camera_manager.process_frame_recognition(frame, camera_name)
                    
                    # Update results in manager
                    self.camera_manager.update_recognition_results(camera_name, recognition_results)
                    
                    # Draw recognition results on frame
                    display_frame = self._draw_recognition_results(frame, recognition_results)
                    
                    # Update GUI (must be done in main thread)
                    self.root.after(0, self._update_camera_display, camera_name, display_frame, recognition_results)
                
                # Update status
                self.root.after(0, self._update_status_display)
                
                # Control update rate
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"⚠️ Error in update loop: {e}")
                time.sleep(0.1)
    
    def _draw_recognition_results(self, frame: np.ndarray, results: list) -> np.ndarray:
        """
        PRIVATE METHOD: Draw bounding boxes and labels on frame
        
        Args:
            frame: Original camera frame
            results: List of recognition results
            
        Returns:
            Frame with recognition results drawn
        """
        display_frame = frame.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            confidence = result['confidence']
            confidence_level = result['confidence_level']
            
            # Choose color based on confidence level
            if confidence_level == "high_confidence":
                color = (0, 255, 0)  # Green
            elif confidence_level == "medium_confidence":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def _update_camera_display(self, camera_name: str, frame: np.ndarray, results: list):
        """
        PRIVATE METHOD: Update camera display in GUI (runs in main thread)
        """
        if camera_name not in self.video_frames:
            return
        
        try:
            # Resize frame for display - UPDATED: Using configurable size
            display_frame = cv2.resize(frame, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
            
            # Convert to PhotoImage
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update video label
            video_label = self.video_frames[camera_name]['video_label']
            video_label.configure(image=photo)
            video_label.image = photo  # Keep reference
            
            # Update info label
            info_text = f"Faces: {len(results)}"
            if results:
                detected_names = [r['name'] for r in results if r['confidence_level'] != 'low_confidence']
                if detected_names:
                    info_text += f" | Recognized: {', '.join(detected_names)}"
            
            self.video_frames[camera_name]['info_label'].configure(text=info_text)
            
        except Exception as e:
            print(f"⚠️ Error updating camera display for {camera_name}: {e}")
    
    def _update_status_display(self):
        """
        PRIVATE METHOD: Update system status display with clear camera info
        """
        try:
            status_info = self.camera_manager.get_camera_status()
            
            self.status_text.delete(1.0, tk.END)
            
            if not status_info:
                self.status_text.insert(tk.END, "No cameras added yet.\nUse 'Add Camera' button to add cameras.")
                return
            
            for camera_name, status in status_info.items():
                # Clean status line with proper camera identification
                status_line = f"{camera_name} (ID:{status['camera_id']}): "
                if status['active']:
                    status_line += f"✅ Active | {status['fps']}fps | Detections: {status['recent_detections']}\n"
                else:
                    status_line += "❌ Inactive\n"
                
                self.status_text.insert(tk.END, status_line)
                
        except Exception as e:
            print(f"⚠️ Error updating status: {e}")
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, f"Status update error: {e}")
    
    def _search_person(self):
        """
        EVENT HANDLER: Search for person across all cameras
        """
        person_name = self.search_var.get().strip()
        
        if not person_name:
            self.search_result_label.configure(text="Please enter a person's name", fg='#ff9999')
            return
        
        # Find person in cameras
        found_cameras = self.camera_manager.find_person_location(person_name)
        
        if found_cameras:
            result_text = f"🎯 {person_name} found in: {', '.join(found_cameras)}"
            self.search_result_label.configure(text=result_text, fg='#99ff99')
        else:
            result_text = f"❌ {person_name} not found in any camera (last 5 seconds)"
            self.search_result_label.configure(text=result_text, fg='#ff9999')
        
        print(f"🔍 Search result: {result_text}")
    
    def run(self):
        """
        Main method to start the GUI application
        """
        try:
            print("🚀 Starting Multi-Camera GUI...")
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"❌ Error running GUI: {e}")
        finally:
            self._cleanup()
    
    def _on_closing(self):
        """
        EVENT HANDLER: Handle window closing
        """
        print("🔄 Shutting down application...")
        self.is_running = False
        self.root.destroy()
    
    def _cleanup(self):
        """
        PRIVATE METHOD: Cleanup resources
        """
        print("🧹 Cleaning up GUI resources...")
        if self.camera_manager:
            self.camera_manager.cleanup()
        print("✅ GUI cleanup completed")
    
    def _fallback_camera_detection(self):
        """
        PRIVATE METHOD: Fallback camera detection when configuration fails
        """
        print("🔄 Attempting fallback camera detection...")
        
        # Try basic camera IDs 0 and 1
        fallback_cameras = [
            (0, "Camera 0"),
            (1, "Camera 1")
        ]
        
        cameras_added = 0
        for camera_id, camera_name in fallback_cameras:
            if self.camera_manager.add_camera(camera_id, camera_name):
                cameras_added += 1
        
        if cameras_added > 0:
            print(f"🎯 Fallback detection found {cameras_added} camera(s)")
            self._update_video_display_layout()
        else:
            print("❌ No cameras detected even in fallback mode")
            print("   Please check camera connections and try adding manually")


# NEW MAIN EXECUTION BLOCK
if __name__ == "__main__":
    """
    Main entry point for the GUI application
    """
    try:
        print("=" * 60)
        print("🎯 Multi-Camera Face Recognition System")
        print("📅 Version 3.0 - Added August 25, 2025")
        print("=" * 60)
        
        app = MultiCameraGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Application interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        print("👋 Application terminated")
