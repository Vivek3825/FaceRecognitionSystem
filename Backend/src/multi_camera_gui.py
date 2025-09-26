# Multi-Camera GUI - Clean Version
# Simple and easy to understand interface
# Author: Cleaned up on August 26, 2025

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from multi_camera_manager import MultiCameraManager


class MultiCameraGUI:
    """Simple GUI for multi-camera face recognition"""
    
    def __init__(self):
        # Main window setup
        self.root = tk.Tk()
        self.root.title("Multi-Camera Face Recognition")
        self.root.geometry("1200x800")
        
        # Camera system
        self.camera_manager = MultiCameraManager()
        
        # GUI variables
        self.video_labels = {}  # Store video display labels
        self.is_running = False
        self.update_thread = None
        
        # Setup interface
        self._create_interface()
        self._start_cameras()
        
        print("✅ GUI Ready!")
    
    def _create_interface(self):
        """Create the main interface"""
        # Top control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Title
        title_label = ttk.Label(control_frame, text="Multi-Camera Face Recognition", 
                               font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Control buttons
        ttk.Button(control_frame, text="Start", command=self._start_recognition).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Stop", command=self._stop_recognition).pack(side=tk.RIGHT, padx=5)
        
        # Person search section
        search_frame = ttk.LabelFrame(self.root, text="Find Person")
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Search input
        search_input_frame = ttk.Frame(search_frame)
        search_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_input_frame, text="Person Name:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        ttk.Entry(search_input_frame, textvariable=self.search_var, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_input_frame, text="Search", command=self._search_person).pack(side=tk.LEFT, padx=5)
        
        # Search results
        self.search_result_label = ttk.Label(search_frame, text="Enter a name to search", foreground="gray")
        self.search_result_label.pack(padx=5, pady=5)
        
        # Camera management section
        camera_frame = ttk.LabelFrame(self.root, text="Camera Management")
        camera_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add camera controls
        add_frame = ttk.Frame(camera_frame)
        add_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(add_frame, text="Camera ID:").pack(side=tk.LEFT)
        self.camera_id_var = tk.StringVar(value="2")
        ttk.Entry(add_frame, textvariable=self.camera_id_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(add_frame, text="Name:").pack(side=tk.LEFT, padx=(10,0))
        self.camera_name_var = tk.StringVar(value="External Camera")
        ttk.Entry(add_frame, textvariable=self.camera_name_var, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(add_frame, text="Add Camera", command=self._add_camera).pack(side=tk.LEFT, padx=5)
        
        # Video display area
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Status bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _start_cameras(self):
        """Start default cameras"""
        cameras_started = self.camera_manager.start_default_cameras()
        if cameras_started > 0:
            self._update_video_layout()
            self.status_label.config(text=f"Started {cameras_started} camera(s)")
        else:
            self.status_label.config(text="No cameras detected")
    
    def _update_video_layout(self):
        """Create video display widgets for each camera"""
        # Clear existing video displays
        for widget in self.video_frame.winfo_children():
            widget.destroy()
        self.video_labels.clear()
        
        # Get active cameras
        camera_names = self.camera_manager.get_all_camera_names()
        
        if not camera_names:
            no_cam_label = ttk.Label(self.video_frame, text="No cameras active", 
                                   font=("Arial", 14), foreground="gray")
            no_cam_label.pack(expand=True)
            return
        
        # Create video displays in a grid
        cols = 2 if len(camera_names) > 1 else 1
        for i, camera_name in enumerate(camera_names):
            # Create frame for this camera
            row = i // cols
            column = i % cols
            
            camera_frame = ttk.LabelFrame(self.video_frame, text=camera_name)
            camera_frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
            
            # Video display label
            video_label = ttk.Label(camera_frame, text="Loading...")
            video_label.pack(padx=5, pady=5)
            
            # Info label for detections
            info_label = ttk.Label(camera_frame, text="No detections", foreground="blue")
            info_label.pack(pady=(0,5))
            
            # Store references
            self.video_labels[camera_name] = {
                'video': video_label,
                'info': info_label
            }
        
        # Configure grid weights
        for i in range(cols):
            self.video_frame.columnconfigure(i, weight=1)
    
    def _start_recognition(self):
        """Start face recognition"""
        if not self.is_running:
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            self.status_label.config(text="Recognition running...")
    
    def _stop_recognition(self):
        """Stop face recognition"""
        self.is_running = False
        self.status_label.config(text="Recognition stopped")
    
    def _update_loop(self):
        """Main update loop for video and recognition"""
        while self.is_running:
            try:
                # Update each camera
                for camera_name, camera in self.camera_manager.cameras.items():
                    if camera_name in self.video_labels:
                        # Get latest frame
                        frame = camera.get_frame()
                        if frame is not None:
                            # Process face recognition
                            results = self.camera_manager.process_frame_recognition(frame, camera_name)
                            self.camera_manager.update_recognition_results(camera_name, results)
                            
                            # Draw detection boxes
                            display_frame = frame.copy()
                            detection_text = "No faces detected"
                            
                            if results:
                                detection_text = f"{len(results)} face(s): "
                                names = []
                                
                                for result in results:
                                    # Draw bounding box
                                    x1, y1, x2, y2 = result['bbox']
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Draw name and confidence
                                    name = result['name']
                                    conf = result['confidence']
                                    label = f"{name} ({conf:.2f})"
                                    cv2.putText(display_frame, label, (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    
                                    names.append(name)
                                
                                detection_text += ", ".join(set(names))
                            
                            # Update video display
                            self._update_video_display(camera_name, display_frame, detection_text)
                
                time.sleep(0.1)  # Limit update rate
                
            except Exception as e:
                print(f"Update error: {e}")
                time.sleep(1)
    
    def _update_video_display(self, camera_name: str, frame: any, info_text: str):
        """Update video display for a camera"""
        try:
            if camera_name not in self.video_labels:
                return
            
            # Resize frame for display
            height, width = frame.shape[:2]
            display_width = 320
            display_height = int(display_width * height / width)
            frame_resized = cv2.resize(frame, (display_width, display_height))
            
            # Convert to PIL and then to PhotoImage
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update displays (must be on main thread)
            self.root.after(0, lambda: self._safe_update_display(camera_name, photo, info_text))
            
        except Exception as e:
            print(f"Display error for {camera_name}: {e}")
    
    def _safe_update_display(self, camera_name: str, photo: ImageTk.PhotoImage, info_text: str):
        """Safely update display on main thread"""
        try:
            if camera_name in self.video_labels:
                # Update video
                self.video_labels[camera_name]['video'].config(image=photo)
                self.video_labels[camera_name]['video'].image = photo  # Keep reference
                
                # Update info
                self.video_labels[camera_name]['info'].config(text=info_text)
        except:
            pass  # Widget may have been destroyed
    
    def _search_person(self):
        """Search for a person across all cameras"""
        name = self.search_var.get().strip()
        if not name:
            self.search_result_label.config(text="Please enter a name", foreground="red")
            return
        
        # Find person in recent detections
        found_cameras = self.camera_manager.find_person_location(name)
        
        if found_cameras:
            result_text = f"'{name}' found in: {', '.join(found_cameras)}"
            self.search_result_label.config(text=result_text, foreground="green")
        else:
            result_text = f"'{name}' not found in any camera"
            self.search_result_label.config(text=result_text, foreground="orange")
    
    def _add_camera(self):
        """Add a new camera manually"""
        try:
            camera_id = int(self.camera_id_var.get())
            camera_name = self.camera_name_var.get().strip()
            
            if not camera_name:
                messagebox.showerror("Error", "Please enter a camera name")
                return
            
            if self.camera_manager.add_camera(camera_id, camera_name):
                self._update_video_layout()
                self.status_label.config(text=f"Added {camera_name}")
                messagebox.showinfo("Success", f"Camera '{camera_name}' added successfully!")
            else:
                messagebox.showerror("Error", f"Failed to add camera '{camera_name}'")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid camera ID number")
    
    def run(self):
        """Start the GUI"""
        try:
            # Start recognition automatically
            self._start_recognition()
            
            # Run GUI
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            # Cleanup
            self.is_running = False
            self.camera_manager.cleanup()


if __name__ == "__main__":
    print("🚀 Starting Multi-Camera Face Recognition GUI...")
    app = MultiCameraGUI()
    app.run()
