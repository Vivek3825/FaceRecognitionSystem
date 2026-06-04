#!/usr/bin/env python3
import sys
import os
import cv2
import base64
from PIL import Image
import io
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap

from backend.src.person_registration import PersonRegistrationSystem
from backend.src.multi_camera_manager import CameraStream
#from .camera_page import CameraMonitorPage

class WorkerThread(QThread):
    """Thread to run heavy ML backend operations without freezing the UI"""
    result_signal = Signal(dict)

    def __init__(self, backend_system, person_name, images_data):
        super().__init__()
        self.backend = backend_system
        self.person_name = person_name
        self.images_data = images_data

    def run(self):
        # Executes your 6-step transactional enrollment pipeline
        result = self.backend.register_person(self.person_name, self.images_data)
        self.result_signal.emit(result)


class RegistrationPage(QWidget):
    """Page for registering new persons connected directly to the ML backend"""
   
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backend = PersonRegistrationSystem()
        
        # Camera & capture state variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam_frame)
        self.current_frame = None
        
        # In-memory storage for base64 encoded strings required by the backend
        self.captured_images_b64 = {}
        self.angle_order = ['front', 'left', 'right']
        self.current_angle_index = 0
        
        self.init_ui()
        
        # Connect app quit signal for cleanup
        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self._safe_cleanup)
    
    def init_ui(self):
        """Initialize registration UI configured to match backend specs"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # --- LEFT SECTION: Backend Matching Information Form ---
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)

        page_title = QLabel("Person Registration")
        page_title.setFont(QFont("Arial", 22, QFont.Bold))
        page_title.setStyleSheet("color: #ffffff;")
        left_layout.addWidget(page_title)
        
        form_title = QLabel("Registration Metadata")
        form_title.setFont(QFont("Arial", 16, QFont.Bold))
        form_title.setStyleSheet("color: #e0e0e0;")
        left_layout.addWidget(form_title)
        
        form_layout = QGridLayout()
        form_layout.setSpacing(12)
        
        # Synchronized with info.csv headers
        name_label = QLabel("Full Name *")
        name_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        form_layout.addWidget(name_label, 0, 0)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter person's name")
        self.name_input.setMinimumHeight(35)
        form_layout.addWidget(self.name_input, 0, 1)
        
        id_label = QLabel("Person ID")
        id_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        form_layout.addWidget(id_label, 1, 0)
        
        self.id_input = QLineEdit()
        self.id_input.setReadOnly(True)
        self.id_input.setText(self.backend.get_next_person_id())
        self.id_input.setStyleSheet("background-color: #2b2e38; color: #888888; border: 1px solid #1e2233;")
        self.id_input.setMinimumHeight(35)
        form_layout.addWidget(self.id_input, 1, 1)
        
        left_layout.addLayout(form_layout)
        
        # Live Buffered Thumbnails
        images_title = QLabel("Captured Buffers")
        images_title.setFont(QFont("Arial", 13, QFont.Bold))
        images_title.setStyleSheet("color: #e0e0e0;")
        left_layout.addWidget(images_title)
        
        self.thumb_labels = {}
        thumbs_layout = QHBoxLayout()
        for angle in ['front', 'left', 'right']:
            box = QFrame()
            box.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
            box.setStyleSheet("background-color: #1a1c23; border: 1px solid #2e313f; border-radius: 4px;")
            box_lay = QVBoxLayout()
            
            lbl_img = QLabel("Empty")
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setStyleSheet("color: #4c566a;")
            lbl_img.setFixedSize(90, 90)
            box_lay.addWidget(lbl_img)
            self.thumb_labels[angle] = lbl_img
            
            lbl_title = QLabel(angle.upper())
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("color: #888888; font-size: 10px;")
            box_lay.addWidget(lbl_title)
            
            box.setLayout(box_lay)
            thumbs_layout.addWidget(box)
        left_layout.addLayout(thumbs_layout)
        
        # Execute Commit Sequence Actions
        actions_layout = QHBoxLayout()
        self.register_btn = QPushButton("✓ Commit to System")
        self.register_btn.setMinimumHeight(45)
        self.register_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; font-weight: bold; font-size: 13px; border-radius: 4px;} QPushButton:hover { background-color: #66bb6a; }")
        self.register_btn.clicked.connect(self.submit_registration)
        
        reset_form_btn = QPushButton("Clear Form")
        reset_form_btn.setMinimumHeight(45)
        reset_form_btn.clicked.connect(self.reset_entire_form)
        
        actions_layout.addWidget(self.register_btn)
        actions_layout.addWidget(reset_form_btn)
        left_layout.addLayout(actions_layout)
        
        left_layout.addStretch()
        main_layout.addLayout(left_layout, 4)

        # --- RIGHT SECTION: Camera Control & Target Angles ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        right_layout.insertSpacing(0, 48)

        
        title = QLabel("Camera Control")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        right_layout.addWidget(title)
        
        # Camera Setup Row
        cam_layout = QHBoxLayout()
        cam_label = QLabel("Camera Index:")
        cam_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        self.cam_input = QLineEdit("0")
        self.cam_input.setFixedWidth(50)
        self.cam_input.setAlignment(Qt.AlignCenter)
        self.cam_input.setStyleSheet("background-color: #2e3440; color: white; border: 1px solid #4c566a; border-radius: 4px; padding: 4px;")
        
        self.start_cam_btn = QPushButton("🔌 Start Camera")
        self.start_cam_btn.setStyleSheet("background-color: #5e81ac; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.start_cam_btn.clicked.connect(self.manage_camera)
        
        cam_layout.addWidget(cam_label)
        cam_layout.addWidget(self.cam_input)
        cam_layout.addWidget(self.start_cam_btn)
        cam_layout.addStretch()
        right_layout.addLayout(cam_layout)
        
        # Webcam Feed Frame
        preview_frame = QFrame()
        preview_frame.setStyleSheet("QFrame { background-color: #000000; border: 2px solid #1e2233; border-radius: 8px; }")
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        self.preview_label = QLabel("📹 Camera Disconnected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFont(QFont("Arial", 14))
        self.preview_label.setStyleSheet("color: #666666;")
        self.preview_label.setFixedSize(900, 500)
        preview_layout.addWidget(self.preview_label)
        preview_frame.setLayout(preview_layout)
        right_layout.addWidget(preview_frame)
        
        # Capture Angle Checklist Tracker
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(10)
        
        progress_label = QLabel("Capture Flow Steps")
        progress_label.setFont(QFont("Arial", 11, QFont.Bold))
        progress_label.setStyleSheet("color: #e0e0e0;")
        progress_layout.addWidget(progress_label)
        
        self.angle_checks = {}
        angles_display = [("front", "1. Front Face"), ("left", "2. Left Profile"), ("right", "3. Right Profile")]
        
        for key, display_text in angles_display:
            angle_row = QHBoxLayout()
            checkbox = QLabel("❌")
            checkbox.setFont(QFont("Arial", 12))
            angle_row.addWidget(checkbox)
            self.angle_checks[key] = checkbox
            
            angle_text = QLabel(display_text)
            angle_text.setStyleSheet("color: #a0a0a0; font-size: 13px;")
            angle_row.addWidget(angle_text)
            angle_row.addStretch()
            progress_layout.addLayout(angle_row)
            
        right_layout.addLayout(progress_layout)
        
        # Immediate Stage Controls
        capture_layout = QHBoxLayout()
        self.capture_btn = QPushButton("📸 Capture Photo")
        self.capture_btn.setMinimumHeight(40)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("QPushButton { background-color: #00bfff; color: white; border-radius: 4px; font-weight: bold; } QPushButton:disabled { background-color: #4c566a; color: #a0a0a0; }")
        self.capture_btn.clicked.connect(self.capture_angle)
        
        retake_btn = QPushButton("🔄 Clear Sequence")
        retake_btn.setMinimumHeight(40)
        retake_btn.clicked.connect(self.reset_capture_sequence)
        
        capture_layout.addWidget(self.capture_btn)
        capture_layout.addWidget(retake_btn)
        right_layout.addLayout(capture_layout)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, 5)

        self.setLayout(main_layout)

    def manage_camera(self):
        if self.cap is None:
            try:
                cam_idx = int(self.cam_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Error", "Camera index must be an integer.")
                return

            self.cap = CameraStream(cam_idx, f"Camera {cam_idx}")
            if not self.cap.start():
                QMessageBox.critical(self, "Hardware Error", f"Could not bind to camera index {cam_idx}")
                self.cap = None
                return

            self.timer.start(30)
            self.start_cam_btn.setText("🛑 Stop Camera")
            self.start_cam_btn.setStyleSheet("background-color: #bf616a; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
            self.capture_btn.setEnabled(True)
            self.cam_input.setReadOnly(True)
        else:
            self.shutdown_camera()

    def shutdown_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.stop()  # CameraStream's own cleanup
            self.cap = None
        self.preview_label.setText("📹 Camera Disconnected")
        self.preview_label.setPixmap(QPixmap())
        self.start_cam_btn.setText("🔌 Start Camera")
        self.start_cam_btn.setStyleSheet("background-color: #5e81ac; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        self.capture_btn.setEnabled(False)
        self.cam_input.setReadOnly(False)

    def update_webcam_frame(self):
        if self.cap is None:
            return

        frame = self.cap.get_frame()
        if frame is not None:
            self.current_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)

    # --- LOGICAL TRANSACTIONS ---
    def capture_angle(self):
        """Encodes captured array into standard Base64 string formats required by backend"""
        if self.current_frame is None:
            return
            
        if self.current_angle_index >= len(self.angle_order):
            QMessageBox.information(self, "Complete", "All 3 target angles have already been loaded into buffers.")
            return

        target_angle = self.angle_order[self.current_angle_index]
        
        # Convert matrix into JPEG byte buffer arrays
        ret, buffer = cv2.imencode('.jpeg', self.current_frame)
        if not ret:
            return
            
        # Standardize matching signature to string format
        b64_str = base64.b64encode(buffer).decode('utf-8')
        self.captured_images_b64[target_angle] = f"data:image/jpeg;base64,{b64_str}"
        
        # Update progress visual assets indicators
        self.angle_checks[target_angle].setText("✅")
        
        # Render tiny structural visual icons directly inside the matrix thumbnail previews
        rgb_thumb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_thumb.shape
        qt_thumb = QImage(rgb_thumb.data, w, h, ch * w, QImage.Format_RGB888)
        pix_thumb = QPixmap.fromImage(qt_thumb).scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.thumb_labels[target_angle].setPixmap(pix_thumb)
        
        self.current_angle_index += 1

    def submit_registration(self):
        """Validates entry fields and handles asynchronous operational processing steps safely"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Field Missing", "Registration cannot proceed without a Valid Name parameter.")
            return
            
        if len(self.captured_images_b64) < 3:
            QMessageBox.warning(self, "Buffers Empty", "Please capture all three required angles (Front, Left, Right) before committing.")
            return

        self.register_btn.setEnabled(False)
        self.register_btn.setText("Processing Backend Engine Async...")

        # Offload face alignment (MTCNN) and generation vectors (InceptionResnet) to Worker Thread
        self.worker = WorkerThread(self.backend, name, self.captured_images_b64)
        self.worker.result_signal.connect(self.handle_backend_response)
        self.worker.start()

    def handle_backend_response(self, response):
        """Processes responses sent back from backend thread loops safely"""
        self.register_btn.setEnabled(True)
        self.register_btn.setText("✓ Commit to System")
        
        if response.get("success"):
            QMessageBox.information(self, "Success", response.get("message", "Registered successfully!"))
            self.reset_entire_form()
        else:
            error_msg = response.get("error", "Unknown pipeline error.")
            issues = response.get("issues", [])
            if issues:
                error_msg += "\nInconsistencies:\n" + "\n".join(issues)
            QMessageBox.critical(self, "Pipeline Rollback Error", error_msg)

    # --- STATE RESET CONTROLS ---
    def reset_capture_sequence(self):
        """Flushes immediate layout photo streams out of device caches safely"""
        self.captured_images_b64.clear()
        self.current_angle_index = 0
        for key in self.angle_order:
            self.angle_checks[key].setText("❌")
            self.thumb_labels[key].setPixmap(QPixmap())
            self.thumb_labels[key].setText("Empty")

    def reset_entire_form(self):
        """Wipes form configuration components clean cleanly"""
        self.name_input.clear()
        self.reset_capture_sequence()
        self.id_input.setText(self.backend.get_next_person_id())

    def _safe_cleanup(self):
        """Cleanup resources when application closes"""
        self.shutdown_camera()
        print("✅ RegistrationPage cleanup complete - Camera resources cleanly released.")

    # def closeEvent(self, event):
    #     """Graceful hardware cleanup overrides on interface closes"""
    #     self._safe_cleanup()
    #     event.accept()
