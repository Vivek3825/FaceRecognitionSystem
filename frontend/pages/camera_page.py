"""
Camera monitor page - multi-camera live monitoring
"""
import time
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QComboBox, QSlider,
    QSpinBox, QDialog, QApplication, QScrollArea, QPlainTextEdit
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QFont, QImage, QPixmap
from frontend.widgets import CameraCard, BaseCard, AlertWidget

from backend.src.multi_camera_manager import MultiCameraManager

LARGE_FEED_W = 800
LARGE_FEED_H = 420

class ModelLoaderThread(QThread):
    """Loads heavy PyTorch models on a separate CPU core to prevent UI freezing"""
    finished_loading = Signal(object) 

    def run(self):
        # This completely bypasses the UI thread and GIL contention!
        from backend.src.multi_camera_manager import MultiCameraManager
        manager = MultiCameraManager() 
        self.finished_loading.emit(manager)

class CameraMonitorPage(QWidget):
    """Multi-camera live monitoring page"""

    # CHANGED: accepts camera_manager from main_window
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_cards = {}              # { camera_name: CameraCard }
        self.frame_times = {}              # { camera_name: last_frame_time } for FPS
        self.selected_camera = None

        self.init_ui()
        # NEW: Listen for the global app quit signal
        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self._safe_cleanup)

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ── LEFT SECTION ─────────────────────────────────────────────
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)

        title = QLabel("Camera Monitor")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        left_layout.addWidget(title)

        # Controls bar
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        view_label = QLabel("View Mode:")
        view_label.setStyleSheet("color: #a0a0a0;")
        controls_layout.addWidget(view_label)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["2x2 Grid", "1x4 Grid", "2x3 Grid"])
        self.view_combo.setMaximumWidth(150)
        self.view_combo.setMinimumHeight(30)
        self.view_combo.currentTextChanged.connect(self._change_grid_layout) #()
        controls_layout.addWidget(self.view_combo)

        load_btn = QPushButton("Load")
        load_btn.setMaximumWidth(120)
        load_btn.setMinimumHeight(30)
        load_btn.setStyleSheet("""
            QPushButton { background-color: #007BFF; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #ff5555; }
        """)
        load_btn.clicked.connect(self._load_button)
        controls_layout.addWidget(load_btn)
        
        start_btn = QPushButton("Start")
        # start_btn.setToolTip("Make sure to click 'Load' before starting!")
        start_btn.setMaximumWidth(120)
        start_btn.setMinimumHeight(30)
        start_btn.setStyleSheet("""
            QPushButton { background-color: #00C853; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #ff5555; }
        """)
        start_btn.clicked.connect(self._start_button)
        controls_layout.addWidget(start_btn)

        stop_btn = QPushButton("Stop")
        # stop_btn.setToolTip("Make sure you have stared cameras")
        stop_btn.setMaximumWidth(120)
        stop_btn.setMinimumHeight(30)
        stop_btn.setStyleSheet("""
            QPushButton { background-color: #D32F2F; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #ff5555; }
        """)
        stop_btn.clicked.connect(self._stop_button)
        controls_layout.addWidget(stop_btn)

        controls_layout.addStretch()
        left_layout.addLayout(controls_layout)


        # ── WARNING LABEL ──
        self.warning_label = QLabel("")  # Start with empty text
        self.warning_label.setStyleSheet("color: #ff5555; font-weight: bold;") # Style it Red
        self.warning_label.hide() # Hide it so the user doesn't see it yet
        left_layout.addWidget(self.warning_label) # Add it just under the buttons

        # ── FIXED DIMENSION GRID BLOCK ───────────────────────────────
        self.grid_container = QFrame()
        self.grid_container.setFixedSize(LARGE_FEED_W, LARGE_FEED_H)
        self.grid_container.setStyleSheet("""
            QFrame { background-color: #0b0e17; border-radius: 8px; border: 1px solid #1e2233; }
        """)

        self.camera_grid = QGridLayout(self.grid_container)
        self.camera_grid.setContentsMargins(10,10,10,10)
        self.camera_grid.setSpacing(10)

        left_layout.addWidget(self.grid_container)

        # Detection list
        detection_label = QLabel("Live Detections")
        detection_label.setFont(QFont("Arial", 14, QFont.Bold))
        detection_label.setStyleSheet("color: #e0e0e0;")
        left_layout.addWidget(detection_label)

        self.detections_layout = QVBoxLayout()
        self.detections_layout.setSpacing(8)
        left_layout.addLayout(self.detections_layout)

        left_layout.addStretch()
        main_layout.addLayout(left_layout)

        # ── RIGHT SECTION ─────────────────────────────────────────────
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        right_layout.insertSpacing(0, 72)

        selected_title = QLabel("Selected Camera")
        selected_title.setFont(QFont("Arial", 16, QFont.Bold))
        selected_title.setStyleSheet("color: #e0e0e0;")
        right_layout.addWidget(selected_title)

        # Large feed for selected camera
        large_feed = QFrame()
        large_feed.setStyleSheet("""
            QFrame { background-color: #000000; border: 2px solid #1e2233; border-radius: 8px; }
        """)
        large_feed_layout = QVBoxLayout()
        large_feed_layout.setContentsMargins(0, 0, 0, 0)

        # self.large_feed_label so we can update it
        self.large_feed_label = QLabel("Select a camera to view")
        self.large_feed_label.setAlignment(Qt.AlignCenter)
        self.large_feed_label.setFont(QFont("Arial", 14))
        self.large_feed_label.setStyleSheet("color: #666666;")
        self.large_feed_label.setFixedSize(LARGE_FEED_W, LARGE_FEED_H)
        large_feed_layout.addWidget(self.large_feed_label)
        large_feed.setLayout(large_feed_layout)
        right_layout.addWidget(large_feed)

        # Camera controls
        camera_detail = QLabel("Camera Details.")
        camera_detail.setFont(QFont("Arial", 16, QFont.Bold))
        camera_detail.setStyleSheet("color: #e0e0e0;")
        right_layout.addWidget(camera_detail)

        self.camera_detail_layout = QVBoxLayout()
        self.camera_detail_layout.setSpacing(8)
        right_layout.addLayout(self.camera_detail_layout)

        # camera_detais = QPlainTextEdit()
        # self.output_log.setReadOnly(True)  # Prevents user typing, makes it a pure "display" box

        right_layout.addStretch()
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def _load_button(self):
            self._show_temporary_warning("⏳ Loading AI Models... Please wait. This may take 30-40 seconds.")
            self.loader_thread = ModelLoaderThread()
            self.loader_thread.finished_loading.connect(self._on_models_loaded)
            self.loader_thread.start()

    def _on_models_loaded(self, manager):
        """This runs automatically the exact millisecond PyTorch is ready"""
        self.manager = manager
        self._show_temporary_warning("✅ AI Models Loaded Successfully! You can now click Start.")

    def _start_button(self):
        if hasattr(self, 'manager'):
            self.check = True
            self._start_cameras()              # start real cameras
            self._start_timer()               # start frame update loop
            self._camera_details()
        else:
            self._show_temporary_warning("⚠️ Please click 'Load' before starting.")

    def _stop_button(self):
        if hasattr(self, 'manager'):
            if hasattr(self, 'avl_cameras'):
                self.manager.cleanup()
                self._clear_camera_details()
                self._reset_camera_view()

                self._show_temporary_warning(
                    "✅ Cameras stopped successfully."
                )
            else:
                self._show_temporary_warning(
                    "⚠️ Please start the cameras first."
                )
        else:
            self._show_temporary_warning(
                "⚠️ Please click 'Load' first."
            )


    def _show_temporary_warning(self, message):
        self.warning_label.setText(message)
        self.warning_label.show()
        
        # Automatically hide the label after 3500 ms (3.5 seconds)
        QTimer.singleShot(3500, self.warning_label.hide)

    # ── GRID LOGIC ──────────────────────────────────────────
    def _change_grid_layout(self, mode_str):
        """Forces the grid layout into specific columns/rows"""
        # 1. Parse string for target dimensions (Cols x Rows)
        if "1x4" in mode_str:
            cols, rows = 4, 1   # 1 Row, 4 Columns (Horizontal strip)
        elif "2x3" in mode_str:
            cols, rows = 3, 2   # 2 Rows, 3 Columns
        else:
            cols, rows = 2, 2   # 2x2 Default

        # 2. Clear existing widgets from layout
        while self.camera_grid.count():
            item = self.camera_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()

        # 3. Reset stretches (Clear old divisions)
        for i in range(10): # Assume a max of 10 rows/cols safely
            self.camera_grid.setColumnStretch(i, 0)
            self.camera_grid.setRowStretch(i, 0)

        # 4. Force rigid cell sizes using stretches
        for c in range(cols):
            self.camera_grid.setColumnStretch(c, 1)
        for r in range(rows):
            self.camera_grid.setRowStretch(r, 1)

        # 5. Populate grid sequentially (0,0 -> 0,1 -> 1,0 ...)
        available_cameras = list(self.camera_cards.values())
        for i, card in enumerate(available_cameras):
            row = i // cols
            col = i % cols
            
            # Add to grid only if it fits inside the specified layout bounds
            if row < rows:
                self.camera_grid.addWidget(card, row, col)
                card.show()

    def _camera_details(self):
        avl_camera = QLabel(f"Available cameras: {len(self.avl_cameras)}")
        avl_camera.setFont(QFont("Arial", 14, QFont.Bold))
        avl_camera.setStyleSheet("color: #e0e0e0;")
        self.camera_detail_layout.addWidget(avl_camera)

        for cameras in self.avl_cameras:
            avl_camera = QLabel(f"{cameras}")
            avl_camera.setFont(QFont("Arial", 14, QFont.Bold))
            avl_camera.setStyleSheet("color: #e0e0e0;")
            self.camera_detail_layout.addWidget(avl_camera)

    def _clear_camera_details(self):
            # Keep looping as long as there is at least 1 item in the layout
            while self.camera_detail_layout.count():
                # Take the first item out of the layout
                item = self.camera_detail_layout.takeAt(0)
                
                # Check if the item is a widget
                widget = item.widget()
                if widget is not None:
                    # Delete the widget properly to free up memory
                    widget.deleteLater()

    # NEW: creates real CameraCards from MultiCameraManager cameras
    def _start_cameras(self):
        self.manager.start_default_cameras()
        self.avl_cameras = []

        for i, camera_name in enumerate(self.manager.get_all_camera_names()):
            self.avl_cameras.append(camera_name)
            card = CameraCard(camera_name, "Active")
            card.clicked.connect(lambda n=camera_name: self.on_camera_selected(n))
            self.camera_cards[camera_name] = card
            self.camera_grid.addWidget(card, i // 2, i % 2)  # 2-column grid
        
        self._change_grid_layout(self.view_combo.currentText())

    def _start_timer(self):
        # Frame timer — fast, for video
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._update_frames_only)
        self.frame_timer.start(33)  # 30fps

        # Detection timer — slow, for the list
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self._update_detection_list)
        self.detection_timer.start(1500)  # once per 1.5 seconds

    # NEW: main loop — runs 30x per second
    def _update_frames_only(self):
        # Only frame + recognition work here, NO detection list call
        for camera_name, card in self.camera_cards.items():
            camera = self.manager.cameras.get(camera_name)
            if not camera:
                continue

            frame = camera.get_frame()
            if frame is None:
                continue

            results = self.manager.process_frame_recognition(frame, camera_name)
            self.manager.update_recognition_results(camera_name, results)

            annotated = self._draw_boxes(frame, results)
            card.update_frame(annotated)
            card.update_status(camera.is_running)

            now = time.time()
            last = self.frame_times.get(camera_name, now)
            elapsed = now - last
            fps = 1.0 / elapsed if elapsed > 0 else 0
            card.update_fps(fps)
            self.frame_times[camera_name] = now

            if camera_name == self.selected_camera:
                self._update_large_feed(annotated)

    # NEW: draws green/red boxes around detected faces
    def _draw_boxes(self, frame, results):
        out = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            is_known = r['name'] != "Unknown"
            color = (0, 255, 0) if is_known else (0, 0, 255)  # green/red
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{r['name']} ({r['confidence']:.0%})"
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out

    # NEW: shows selected camera in the large right-panel feed
    def _update_large_feed(self, frame):
        target_w = LARGE_FEED_W
        target_h = LARGE_FEED_H

        frame_h, frame_w = frame.shape[:2]
        scale = max(target_w / frame_w, target_h / frame_h)
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)
        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        x_start = max((scaled_w - target_w) // 2, 0)
        y_start = max((scaled_h - target_h) // 2, 0)
        cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]

        if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
            cropped = cv2.resize(cropped, (target_w, target_h))

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.large_feed_label.setPixmap(QPixmap.fromImage(qimg))

    # NEW: refreshes detection list from real recognition results
    def _update_detection_list(self):
        # Clear old alerts
        while self.detections_layout.count():
            item = self.detections_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not hasattr(self, 'manager'):
            return

        cutoff = time.time() - 5

        # Step 1: Collect best result per (name, camera) pair
        per_camera_best = {}  # key: (name, camera) → best result dict
        for camera_name, results in self.manager.recognition_results.items():
            for r in results:
                if r['timestamp'] < cutoff:
                    continue
                key = (r['name'], camera_name)
                if key not in per_camera_best or r['confidence'] > per_camera_best[key]['confidence']:
                    per_camera_best[key] = r

        # Step 2: Global merge by name — same person across cameras → one entry
        global_persons = {}  # name → {confidence, cameras: [], timestamp}
        for (name, camera_name), r in per_camera_best.items():
            if name not in global_persons:
                global_persons[name] = {
                    'name': name,
                    'confidence': r['confidence'],
                    'cameras': [camera_name],
                    'timestamp': r['timestamp']
                }
            else:
                entry = global_persons[name]
                if camera_name not in entry['cameras']:
                    entry['cameras'].append(camera_name)
                # Keep highest confidence across cameras
                if r['confidence'] > entry['confidence']:
                    entry['confidence'] = r['confidence']
                if r['timestamp'] > entry['timestamp']:
                    entry['timestamp'] = r['timestamp']

        # Step 3: Sort by most recent, display up to 6
        sorted_persons = sorted(global_persons.values(), 
                            key=lambda x: x['timestamp'], reverse=True)

        for entry in sorted_persons[:6]:
            camera_str = ", ".join(entry['cameras'])  # "Camera 1, Camera 2"
            severity = "success" if entry['name'] != "Unknown" else "info"
            alert = AlertWidget(
                entry['name'],
                f"{camera_str} — {entry['confidence']:.0%} confidence",
                severity
            )
            self.detections_layout.addWidget(alert)

    def on_camera_selected(self, camera_name):
        self.selected_camera = camera_name

        # Highlight selected card, unhighlight others
        for name, card in self.camera_cards.items():
            if name == camera_name:
                card.setStyleSheet("""
                    QFrame {
                        background-color: #141829;
                        border: 2px solid #00bfff;
                        border-radius: 8px;
                        padding: 15px;
                    }
                """)
            else:
                card.setStyleSheet("""
                    QFrame {
                        background-color: #141829;
                        border: 1px solid #1e2233;
                        border-radius: 8px;
                        padding: 15px;
                    }
                """)

    def _reset_camera_view(self):
        # Stop timers
        if hasattr(self, 'frame_timer'):
            self.frame_timer.stop()

        if hasattr(self, 'detection_timer'):
            self.detection_timer.stop()

        # Remove all camera cards from grid
        while self.camera_grid.count():
            item = self.camera_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Clear camera data
        self.camera_cards.clear()
        self.frame_times.clear()
        self.selected_camera = None

        # Reset right-side preview
        self.large_feed_label.clear()
        self.large_feed_label.setText("Select a camera to view")

        # Clear detections
        while self.detections_layout.count():
            item = self.detections_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Restore empty grid appearance
        placeholder = QLabel("No cameras running")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666666; font-size: 16px;")
        self.camera_grid.addWidget(placeholder, 0, 0)

    
    def _safe_cleanup(self):
        """Safely stops timers and releases cameras before the app closes."""
        # Safely stop frame timer
        if hasattr(self, 'frame_timer') and self.frame_timer.isActive():
            self.frame_timer.stop()
            
        # Safely stop detection timer
        if hasattr(self, 'detection_timer') and self.detection_timer.isActive():
            self.detection_timer.stop()
            
        # Safely cleanup cameras
        if hasattr(self, 'manager'):
            self.manager.cleanup()
            
        print("✅ CameraPage cleanup complete - Camera resources cleanly released.")
