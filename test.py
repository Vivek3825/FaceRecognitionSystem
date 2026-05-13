"""
Camera monitor page - multi-camera live monitoring
"""
import time
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QComboBox, QSlider,
    QSpinBox, QDialog, QApplication, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from frontend.widgets import CameraCard, BaseCard, AlertWidget


class CameraMonitorPage(QWidget):
    """Multi-camera live monitoring page"""

    # ✅ CHANGED: accepts camera_manager from main_window
    def __init__(self, camera_manager, parent=None):
        super().__init__(parent)
        self.manager = camera_manager       # MultiCameraManager instance
        self.camera_cards = {}              # { camera_name: CameraCard }
        self.frame_times = {}              # { camera_name: last_frame_time } for FPS
        self.selected_camera = None

        self.init_ui()
        self._start_cameras()              # start real cameras
        self._start_timer()               # start frame update loop

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

        view_combo = QComboBox()
        view_combo.addItems(["2x2 Grid", "1x4 Grid", "2x3 Grid"])
        view_combo.setMaximumWidth(150)
        view_combo.setMinimumHeight(30)
        controls_layout.addWidget(view_combo)

        record_btn = QPushButton("🔴 Record All")
        record_btn.setMaximumWidth(120)
        record_btn.setMinimumHeight(30)
        record_btn.setStyleSheet("""
            QPushButton { background-color: #ff4444; color: white; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #ff5555; }
        """)
        controls_layout.addWidget(record_btn)

        snapshot_btn = QPushButton("📸 Snapshot All")
        snapshot_btn.setMaximumWidth(140)
        snapshot_btn.setMinimumHeight(30)
        controls_layout.addWidget(snapshot_btn)

        controls_layout.addStretch()
        left_layout.addLayout(controls_layout)

        # ✅ CHANGED: self.camera_grid (was local variable)
        #    We add cards to it later in _start_cameras()
        self.camera_grid = QGridLayout()
        self.camera_grid.setSpacing(15)
        left_layout.addLayout(self.camera_grid)

        # Detection list
        detection_label = QLabel("Live Detections")
        detection_label.setFont(QFont("Arial", 14, QFont.Bold))
        detection_label.setStyleSheet("color: #e0e0e0;")
        left_layout.addWidget(detection_label)

        # ✅ CHANGED: self.detections_layout (was local variable)
        #    We clear and repopulate it in _update_detection_list()
        self.detections_layout = QVBoxLayout()
        self.detections_layout.setSpacing(8)
        left_layout.addLayout(self.detections_layout)
        left_layout.addStretch()

        main_layout.addLayout(left_layout)

        # ── RIGHT SECTION ─────────────────────────────────────────────
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

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

        # ✅ CHANGED: self.large_feed_label so we can update it
        self.large_feed_label = QLabel("Select a camera to view")
        self.large_feed_label.setAlignment(Qt.AlignCenter)
        self.large_feed_label.setFont(QFont("Arial", 14))
        self.large_feed_label.setStyleSheet("color: #666666;")
        self.large_feed_label.setMinimumHeight(300)
        large_feed_layout.addWidget(self.large_feed_label)
        large_feed.setLayout(large_feed_layout)
        right_layout.addWidget(large_feed)

        # Camera controls
        controls_title = QLabel("Camera Controls")
        controls_title.setFont(QFont("Arial", 14, QFont.Bold))
        controls_title.setStyleSheet("color: #e0e0e0;")
        right_layout.addWidget(controls_title)

        zoom_label = QLabel("Zoom Level")
        zoom_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        right_layout.addWidget(zoom_label)
        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setRange(100, 400)
        zoom_slider.setValue(100)
        right_layout.addWidget(zoom_slider)

        brightness_label = QLabel("Brightness")
        brightness_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        right_layout.addWidget(brightness_label)
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setRange(0, 200)
        brightness_slider.setValue(100)
        right_layout.addWidget(brightness_slider)

        right_layout.addStretch()
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    # ✅ NEW: creates real CameraCards from MultiCameraManager cameras
    def _start_cameras(self):
        self.manager.start_default_cameras()

        for i, camera_name in enumerate(self.manager.get_all_camera_names()):
            card = CameraCard(camera_name, "Active")
            card.clicked.connect(lambda n=camera_name: self.on_camera_selected(n))
            self.camera_cards[camera_name] = card
            self.camera_grid.addWidget(card, i // 2, i % 2)  # 2-column grid

    # ✅ NEW: QTimer fires every 33ms (~30 FPS)
    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_all)
        self.timer.start(33)

    # ✅ NEW: main loop — runs 30x per second
    def _update_all(self):
        for camera_name, card in self.camera_cards.items():
            camera = self.manager.cameras.get(camera_name)
            if not camera:
                continue

            frame = camera.get_frame()
            if frame is None:
                continue

            # Run face recognition on this frame
            results = self.manager.process_frame_recognition(frame, camera_name)
            self.manager.update_recognition_results(camera_name, results)

            # Draw face boxes + names on frame
            annotated = self._draw_boxes(frame, results)

            # Push frame to CameraCard
            card.update_frame(annotated)
            card.update_status(camera.is_running)

            # Calculate and show FPS
            now = time.time()
            last = self.frame_times.get(camera_name, now)
            elapsed = now - last
            fps = 1.0 / elapsed if elapsed > 0 else 0
            card.update_fps(fps)
            self.frame_times[camera_name] = now

            # If this is the selected camera, update large feed too
            if camera_name == self.selected_camera:
                self._update_large_feed(annotated)

        self._update_detection_list()

    # ✅ NEW: draws green/red boxes around detected faces
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

    # ✅ NEW: shows selected camera in the large right-panel feed
    def _update_large_feed(self, frame):
        from PySide6.QtGui import QImage, QPixmap
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.large_feed_label.width(),
            self.large_feed_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.large_feed_label.setPixmap(pixmap)

    # ✅ NEW: refreshes detection list from real recognition results
    def _update_detection_list(self):
        # Clear old alerts
        while self.detections_layout.count():
            item = self.detections_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add recent detections (last 5 seconds)
        cutoff = time.time() - 5
        shown = 0
        for camera_name, results in self.manager.recognition_results.items():
            recent = [r for r in results if r['timestamp'] >= cutoff]
            for r in recent[-2:]:  # max 2 per camera
                severity = "success" if r['name'] != "Unknown" else "info"
                alert = AlertWidget(
                    r['name'],
                    f"{camera_name} — {r['confidence']:.0%} confidence",
                    severity
                )
                self.detections_layout.addWidget(alert)
                shown += 1
                if shown >= 6:  # cap total shown alerts
                    return

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

    # ✅ NEW: stop timer and cleanup when page closes
    def closeEvent(self, event):
        self.timer.stop()
        self.manager.cleanup()
        super().closeEvent(event)