"""
Reusable card widgets for displaying information
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QImage, QColor, QPainter
import cv2

FEED_W = 400
FEED_H = 225

class BaseCard(QFrame):
    """Base card widget with rounded corners and shadow"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background-color: #141829;
                border: 1px solid #1e2233;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        self.setMinimumHeight(100)

class StatsCard(BaseCard):
    """Card for displaying statistics"""
    
    def __init__(self, title, value, subtitle="", icon="📊", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Header with icon and title
        header_layout = QHBoxLayout()
        
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 20))
        icon_label.setMaximumWidth(50)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("color: #a0a0a0;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Value
        value_label = QLabel(str(value))
        value_font = QFont("Arial", 28, QFont.Bold)
        value_label.setFont(value_font)
        value_label.setStyleSheet("color: #00bfff;")
        layout.addWidget(value_label)
        
        # Subtitle
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setFont(QFont("Arial", 10))
            subtitle_label.setStyleSheet("color: #666666;")
            layout.addWidget(subtitle_label)
        
        layout.addStretch()
        self.setLayout(layout)

class CameraCard(BaseCard):
    """Card for displaying camera feed and controls"""

    clicked = Signal()
    fullscreen_requested = Signal()

    def __init__(self, camera_name="Camera 1", status="Active", parent=None):
        super().__init__(parent)
        self.camera_name = camera_name
        self.status = status
        
        # Setup constraints
        #self.setFixedSize(FEED_W + 20, FEED_H + 60) # Reduced height since info is hidden
        self.setMinimumSize(150, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Feed Section ───────────────────────────────────────────
        feed_frame = QFrame()
        feed_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border: none;
                border-radius: 4px;
            }
        """)
        
        feed_layout = QVBoxLayout(feed_frame)
        feed_layout.setContentsMargins(0, 0, 0, 0)

        self.feed_label = QLabel("📹 Connecting...")
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setStyleSheet("color: #666666; font-size: 14px;")
        #self.feed_label.setFixedSize(FEED_W, FEED_H)
        self.feed_label.setScaledContents(False)
        self.feed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        feed_layout.addWidget(self.feed_label)
        layout.addWidget(feed_frame)

        # ── Info Section (HIDDEN BUT INITIALIZED) ──────────────────
        self.name_label = QLabel(self.camera_name)
        self.name_label.hide() 
        
        self.status_label = QLabel(status)
        self.status_label.hide() 
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.hide() 

    def update_frame(self, frame):
        if frame is None:
            return

        target_w = self.feed_label.width()   # hardcoded constants, NOT feed_label.width()
        target_h = self.feed_label.height()

        if target_w <= 0 or target_h <= 0:
            return


        frame_h, frame_w = frame.shape[:2]

        # Cover fit — fills box completely, crops centre
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
        self.feed_label.setPixmap(QPixmap.fromImage(qimg))

    # updates FPS label
    def update_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.0f}")

    # updates status label color
    def update_status(self, active: bool):
        if active:
            self.status_label.setText("● Active")
            self.status_label.setStyleSheet("color: #4caf50; font-size: 11px;")
        else:
            self.status_label.setText("● Inactive")
            self.status_label.setStyleSheet("color: #ff4444; font-size: 11px;")

    # emit clicked signal when user clicks the card
    # Before this, clicked signal was defined but never triggered!
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)



class PersonCard(BaseCard):
    """Card for displaying person information"""
    
    def __init__(self, name="John Doe", person_id="P001", confidence=95, 
                 last_seen="Camera 1", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Person image placeholder
        img_frame = QFrame()
        img_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border: 1px solid #2a3452;
                border-radius: 4px;
            }
        """)
        img_layout = QVBoxLayout()
        img_layout.setContentsMargins(0, 0, 0, 0)
        
        img_label = QLabel("👤")
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setFont(QFont("Arial", 40))
        img_label.setMinimumHeight(80)
        img_layout.addWidget(img_label)
        img_frame.setLayout(img_layout)
        layout.addWidget(img_frame)
        
        # Person info
        name_label = QLabel(name)
        name_label.setFont(QFont("Arial", 13, QFont.Bold))
        name_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(name_label)
        
        id_label = QLabel(f"ID: {person_id}")
        id_label.setFont(QFont("Arial", 10))
        id_label.setStyleSheet("color: #a0a0a0;")
        layout.addWidget(id_label)
        
        # Confidence
        conf_layout = QHBoxLayout()
        conf_text = QLabel("Confidence:")
        conf_text.setStyleSheet("color: #a0a0a0; font-size: 10px;")
        conf_layout.addWidget(conf_text)
        
        conf_value = QLabel(f"{confidence}%")
        conf_value.setStyleSheet("color: #4caf50; font-weight: bold;")
        conf_layout.addWidget(conf_value)
        conf_layout.addStretch()
        layout.addLayout(conf_layout)
        
        # Last seen
        seen_label = QLabel(f"Last seen: {last_seen}")
        seen_label.setFont(QFont("Arial", 9))
        seen_label.setStyleSheet("color: #666666;")
        layout.addWidget(seen_label)
        
        layout.addStretch()
        self.setLayout(layout)


class AlertWidget(BaseCard):
    """Card for displaying alerts"""
    
    def __init__(self, title="Alert", message="", severity="info", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        
        # Set severity color
        severity_colors = {
            "info": "#00bfff",
            "warning": "#ffb300",
            "danger": "#ff4444",
            "success": "#4caf50"
        }
        color = severity_colors.get(severity, "#00bfff")
        
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        # Severity indicator
        icon_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "danger": "🚨",
            "success": "✓"
        }
        
        icon_label = QLabel(icon_map.get(severity, "ℹ️"))
        icon_label.setFont(QFont("Arial", 20))
        icon_label.setMaximumWidth(40)
        layout.addWidget(icon_label)
        
        # Message section
        msg_layout = QVBoxLayout()
        msg_layout.setSpacing(5)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet(f"color: {color};")
        msg_layout.addWidget(title_label)
        
        if message:
            msg_text = QLabel(message)
            msg_text.setFont(QFont("Arial", 10))
            msg_text.setStyleSheet("color: #a0a0a0;")
            msg_text.setWordWrap(True)
            msg_layout.addWidget(msg_text)
        
        layout.addLayout(msg_layout)
        layout.addStretch()
        
        # Dismiss button
        dismiss_btn = QPushButton("✕")
        dismiss_btn.setMaximumWidth(35)
        dismiss_btn.setMaximumHeight(35)
        dismiss_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #a0a0a0;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1e2844;
                border-radius: 4px;
            }
        """)
        layout.addWidget(dismiss_btn)
        
        self.setLayout(layout)
