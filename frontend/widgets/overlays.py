"""
Notification and loading overlay widgets
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtGui import QFont, QMovie, QPixmap, QColor, QPainter
from PySide6.QtCore import QPropertyAnimation, QEasingCurve


class NotificationWidget(QFrame):
    """Notification popup widget"""
    
    closed = Signal()
    
    def __init__(self, title="", message="", notification_type="info", duration=5000, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setMinimumHeight(80)
        self.setMaximumWidth(400)
        
        # Type colors
        type_colors = {
            "info": "#00bfff",
            "success": "#4caf50",
            "warning": "#ffb300",
            "error": "#ff4444"
        }
        
        type_icons = {
            "info": "ℹ️",
            "success": "✓",
            "warning": "⚠️",
            "error": "❌"
        }
        
        color = type_colors.get(notification_type, "#00bfff")
        icon = type_icons.get(notification_type, "ℹ️")
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #141829;
                border: 2px solid {color};
                border-radius: 6px;
                padding: 12px;
            }}
        """)
        
        layout = QHBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 18))
        icon_label.setAlignment(Qt.AlignTop)
        icon_label.setMaximumWidth(30)
        layout.addWidget(icon_label)
        
        # Content
        content_layout = QVBoxLayout()
        content_layout.setSpacing(5)
        
        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Arial", 11, QFont.Bold))
            title_label.setStyleSheet(f"color: {color};")
            content_layout.addWidget(title_label)
        
        if message:
            msg_label = QLabel(message)
            msg_label.setFont(QFont("Arial", 10))
            msg_label.setStyleSheet("color: #a0a0a0;")
            msg_label.setWordWrap(True)
            content_layout.addWidget(msg_label)
        
        content_layout.addStretch()
        layout.addLayout(content_layout)
        
        # Close button
        close_btn = QPushButton("✕")
        close_btn.setMaximumSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #a0a0a0;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #00bfff;
            }
        """)
        close_btn.clicked.connect(self.close_notification)
        layout.addWidget(close_btn, alignment=Qt.AlignTop)
        
        self.setLayout(layout)
        
        # Auto close timer
        self.close_timer = QTimer()
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.close_notification)
        if duration > 0:
            self.close_timer.start(duration)
    
    def close_notification(self):
        """Close the notification"""
        self.close_timer.stop()
        self.closed.emit()
        self.deleteLater()


class LoadingOverlay(QWidget):
    """Full screen loading overlay with spinner"""
    
    def __init__(self, text="Loading...", parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.TransparentForMouseEvents)
        self.setStyleSheet("background-color: rgba(10, 14, 39, 200);")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # Spinner animation
        self.spinner = QLabel()
        spinner_style = """
            ╭─ ╮
            │ ◉ │
            ╰─ ╯
        """
        self.spinner.setText("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        self.spinner.setFont(QFont("Courier", 20))
        self.spinner.setAlignment(Qt.AlignCenter)
        self.spinner.setStyleSheet("color: #00bfff;")
        layout.addWidget(self.spinner)
        
        # Text
        text_label = QLabel(text)
        text_label.setFont(QFont("Arial", 13))
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(text_label)
        
        self.setLayout(layout)
        
        # Animation
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_index = 0
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_spinner)
        self.animation_timer.start(80)
    
    def animate_spinner(self):
        """Animate the spinner"""
        self.current_index = (self.current_index + 1) % len(self.spinner_chars)
        char = self.spinner_chars[self.current_index]
        self.spinner.setText(char)
    
    def stop(self):
        """Stop the animation"""
        self.animation_timer.stop()
        self.deleteLater()


class ProgressDialog(QFrame):
    """Custom progress dialog"""
    
    def __init__(self, title="Processing", steps=0, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background-color: #141829;
                border: 1px solid #1e2233;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 13, QFont.Bold))
        title_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title_label)
        
        # Progress bar
        from PySide6.QtWidgets import QProgressBar
        self.progress = QProgressBar()
        self.progress.setMinimumHeight(25)
        self.progress.setMaximum(100)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #2a3452;
                border: none;
                border-radius: 4px;
                text-align: center;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #00bfff;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress)
        
        # Status text
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #a0a0a0;")
        layout.addWidget(self.status_label)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMaximumWidth(100)
        layout.addWidget(cancel_btn)
        
        self.setLayout(layout)
    
    def set_progress(self, value, status=""):
        """Update progress"""
        self.progress.setValue(value)
        if status:
            self.status_label.setText(status)
    
    def set_max(self, max_value):
        """Set maximum progress value"""
        self.progress.setMaximum(max_value)
