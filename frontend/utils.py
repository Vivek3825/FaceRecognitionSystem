"""
Utility functions for the frontend application
"""
from PySide6.QtCore import QSize, Qt, QRect
from PySide6.QtGui import QFont, QPixmap, QIcon, QColor, QPainter
from PySide6.QtWidgets import QApplication, QMessageBox
from datetime import datetime
import json


def get_screen_geometry():
    """Get the primary screen geometry"""
    app = QApplication.instance()
    if app:
        screen = app.primaryScreen()
        return screen.geometry()
    return QRect(0, 0, 1920, 1080)


def center_window(window):
    """Center a window on the screen"""
    screen_geometry = get_screen_geometry()
    window_geometry = window.frameGeometry()
    center_point = screen_geometry.center()
    window_geometry.moveCenter(center_point)
    window.move(window_geometry.topLeft())


def create_rounded_pixmap(pixmap, radius=10):
    """Create a rounded pixmap"""
    size = pixmap.size()
    
    # Create a new pixmap with alpha channel
    rounded_pixmap = QPixmap(size)
    rounded_pixmap.fill(QColor(0, 0, 0, 0))
    
    # Draw rounded rectangle
    painter = QPainter(rounded_pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.drawRoundedRect(
        0, 0, size.width(), size.height(),
        radius, radius
    )
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()
    
    return rounded_pixmap


def format_timestamp(dt=None, format_str="%Y-%m-%d %H:%M:%S"):
    """Format datetime to string"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)


def parse_timestamp(timestamp_str, format_str="%Y-%m-%d %H:%M:%S"):
    """Parse timestamp string to datetime"""
    try:
        return datetime.strptime(timestamp_str, format_str)
    except ValueError:
        return None


def format_filesize(size_bytes):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_confidence(confidence):
    """Format confidence score"""
    if isinstance(confidence, (int, float)):
        return f"{confidence:.0f}%"
    return str(confidence)


def validate_email(email):
    """Validate email address"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def show_message_box(parent, title, message, msg_type="information"):
    """Show message box dialog"""
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    
    if msg_type == "information":
        msg_box.setIcon(QMessageBox.Information)
    elif msg_type == "warning":
        msg_box.setIcon(QMessageBox.Warning)
    elif msg_type == "error":
        msg_box.setIcon(QMessageBox.Critical)
    elif msg_type == "question":
        msg_box.setIcon(QMessageBox.Question)
    
    msg_box.exec()


def show_confirm_dialog(parent, title, message):
    """Show confirmation dialog and return result"""
    reply = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    return reply == QMessageBox.Yes


def load_json(filepath):
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def save_json(data, filepath):
    """Save data to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False


def get_font(family="Arial", size=12, bold=False, italic=False):
    """Create a QFont with given parameters"""
    font = QFont(family)
    font.setPointSize(size)
    font.setBold(bold)
    font.setItalic(italic)
    return font


def get_icon_from_text(text, size=24, color="#00bfff"):
    """Create a simple icon from text/emoji"""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    
    painter = QPainter(pixmap)
    painter.setFont(get_font(size=size // 2))
    painter.setPen(QColor(color))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
    painter.end()
    
    return QIcon(pixmap)


def clamp(value, min_value, max_value):
    """Clamp value between min and max"""
    return max(min_value, min(value, max_value))


def interpolate(start, end, factor):
    """Linear interpolation between two values"""
    return start + (end - start) * factor


def get_system_info():
    """Get system information"""
    import platform
    import os
    
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def create_status_badge(status, status_map=None):
    """Create a status badge dictionary"""
    if status_map is None:
        status_map = {
            "active": ("●", "#4caf50"),
            "inactive": ("●", "#999999"),
            "error": ("●", "#ff4444"),
            "warning": ("●", "#ffb300"),
            "processing": ("⏳", "#00bfff"),
        }
    
    if status in status_map:
        icon, color = status_map[status]
        return {"icon": icon, "color": color, "status": status}
    
    return {"icon": "●", "color": "#999999", "status": "unknown"}


def format_detection(person_name, confidence, camera, timestamp):
    """Format a detection result"""
    return {
        "person": person_name,
        "confidence": format_confidence(confidence),
        "camera": camera,
        "timestamp": format_timestamp(timestamp),
    }


class Logger:
    """Simple logging utility"""
    
    @staticmethod
    def info(message):
        """Log info message"""
        timestamp = format_timestamp()
        print(f"[{timestamp}] INFO: {message}")
    
    @staticmethod
    def warning(message):
        """Log warning message"""
        timestamp = format_timestamp()
        print(f"[{timestamp}] WARNING: {message}")
    
    @staticmethod
    def error(message):
        """Log error message"""
        timestamp = format_timestamp()
        print(f"[{timestamp}] ERROR: {message}")
    
    @staticmethod
    def debug(message):
        """Log debug message"""
        timestamp = format_timestamp()
        print(f"[{timestamp}] DEBUG: {message}")


# Export common utilities
__all__ = [
    'get_screen_geometry',
    'center_window',
    'create_rounded_pixmap',
    'format_timestamp',
    'parse_timestamp',
    'format_filesize',
    'format_confidence',
    'validate_email',
    'show_message_box',
    'show_confirm_dialog',
    'load_json',
    'save_json',
    'get_font',
    'get_icon_from_text',
    'clamp',
    'interpolate',
    'get_system_info',
    'create_status_badge',
    'format_detection',
    'Logger',
]
