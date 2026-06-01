"""
Configuration settings for the frontend application
"""
from enum import Enum
from dataclasses import dataclass
from pathlib import Path


class Theme(Enum):
    """Application themes"""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"


@dataclass
class ApplicationConfig:
    """Application configuration"""
    APP_NAME = "Face Recognition Surveillance System"
    APP_VERSION = "1.0.0"
    APP_TITLE = "Face Recognition Surveillance System"
    
    # Window settings
    DEFAULT_WINDOW_WIDTH = 1920
    DEFAULT_WINDOW_HEIGHT = 1080
    MIN_WINDOW_WIDTH = 1400
    MIN_WINDOW_HEIGHT = 800
    
    # Theme
    DEFAULT_THEME = Theme.DARK
    THEME_STYLESHEET_PATH = Path(__file__).parent / "styles" / "dark_theme.qss"
    
    # Language
    DEFAULT_LANGUAGE = Language.ENGLISH
    
    # Colors
    COLOR_PRIMARY = "#00bfff"
    COLOR_SECONDARY = "#141829"
    COLOR_BACKGROUND = "#0a0e27"
    COLOR_SUCCESS = "#4caf50"
    COLOR_WARNING = "#ffb300"
    COLOR_DANGER = "#ff4444"
    COLOR_TEXT = "#e0e0e0"
    
    # Camera settings
    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = "1080p"
    DEFAULT_CODEC = "H.264"
    
    # Recognition settings
    DEFAULT_CONFIDENCE_THRESHOLD = 85
    DEFAULT_RECOGNITION_MODEL = "Default"
    
    # UI settings
    SIDEBAR_WIDTH = 250
    TOPBAR_HEIGHT = 60
    CARD_BORDER_RADIUS = 8
    BUTTON_MIN_HEIGHT = 35
    
    # Animation settings
    ANIMATION_DURATION = 300  # milliseconds
    
    # Update intervals
    DATETIME_UPDATE_INTERVAL = 1000  # milliseconds
    STATS_UPDATE_INTERVAL = 5000  # milliseconds
    CAMERA_UPDATE_INTERVAL = 100  # milliseconds
    
    # Grid settings
    CAMERA_GRID_COLS = 2
    CAMERA_GRID_ROWS = 2
    
    # Database mock data
    MOCK_CAMERAS = [
        {"name": "Gate A - Entrance", "status": "Active", "fps": 30},
        {"name": "Gate B - Exit", "status": "Active", "fps": 30},
        {"name": "Lobby - Main Hall", "status": "Active", "fps": 30},
        {"name": "Parking - Level 1", "status": "Inactive", "fps": 0},
    ]
    
    MOCK_PERSONS = [
        {"name": "John Doe", "id": "P001", "confidence": 98, "camera": "Camera 1"},
        {"name": "Sarah Smith", "id": "P002", "confidence": 96, "camera": "Camera 2"},
        {"name": "Mike Johnson", "id": "P003", "confidence": 94, "camera": "Camera 1"},
    ]
    
    # Log levels
    LOG_LEVEL = "INFO"
    
    # Performance settings
    MAX_TABLE_ROWS = 100
    TABLE_PAGINATION_SIZE = 10
    SEARCH_RESULTS_PER_PAGE = 6


class CameraConfig:
    """Camera-specific configuration"""
    DEFAULT_FPS = 30
    MIN_FPS = 1
    MAX_FPS = 60
    
    RESOLUTIONS = ["480p", "720p", "1080p", "2K"]
    CODECS = ["H.264", "H.265", "VP9"]
    
    DEFAULT_BRIGHTNESS = 100
    DEFAULT_ZOOM = 100


class RecognitionConfig:
    """Face recognition configuration"""
    MIN_CONFIDENCE = 50
    MAX_CONFIDENCE = 99
    DEFAULT_CONFIDENCE = 85
    
    MODELS = ["Default", "Optimized", "High Accuracy"]
    DEFAULT_MODEL = "Default"


class NotificationConfig:
    """Notification configuration"""
    NOTIFICATION_DURATION = 5000  # milliseconds
    NOTIFICATION_MAX_DISPLAY = 5  # Maximum notifications shown at once
    
    SOUND_ENABLED = True
    DESKTOP_NOTIFICATIONS = True
    EMAIL_ALERTS = False


# Export configuration
config = ApplicationConfig()
