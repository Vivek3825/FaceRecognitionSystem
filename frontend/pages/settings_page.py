"""
Settings page - application settings and preferences
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QCheckBox,
    QSpinBox, QComboBox, QFrame, QTabWidget, QGroupBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from frontend.widgets import BaseCard


class SettingsPage(QWidget):
    """Page for application settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize settings UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel("Settings")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        main_layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabBar::tab {
                background-color: #141829;
                color: #a0a0a0;
                padding: 10px 20px;
                border: none;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:hover {
                color: #00bfff;
            }
            QTabBar::tab:selected {
                color: #ffffff;
                border-bottom: 2px solid #00bfff;
                background-color: #0a0e27;
            }
            QTabWidget::pane {
                border: 1px solid #1e2233;
            }
        """)
        
        # System settings tab
        system_tab = self.create_system_settings()
        tabs.addTab(system_tab, "System")
        
        # Camera settings tab
        camera_tab = self.create_camera_settings()
        tabs.addTab(camera_tab, "Cameras")
        
        # Recognition settings tab
        recognition_tab = self.create_recognition_settings()
        tabs.addTab(recognition_tab, "Recognition")
        
        # Notification settings tab
        notification_tab = self.create_notification_settings()
        tabs.addTab(notification_tab, "Notifications")
        
        # Database settings tab
        database_tab = self.create_database_settings()
        tabs.addTab(database_tab, "Database")
        
        main_layout.addWidget(tabs)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.addStretch()
        
        save_btn = QPushButton("💾 Save Settings")
        save_btn.setMinimumHeight(40)
        save_btn.setMinimumWidth(150)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #66bb6a;
            }
        """)
        actions_layout.addWidget(save_btn)
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setMinimumHeight(40)
        reset_btn.setMinimumWidth(150)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e2844;
                color: #a0a0a0;
                border: 1px solid #2a3452;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2a3452;
                color: #00bfff;
            }
        """)
        actions_layout.addWidget(reset_btn)
        
        close_btn = QPushButton("✕ Close")
        close_btn.setMinimumHeight(40)
        close_btn.setMaximumWidth(100)
        actions_layout.addWidget(close_btn)
        
        main_layout.addLayout(actions_layout)
        
        self.setLayout(main_layout)
    
    def create_system_settings(self):
        """Create system settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Application settings
        app_group = QGroupBox("Application")
        app_layout = QGridLayout()
        app_layout.setSpacing(15)
        
        # Theme
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet("color: #a0a0a0;")
        app_layout.addWidget(theme_label, 0, 0)
        
        theme_combo = QComboBox()
        theme_combo.addItems(["Dark", "Light", "Auto"])
        theme_combo.setCurrentText("Dark")
        app_layout.addWidget(theme_combo, 0, 1)
        
        # Language
        lang_label = QLabel("Language:")
        lang_label.setStyleSheet("color: #a0a0a0;")
        app_layout.addWidget(lang_label, 1, 0)
        
        lang_combo = QComboBox()
        lang_combo.addItems(["English", "Spanish", "French", "German"])
        lang_combo.setCurrentText("English")
        app_layout.addWidget(lang_combo, 1, 1)
        
        # Auto-start
        autostart_check = QCheckBox("Start application on system startup")
        autostart_check.setStyleSheet("color: #a0a0a0;")
        autostart_check.setChecked(True)
        app_layout.addWidget(autostart_check, 2, 0, 1, 2)
        
        # Minimize to tray
        tray_check = QCheckBox("Minimize to system tray")
        tray_check.setStyleSheet("color: #a0a0a0;")
        tray_check.setChecked(True)
        app_layout.addWidget(tray_check, 3, 0, 1, 2)
        
        app_group.setLayout(app_layout)
        layout.addWidget(app_group)
        
        # System resources
        resources_group = QGroupBox("System Resources")
        resources_layout = QGridLayout()
        resources_layout.setSpacing(15)
        
        # Max CPU usage
        cpu_label = QLabel("Max CPU Usage (%):")
        cpu_label.setStyleSheet("color: #a0a0a0;")
        resources_layout.addWidget(cpu_label, 0, 0)
        
        cpu_spin = QSpinBox()
        cpu_spin.setRange(10, 100)
        cpu_spin.setValue(80)
        cpu_spin.setMinimumHeight(30)
        resources_layout.addWidget(cpu_spin, 0, 1)
        
        # Max RAM usage
        ram_label = QLabel("Max RAM Usage (%):")
        ram_label.setStyleSheet("color: #a0a0a0;")
        resources_layout.addWidget(ram_label, 1, 0)
        
        ram_spin = QSpinBox()
        ram_spin.setRange(10, 100)
        ram_spin.setValue(70)
        ram_spin.setMinimumHeight(30)
        resources_layout.addWidget(ram_spin, 1, 1)
        
        resources_group.setLayout(resources_layout)
        layout.addWidget(resources_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_camera_settings(self):
        """Create camera settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        group = QGroupBox("Camera Configuration")
        group_layout = QGridLayout()
        group_layout.setSpacing(15)
        
        # FPS limit
        fps_label = QLabel("FPS Limit:")
        fps_label.setStyleSheet("color: #a0a0a0;")
        group_layout.addWidget(fps_label, 0, 0)
        
        fps_spin = QSpinBox()
        fps_spin.setRange(1, 60)
        fps_spin.setValue(30)
        fps_spin.setMinimumHeight(30)
        group_layout.addWidget(fps_spin, 0, 1)
        
        # Resolution
        res_label = QLabel("Resolution:")
        res_label.setStyleSheet("color: #a0a0a0;")
        group_layout.addWidget(res_label, 1, 0)
        
        res_combo = QComboBox()
        res_combo.addItems(["480p", "720p", "1080p", "2K"])
        res_combo.setCurrentText("1080p")
        group_layout.addWidget(res_combo, 1, 1)
        
        # Encoding
        enc_label = QLabel("Video Codec:")
        enc_label.setStyleSheet("color: #a0a0a0;")
        group_layout.addWidget(enc_label, 2, 0)
        
        enc_combo = QComboBox()
        enc_combo.addItems(["H.264", "H.265", "VP9"])
        enc_combo.setCurrentText("H.264")
        group_layout.addWidget(enc_combo, 2, 1)
        
        # Recording
        recording_check = QCheckBox("Enable continuous recording")
        recording_check.setStyleSheet("color: #a0a0a0;")
        recording_check.setChecked(True)
        group_layout.addWidget(recording_check, 3, 0, 1, 2)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_recognition_settings(self):
        """Create recognition settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        group = QGroupBox("Face Recognition")
        group_layout = QGridLayout()
        group_layout.setSpacing(15)
        
        # Confidence threshold
        conf_label = QLabel("Confidence Threshold:")
        conf_label.setStyleSheet("color: #a0a0a0;")
        group_layout.addWidget(conf_label, 0, 0)
        
        conf_slider = QSlider(Qt.Horizontal)
        conf_slider.setRange(50, 99)
        conf_slider.setValue(85)
        conf_slider.setMinimumHeight(30)
        group_layout.addWidget(conf_slider, 0, 1)
        
        conf_value = QLabel("85%")
        conf_value.setStyleSheet("color: #00bfff;")
        group_layout.addWidget(conf_value, 0, 2)
        
        # Model
        model_label = QLabel("Model:")
        model_label.setStyleSheet("color: #a0a0a0;")
        group_layout.addWidget(model_label, 1, 0)
        
        model_combo = QComboBox()
        model_combo.addItems(["Default", "Optimized", "High Accuracy"])
        model_combo.setCurrentText("Default")
        group_layout.addWidget(model_combo, 1, 1, 1, 2)
        
        # Duplicate detection
        duplicate_check = QCheckBox("Prevent duplicate detections")
        duplicate_check.setStyleSheet("color: #a0a0a0;")
        duplicate_check.setChecked(True)
        group_layout.addWidget(duplicate_check, 2, 0, 1, 3)
        
        # Enable mask detection
        mask_check = QCheckBox("Enable face mask detection")
        mask_check.setStyleSheet("color: #a0a0a0;")
        mask_check.setChecked(False)
        group_layout.addWidget(mask_check, 3, 0, 1, 3)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_notification_settings(self):
        """Create notification settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        group = QGroupBox("Notifications")
        group_layout = QGridLayout()
        group_layout.setSpacing(10)
        
        notifications = [
            ("Desktop notifications", True),
            ("Email alerts", False),
            ("Sound alerts", True),
            ("Watchlist matches", True),
            ("System errors", True),
            ("Low disk space", True),
        ]
        
        for i, (name, checked) in enumerate(notifications):
            check = QCheckBox(name)
            check.setStyleSheet("color: #a0a0a0;")
            check.setChecked(checked)
            group_layout.addWidget(check, i, 0)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_database_settings(self):
        """Create database settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        group = QGroupBox("Database & Storage")
        group_layout = QGridLayout()
        group_layout.setSpacing(15)
        
        # Database location
        db_label = QLabel("Database Location:")
        db_label.setStyleSheet("color: #a0a0a0;")
        group_layout.addWidget(db_label, 0, 0)
        
        db_path_label = QLabel("/data/database/")
        db_path_label.setStyleSheet("color: #666666;")
        group_layout.addWidget(db_path_label, 0, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        group_layout.addWidget(browse_btn, 0, 2)
        
        # Backup
        backup_group = QGroupBox("Backup")
        backup_layout = QGridLayout()
        backup_layout.setSpacing(10)
        
        auto_backup = QCheckBox("Enable automatic backup")
        auto_backup.setStyleSheet("color: #a0a0a0;")
        auto_backup.setChecked(True)
        backup_layout.addWidget(auto_backup, 0, 0, 1, 2)
        
        backup_freq_label = QLabel("Backup Frequency:")
        backup_freq_label.setStyleSheet("color: #a0a0a0;")
        backup_layout.addWidget(backup_freq_label, 1, 0)
        
        freq_combo = QComboBox()
        freq_combo.addItems(["Daily", "Weekly", "Monthly"])
        freq_combo.setCurrentText("Daily")
        backup_layout.addWidget(freq_combo, 1, 1)
        
        backup_btn = QPushButton("🔄 Backup Now")
        backup_btn.setMaximumWidth(120)
        backup_layout.addWidget(backup_btn, 2, 0, 1, 2)
        
        backup_group.setLayout(backup_layout)
        group_layout.addWidget(backup_group, 1, 0, 1, 3)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
