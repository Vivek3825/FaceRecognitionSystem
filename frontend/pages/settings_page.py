"""
Settings page - application settings and preferences
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QLineEdit, QGroupBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import QTimer, Qt
from typing import List

class SettingsPage(QWidget):
    """Dynamic page for application settings"""
    
    _default_camera_ids: List[int] = [0] 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize a clean, minimal settings UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)
        
        # Title
        title = QLabel("System Settings")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        main_layout.addWidget(title)
        
        # --- NEW: Active Configuration Display Layout ---
        display_group = QGroupBox("Active Camera List")
        display_group.setStyleSheet("""
            QGroupBox {
                color: #a0a0a0;
                font-weight: bold;
                border: 1px solid #1e2233;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 20px;
            }
        """)
        display_layout = QVBoxLayout()
        
        self.active_list_label = QLabel(self._get_formatted_id_string())
        self.active_list_label.setAlignment(Qt.AlignCenter)
        self.active_list_label.setStyleSheet("""
            color: #00bfff; 
            font-size: 18px; 
            font-weight: bold;
            padding: 10px;
            background-color: #0a0e27;
            border-radius: 4px;
        """)
        display_layout.addWidget(self.active_list_label)
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)
        
        # --- Dynamic Camera Settings Group ---
        cam_group = QGroupBox("Update Configuration")
        cam_group.setStyleSheet("""
            QGroupBox {
                color: #a0a0a0;
                font-weight: bold;
                border: 1px solid #1e2233;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 20px;
            }
        """)
        
        cam_layout = QVBoxLayout()
        cam_layout.setSpacing(10)
        
        instruction_label = QLabel("Default Camera IDs (Comma-separated, e.g., 0, 1, 2):")
        instruction_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        cam_layout.addWidget(instruction_label)
        
        # Input field for the comma-separated list
        self.cam_input = QLineEdit()
        self.cam_input.setText(", ".join(map(str, SettingsPage._default_camera_ids)))
        self.cam_input.setPlaceholderText("Enter IDs like: 0, 1")
        self.cam_input.setStyleSheet("""
            QLineEdit {
                background-color: #0a0e27;
                color: #ffffff;
                border: 1px solid #2a3452;
                border-radius: 4px;
                padding: 10px;
                font-size: 16px;
            }
        """)
        cam_layout.addWidget(self.cam_input)
        
        cam_group.setLayout(cam_layout)
        main_layout.addWidget(cam_group)
        
        # --- Action Buttons & Feedback ---
        actions_layout = QHBoxLayout()
        
        # NEW: Status label for transient success messages
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
        actions_layout.addWidget(self.status_label)
        
        actions_layout.addStretch()
        
        save_btn = QPushButton("🔄 Update Camera List")
        save_btn.setMinimumHeight(45)
        save_btn.setMinimumWidth(150)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
                padding: 0px 15px;
            }
            QPushButton:hover { background-color: #66bb6a; }
            QPushButton:pressed { background-color: #388e3c; }
        """)
        save_btn.clicked.connect(self._process_and_save)
        actions_layout.addWidget(save_btn)
        
        main_layout.addLayout(actions_layout)
        main_layout.addStretch()
        
        self.setLayout(main_layout)

    def _get_formatted_id_string(self) -> str:
        """Helper to format the list for the display label"""
        if not SettingsPage._default_camera_ids:
            return "No Cameras Configured"
        return f"Currently Active IDs: [ {', '.join(map(str, SettingsPage._default_camera_ids))} ]"

    def _process_and_save(self):
        """Process the comma-separated string into a list and store it globally"""
        raw_text = self.cam_input.text()
        try:
            # Parse list: split by comma, strip whitespace, check if it's a number
            parsed_ids = [int(x.strip()) for x in raw_text.split(',') if x.strip().isdigit()]
            
            # Fallback if user entered garbage or left it empty
            if not parsed_ids:
                parsed_ids = [0]
                
            # Sort and save to the CLASS variable
            SettingsPage._default_camera_ids = sorted(list(set(parsed_ids))) 
            
            # Update the text box to show the cleaned up input
            cleaned_str = ", ".join(map(str, SettingsPage._default_camera_ids))
            self.cam_input.setText(cleaned_str)
            
            # NEW: Update the display label above
            self.active_list_label.setText(self._get_formatted_id_string())
            
            # NEW: Show transient success message
            self._show_temporary_message("✅ Settings Saved Successfully!")
            
            print(f"✅ Settings Saved. Default IDs: {SettingsPage._default_camera_ids}")
            
        except Exception as e:
            self._show_temporary_message("⚠️ Error parsing settings", error=True)
            print(f"⚠️ Error parsing settings: {e}")

    def _show_temporary_message(self, message: str, error: bool = False):
        """Displays a message next to the button that fades out after a few seconds"""
        color = "#f44336" if error else "#4caf50"
        self.status_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        self.status_label.setText(message)
        
        # Clear the message after 2500 ms (2.5 seconds)
        QTimer.singleShot(2500, lambda: self.status_label.setText(""))

    @classmethod
    def get_startup_cameras(cls) -> List[int]:
        """Class method to allow any file to grab the settings instantly"""
        return cls._default_camera_ids