"""
Settings page - application settings and preferences
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QLineEdit, QGroupBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtGui import QFont
from PySide6.QtCore import QTimer, Qt
from typing import List
from pathlib import Path
import configparser

class SettingsPage(QWidget):
    """Dynamic page for application settings"""
    
    _default_camera_ids: List[int] = [0] 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def _get_config_path(self) -> Path:
        """Resolve the path to backend/camera_config.ini"""
        base_path = Path(__file__).resolve().parents[2]
        config_path = base_path / 'backend' / 'camera_config.ini'
        
        if not config_path.parent.exists():
            config_path = Path.cwd() / 'backend' / 'camera_config.ini'
            
        return config_path

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
        
        # --- Active Configuration Display Layout ---
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
        
        # Update IDs Button Layout
        id_actions_layout = QHBoxLayout()
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
        id_actions_layout.addWidget(self.status_label)
        id_actions_layout.addStretch()
        
        save_btn = QPushButton("🔄 Update Camera List")
        save_btn.setMinimumHeight(40)
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
        id_actions_layout.addWidget(save_btn)
        
        cam_layout.addLayout(id_actions_layout)
        cam_group.setLayout(cam_layout)
        main_layout.addWidget(cam_group)

        # --- Camera Names Configuration Group ---
        names_group = QGroupBox("Camera Display Names")
        names_group.setStyleSheet("""
            QGroupBox {
                color: #a0a0a0;
                font-weight: bold;
                border: 1px solid #1e2233;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 20px;
            }
        """)
        names_layout = QVBoxLayout()
        
        # Table Setup
        self.cam_table = QTableWidget(0, 3)
        self.cam_table.setHorizontalHeaderLabels(["Camera ID", "Current Name", "Modify Camera Name"])
        self.cam_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # UI IMPROVEMENT: Increase row height and hide side numbers
        self.cam_table.verticalHeader().setDefaultSectionSize(60) 
        self.cam_table.verticalHeader().setVisible(False)
        
        self.cam_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cam_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.cam_table.setStyleSheet("""
            QTableWidget {
                background-color: #0a0e27;
                color: #ffffff;
                border: 1px solid #2a3452;
                border-radius: 4px;
                gridline-color: #1e2233;
            }
            QHeaderView::section {
                background-color: #1e2233;
                color: white;
                padding: 10px;
                border: 1px solid #2a3452;
                font-weight: bold;
                font-size: 14px;
            }
            QTableWidget::item { padding: 10px; }
        """)
        names_layout.addWidget(self.cam_table)

        # Update Names Button Layout
        name_actions_layout = QHBoxLayout()
        self.name_status_label = QLabel("")
        self.name_status_label.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
        name_actions_layout.addWidget(self.name_status_label)
        name_actions_layout.addStretch()

        update_names_btn = QPushButton("💾 Update/Overwrite Camera Names")
        update_names_btn.setMinimumHeight(45)
        update_names_btn.setMinimumWidth(220)
        update_names_btn.setStyleSheet("""
            QPushButton {
                background-color: #008CBA;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
                padding: 0px 15px;
            }
            QPushButton:hover { background-color: #009dda; }
            QPushButton:pressed { background-color: #007399; }
        """)
        update_names_btn.clicked.connect(self._update_camera_names)
        name_actions_layout.addWidget(update_names_btn)

        names_layout.addLayout(name_actions_layout)
        names_group.setLayout(names_layout)
        main_layout.addWidget(names_group)

        main_layout.addStretch()
        self.setLayout(main_layout)
        
        self._populate_camera_table()

    def _populate_camera_table(self):
        """Fill the table with active camera IDs and their configured names"""
        self.cam_table.setRowCount(0)
        
        config = configparser.ConfigParser()
        config_path = self._get_config_path()
        if config_path.exists():
            config.read(config_path)

        for cam_id in SettingsPage._default_camera_ids:
            row_idx = self.cam_table.rowCount()
            self.cam_table.insertRow(row_idx)
            
            # Column 0: Camera ID
            id_item = QTableWidgetItem(str(cam_id))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.cam_table.setItem(row_idx, 0, id_item)
            
            # Column 1: Current Name
            current_name = f"Default Camera {cam_id}"
            if config.has_section('Display Names') and str(cam_id) in config['Display Names']:
                current_name = config['Display Names'][str(cam_id)]
                
            name_item = QTableWidgetItem(current_name)
            name_item.setTextAlignment(Qt.AlignCenter)
            self.cam_table.setItem(row_idx, 1, name_item)
            
            # Column 2: Modify Input
            input_widget = QLineEdit()
            input_widget.setPlaceholderText("Enter new name...")
            input_widget.setStyleSheet("""
                QLineEdit {
                    background-color: #1e2233;
                    color: white;
                    border: 1px solid #2a3452;
                    border-radius: 4px;
                    padding: 8px;
                    font-size: 14px;
                }
                QLineEdit:focus { border: 1px solid #00bfff; }
            """)
            self.cam_table.setCellWidget(row_idx, 2, input_widget)

    def _update_camera_names(self):
        """Read modified names from the table and COMPLETELY OVERWRITE camera_config.ini"""
        config_path = self._get_config_path()
        config = configparser.ConfigParser()
        
        if config_path.exists():
            config.read(config_path)
            
        # FIX: Completely wipe out the old section to prevent keeping history
        if config.has_section('Display Names'):
            config.remove_section('Display Names')
            
        # Add a fresh section
        config.add_section('Display Names')
        
        # Loop through table rows and rebuild the file
        for row in range(self.cam_table.rowCount()):
            cam_id = self.cam_table.item(row, 0).text()
            current_name = self.cam_table.item(row, 1).text()
            input_widget = self.cam_table.cellWidget(row, 2)
            new_name = input_widget.text().strip()
            
            # If the user left it blank, keep the old name, otherwise use the new one
            final_name = new_name if new_name else current_name
            config['Display Names'][cam_id] = final_name
                
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as configfile:
                config.write(configfile)
            
            self._show_temporary_message("✅ Camera names overwritten!", target_label="name")
            self._populate_camera_table() 
        except Exception as e:
            self._show_temporary_message(f"⚠️ Error saving config: {e}", error=True, target_label="name")

    def _get_formatted_id_string(self) -> str:
        """Helper to format the list for the display label"""
        if not SettingsPage._default_camera_ids:
            return "No Cameras Configured"
        return f"Currently Active IDs: [ {', '.join(map(str, SettingsPage._default_camera_ids))} ]"

    def _process_and_save(self):
        """Process the comma-separated string into a list and store it globally"""
        raw_text = self.cam_input.text()
        try:
            parsed_ids = [int(x.strip()) for x in raw_text.split(',') if x.strip().isdigit()]
            
            if not parsed_ids:
                parsed_ids = [0]
                
            SettingsPage._default_camera_ids = sorted(list(set(parsed_ids))) 
            
            cleaned_str = ", ".join(map(str, SettingsPage._default_camera_ids))
            self.cam_input.setText(cleaned_str)
            self.active_list_label.setText(self._get_formatted_id_string())
            
            self._show_temporary_message("✅ Settings Saved Successfully!", target_label="id")
            
            self._populate_camera_table()
            
        except Exception as e:
            self._show_temporary_message("⚠️ Error parsing settings", error=True, target_label="id")

    def _show_temporary_message(self, message: str, error: bool = False, target_label: str = "id"):
        """Displays a message next to a specific button that fades out after a few seconds"""
        color = "#f44336" if error else "#4caf50"
        label = self.status_label if target_label == "id" else self.name_status_label
        
        label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        label.setText(message)
        
        QTimer.singleShot(2500, lambda: label.setText(""))

    @classmethod
    def get_startup_cameras(cls) -> List[int]:
        """Class method to allow any file to grab the settings instantly"""
        return cls._default_camera_ids