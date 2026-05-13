"""
Registration page - register new persons
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFrame,
    QComboBox, QTextEdit, QProgressBar, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from frontend.widgets import BaseCard, AlertWidget


class RegistrationPage(QWidget):
    """Page for registering new persons"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize registration UI"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Left section - Capture
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        
        # Title
        title = QLabel("Person Registration")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        left_layout.addWidget(title)
        
        # Capture instructions
        instructions = QLabel("📷 Multi-Angle Face Capture")
        instructions.setFont(QFont("Arial", 14, QFont.Bold))
        instructions.setStyleSheet("color: #e0e0e0;")
        left_layout.addWidget(instructions)
        
        instructions_text = QLabel(
            "Please capture faces from 3 angles:\n"
            "1. Front facing\n"
            "2. Left profile\n"
            "3. Right profile"
        )
        instructions_text.setStyleSheet("color: #a0a0a0;")
        left_layout.addWidget(instructions_text)
        
        # Webcam preview
        preview_frame = QFrame()
        preview_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border: 2px solid #1e2233;
                border-radius: 8px;
            }
        """)
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        preview_label = QLabel("📹 Webcam Feed")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setFont(QFont("Arial", 16))
        preview_label.setStyleSheet("color: #666666;")
        preview_label.setMinimumHeight(300)
        preview_layout.addWidget(preview_label)
        preview_frame.setLayout(preview_layout)
        left_layout.addWidget(preview_frame)
        
        # Capture progress
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(10)
        
        progress_label = QLabel("Capture Progress")
        progress_label.setFont(QFont("Arial", 11, QFont.Bold))
        progress_label.setStyleSheet("color: #e0e0e0;")
        progress_layout.addWidget(progress_label)
        
        # Progress items
        angles = ["Front Face", "Left Profile", "Right Profile"]
        self.angle_checks = {}
        
        for i, angle in enumerate(angles):
            angle_layout = QHBoxLayout()
            angle_layout.setSpacing(10)
            
            checkbox = QLabel("☐")
            checkbox.setFont(QFont("Arial", 16))
            checkbox.setStyleSheet("color: #666666;")
            angle_layout.addWidget(checkbox)
            self.angle_checks[angle] = checkbox
            
            angle_text = QLabel(angle)
            angle_text.setStyleSheet("color: #a0a0a0;")
            angle_layout.addWidget(angle_text)
            
            angle_layout.addStretch()
            progress_layout.addLayout(angle_layout)
        
        left_layout.addLayout(progress_layout)
        
        # Capture buttons
        capture_layout = QHBoxLayout()
        capture_layout.setSpacing(10)
        
        capture_btn = QPushButton("📸 Capture Photo")
        capture_btn.setMinimumHeight(40)
        capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #00bfff;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #00d4ff;
            }
        """)
        capture_layout.addWidget(capture_btn)
        
        retake_btn = QPushButton("🔄 Retake")
        retake_btn.setMinimumHeight(40)
        retake_btn.setMaximumWidth(120)
        capture_layout.addWidget(retake_btn)
        
        left_layout.addLayout(capture_layout)
        left_layout.addStretch()
        
        main_layout.addLayout(left_layout, 1)
        
        # Right section - Form
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        form_title = QLabel("Person Information")
        form_title.setFont(QFont("Arial", 16, QFont.Bold))
        form_title.setStyleSheet("color: #e0e0e0;")
        right_layout.addWidget(form_title)
        
        # Form fields
        form_layout = QGridLayout()
        form_layout.setSpacing(12)
        
        # Full name
        name_label = QLabel("Full Name *")
        name_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        form_layout.addWidget(name_label, 0, 0)
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter full name")
        name_input.setMinimumHeight(35)
        form_layout.addWidget(name_input, 0, 1)
        
        # Person ID
        id_label = QLabel("Person ID *")
        id_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        form_layout.addWidget(id_label, 1, 0)
        
        id_input = QLineEdit()
        id_input.setPlaceholderText("Auto-generated or enter custom")
        id_input.setMinimumHeight(35)
        form_layout.addWidget(id_input, 1, 1)
        
        # Department
        dept_label = QLabel("Department")
        dept_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        form_layout.addWidget(dept_label, 2, 0)
        
        dept_combo = QComboBox()
        dept_combo.addItems(["Operations", "Security", "Administration", "IT", "Other"])
        dept_combo.setMinimumHeight(35)
        form_layout.addWidget(dept_combo, 2, 1)
        
        # Access Level
        access_label = QLabel("Access Level *")
        access_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        form_layout.addWidget(access_label, 3, 0)
        
        access_combo = QComboBox()
        access_combo.addItems(["Level 1 - Basic", "Level 2 - Standard", "Level 3 - Advanced", "Level 4 - Admin"])
        access_combo.setMinimumHeight(35)
        form_layout.addWidget(access_combo, 3, 1)
        
        # Notes
        notes_label = QLabel("Additional Notes")
        notes_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        form_layout.addWidget(notes_label, 4, 0)
        
        notes_text = QTextEdit()
        notes_text.setPlaceholderText("Add any additional information...")
        notes_text.setMinimumHeight(80)
        form_layout.addWidget(notes_text, 4, 1)
        
        right_layout.addLayout(form_layout)
        
        # Images preview
        images_title = QLabel("Captured Images")
        images_title.setFont(QFont("Arial", 13, QFont.Bold))
        images_title.setStyleSheet("color: #e0e0e0;")
        right_layout.addWidget(images_title)
        
        images_layout = QHBoxLayout()
        images_layout.setSpacing(10)
        
        for angle in ["Front", "Left", "Right"]:
            image_card = BaseCard()
            image_layout = QVBoxLayout()
            image_layout.setAlignment(Qt.AlignCenter)
            
            image_label = QLabel("📷")
            image_label.setFont(QFont("Arial", 24))
            image_label.setAlignment(Qt.AlignCenter)
            image_layout.addWidget(image_label)
            
            angle_label = QLabel(angle)
            angle_label.setAlignment(Qt.AlignCenter)
            angle_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
            image_layout.addWidget(angle_label)
            
            image_card.setLayout(image_layout)
            image_card.setMinimumHeight(100)
            images_layout.addWidget(image_card)
        
        right_layout.addLayout(images_layout)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)
        
        register_btn = QPushButton("✓ Register Person")
        register_btn.setMinimumHeight(40)
        register_btn.setStyleSheet("""
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
        actions_layout.addWidget(register_btn)
        
        reset_btn = QPushButton("🔄 Reset Form")
        reset_btn.setMinimumHeight(40)
        reset_btn.setMaximumWidth(150)
        actions_layout.addWidget(reset_btn)
        
        right_layout.addLayout(actions_layout)
        right_layout.addStretch()
        
        main_layout.addLayout(right_layout, 1)
        
        self.setLayout(main_layout)
