"""
Top bar widget for application controls
"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame
)
from PySide6.QtCore import Qt, Signal, QDateTime
from PySide6.QtGui import QFont, QIcon
from datetime import datetime


class TopBarWidget(QFrame):
    """Professional top bar with controls and information"""
    
    theme_toggled = Signal()
    search_triggered = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topbar")
        self.setFrameStyle(QFrame.NoFrame)
        self.setFixedHeight(60)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the top bar UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(15)
        
        # Application title
        title = QLabel("Face Recognition Surveillance System")
        title_font = QFont("Arial", 13, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Search bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.setMaximumWidth(300)
        self.search_input.setMinimumHeight(35)
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #1a1f3a;
                border: 1px solid #2a3452;
                border-radius: 4px;
                padding: 8px 12px;
                color: #e0e0e0;
            }
            QLineEdit:focus {
                border: 1px solid #00bfff;
            }
        """)
        self.search_input.returnPressed.connect(
            lambda: self.search_triggered.emit(self.search_input.text())
        )
        layout.addWidget(self.search_input)
        
        # Current date/time
        self.datetime_label = QLabel()
        self.datetime_label.setMinimumWidth(200)
        self.datetime_label.setAlignment(Qt.AlignRight)
        self.datetime_label.setStyleSheet("color: #a0a0a0; font-size: 12px;")
        self.update_datetime()
        layout.addWidget(self.datetime_label)
        
        # Notification button
        notif_btn = QPushButton("🔔")
        notif_btn.setMaximumWidth(45)
        notif_btn.setMinimumHeight(35)
        notif_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #1e2844;
                border-radius: 4px;
            }
        """)
        layout.addWidget(notif_btn)
        
        # Theme toggle button
        theme_btn = QPushButton("🌙")
        theme_btn.setMaximumWidth(45)
        theme_btn.setMinimumHeight(35)
        theme_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #1e2844;
                border-radius: 4px;
            }
        """)
        theme_btn.clicked.connect(self.theme_toggled.emit)
        layout.addWidget(theme_btn)
        
        # User profile button
        user_btn = QPushButton("👤 Admin")
        user_btn.setMinimumHeight(35)
        user_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #2a3452;
                border-radius: 4px;
                padding: 6px 12px;
                color: #a0a0a0;
            }
            QPushButton:hover {
                background-color: #1e2844;
                color: #00bfff;
            }
        """)
        layout.addWidget(user_btn)
        
        self.setLayout(layout)
    
    def update_datetime(self):
        """Update the date/time display"""
        now = datetime.now()
        date_str = now.strftime("%a, %b %d, %Y")
        time_str = now.strftime("%H:%M:%S")
        self.datetime_label.setText(f"{date_str} | {time_str}")
