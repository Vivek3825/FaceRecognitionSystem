"""
Sidebar widget for navigation
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QIcon


class SidebarWidget(QFrame):
    """Professional navigation sidebar with collapsible support"""
    
    page_changed = Signal(str)  # Emits page name when navigation clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFrameStyle(QFrame.NoFrame)
        self.setMinimumWidth(250)
        self.setMaximumWidth(250)
        self.current_button = None

        
        self.init_ui()
        #self.current_button = None
        
    def init_ui(self):
        """Initialize the sidebar UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(0)
        
        # Logo/Title
        logo_layout = QHBoxLayout()
        logo_layout.setContentsMargins(15, 0, 15, 0)
        logo_label = QLabel("FACE RECOGNITION")
        logo_label.setFont(QFont("Arial", 12, QFont.Bold))
        logo_label.setStyleSheet("color: #00bfff;")
        logo_layout.addWidget(logo_label)
        layout.addLayout(logo_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #1e2233;")
        layout.addWidget(separator)
        layout.addSpacing(20)
        
        # Navigation items
        self.nav_items = {
            "dashboard": "Dashboard",
            "camera": "Camera Monitor",
            "search": "Person Search",
            "registration": "Registration",
            "reports": "Reports",
            "settings": "Settings"
        }
        
        self.nav_buttons = {}
        for key, label in self.nav_items.items():
            btn = self.create_nav_button(label, key)
            self.nav_buttons[key] = btn
            layout.addWidget(btn)
        
        layout.addStretch()
        
        # Logout button
        logout_btn = QPushButton("Logout")
        logout_btn.setMinimumHeight(40)
        logout_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #ff5555;
            }
        """)
        layout.addWidget(logout_btn)
        
        self.setLayout(layout)
        
        # Set first button as active
        self.set_active_button("dashboard")
        
    def create_nav_button(self, label, key):
        """Create a navigation button"""
        btn = QPushButton(label)
        btn.setMinimumHeight(45)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: self.on_nav_clicked(key))
        return btn
    
    def on_nav_clicked(self, key):
        """Handle navigation button click"""
        self.set_active_button(key)
        self.page_changed.emit(key)
    
    def set_active_button(self, key):
        """Set the active navigation button"""
        # Remove active style from previous button
        if self.current_button:
            self.current_button.setProperty("active", False)
        
        # Set active style to new button
        if key in self.nav_buttons:
            btn = self.nav_buttons[key]
            btn.setProperty("active", True)
            self.current_button = btn
            # Reapply stylesheet to reflect the change
            self.style().unpolish(btn)
            self.style().polish(btn)
