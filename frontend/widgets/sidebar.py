"""
Sidebar widget for navigation
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class SidebarWidget(QFrame):
    """Professional navigation sidebar with a unified background"""
    
    page_changed = Signal(str)  # Emits page name when navigation clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFrameStyle(QFrame.NoFrame)
        self.setMinimumWidth(250)
        self.setMaximumWidth(250)
        self.current_button = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the sidebar UI"""
        
        # ── MAIN SIDEBAR STYLING ──
        # This gives the entire sidebar a solid, distinct background color
        # and styles the navigation buttons dynamically.
        self.setStyleSheet("""
            QFrame#sidebar {
                background-color: #121626;  /* Distinct dark background for the whole sidebar */
                border-right: 1px solid #1e2233;
            }
            QPushButton[nav_btn="true"] {
                text-align: left;
                padding-left: 25px;
                color: #a0a0a0;
                background-color: transparent;
                border: none;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton[nav_btn="true"]:hover {
                background-color: #1e2844;
                color: #ffffff;
            }
            QPushButton[nav_btn="true"][active="true"] {
                background-color: #1a223a;
                color: #00bfff;
                border-left: 4px solid #00bfff; /* Professional left highlight */
                padding-left: 21px; /* Adjust padding to offset the 4px border */
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(0)
        
        # ── LOGO/TITLE ──
        logo_layout = QHBoxLayout()
        logo_layout.setContentsMargins(20, 0, 20, 0)
        logo_label = QLabel("FACE RECOGNITION")
        logo_label.setFont(QFont("Arial", 12, QFont.Bold))
        logo_label.setStyleSheet("color: #00bfff; background: transparent;")
        logo_layout.addWidget(logo_label)
        layout.addLayout(logo_layout)
        
        # Separator
        layout.addSpacing(15)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #1e2233; margin: 0px 20px;")
        layout.addWidget(separator)
        layout.addSpacing(20)
        
        # ── NAVIGATION ITEMS ──
        self.nav_items = {
            "dashboard": "Dashboard",
            "camera": "Camera Monitor",
            "registration": "Registration",
            # "search": "Person Search",      
            # "reports": "Reports",
            "settings": "Settings"
        }
        
        self.nav_buttons = {}
        for key, label in self.nav_items.items():
            btn = self.create_nav_button(label, key)
            self.nav_buttons[key] = btn
            layout.addWidget(btn)
        
        # This stretch pushes the logout button to the bottom. 
        # Because the QFrame now has a background color, it will look unified!
        layout.addStretch()
        
        # ── LOGOUT BUTTON ──
        logout_btn = QPushButton("Logout")
        logout_btn.setMinimumHeight(45)
        logout_btn.setCursor(Qt.PointingHandCursor)
        logout_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                margin: 0px 20px; /* Keeps it padded inside the sidebar */
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
        btn.setMinimumHeight(50)
        btn.setCursor(Qt.PointingHandCursor)
        
        # Custom properties for our stylesheet to hook into
        btn.setProperty("nav_btn", "true")
        btn.setProperty("active", "false")
        
        btn.clicked.connect(lambda: self.on_nav_clicked(key))
        return btn
    
    def on_nav_clicked(self, key):
        """Handle navigation button click"""
        self.set_active_button(key)
        self.page_changed.emit(key)
    
    def set_active_button(self, key):
        """Set the active navigation button and update visuals"""
        # Remove active style from previous button
        if self.current_button:
            self.current_button.setProperty("active", "false")
            self.style().unpolish(self.current_button)
            self.style().polish(self.current_button)
        
        # Set active style to new button
        if key in self.nav_buttons:
            btn = self.nav_buttons[key]
            btn.setProperty("active", "true")
            self.current_button = btn
            
            # Reapply stylesheet to reflect the change dynamically
            self.style().unpolish(btn)
            self.style().polish(btn)