"""
Main application window for Face Recognition System
"""

import sys
from pathlib import Path
import shutil

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QApplication, QDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon

from frontend.pages.login_page import LoginPage
from frontend.widgets import SidebarWidget, TopBarWidget
from frontend.pages import (
    DashboardPage, CameraMonitorPage, RegistrationPage,
    SettingsPage 
)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()

        self.logout_var = False 

        base_dir = Path(__file__).parent.parent
        dataset_dir = base_dir / "backend" / "dataset"
        zip_path = base_dir / "backend" / "dummy_dataset.zip"

        # Check if dataset exists, if not, try to extract
        if not dataset_dir.is_dir():
            print(f"Dataset not found. Attempting to extract from {zip_path}...")
            try:
                if zip_path.exists():
                    shutil.unpack_archive(str(zip_path), str(dataset_dir))
                    print("Dataset extracted successfully!")
                else:
                    print(f"Warning: ZIP file not found at {zip_path}. App may not function correctly.")
            except Exception as e:
                print(f"Error extracting dataset: {e}")

        
        self.setWindowTitle("Face Recognition Surveillance System")
        self.setGeometry(100, 100, 1920, 1080)
        self.setMinimumSize(1400, 800)
        
        # Load stylesheet
        self.apply_stylesheet()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar
        self.topbar = TopBarWidget()
        main_layout.addWidget(self.topbar)
        
        # Content area with sidebar
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.page_changed.connect(self.on_page_changed)
        
        self.sidebar.logout_btn.clicked.connect(self.handle_logout)
        
        content_layout.addWidget(self.sidebar)
        
        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("QStackedWidget { background-color: #0a0e27; }")
        
        # Create pages
        self.pages = {
            "dashboard": DashboardPage(),
            "camera": CameraMonitorPage(), 
            "registration": RegistrationPage(),
            "settings": SettingsPage(),
        }
        
        # Add pages to stacked widget
        for key, page in self.pages.items():
            self.stacked_widget.addWidget(page)
        
        content_layout.addWidget(self.stacked_widget)
        
        # Add content to main layout
        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)
        
        central_widget.setLayout(main_layout)
        
        # Show dashboard by default
        self.show_page("dashboard")
        
        # Update datetime in topbar
        self.datetime_timer = QTimer()
        self.datetime_timer.timeout.connect(self.topbar.update_datetime)
        self.datetime_timer.start(1000)
    
    def apply_stylesheet(self):
        stylesheet_path = Path(__file__).parent / "styles" / "dark_theme.qss"
        try:
            with open(stylesheet_path, 'r') as f:
                stylesheet = f.read()
                self.setStyleSheet(stylesheet)
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at {stylesheet_path}")
    
    def on_page_changed(self, page_name):
        self.show_page(page_name)
    
    def show_page(self, page_name):
        if page_name in self.pages:
            page_index = list(self.pages.keys()).index(page_name)
            self.stacked_widget.setCurrentIndex(page_index)

    def handle_logout(self):
        # Optional: Add a confirmation pop-up here if you want!
        self.logout_var = True  
        self.close()  


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    logout_var = True  

    while logout_var:

        login = LoginPage()
        
        if login.exec() == QDialog.Accepted:
            logout_var = False # User logged in, turn off the flag for now
            
            window = MainWindow() 
            window.show()
            
            app.exec()  # Notice sys.exit() is gone! It pauses here until window closes.
            
            if window.logout_var == True:
                logout_var = True # They clicked logout! Set to True so the loop restarts
            else:
                break 
                
        else:
            # User clicked 'X' on the Login Screen
            break 

    sys.exit(0) # This cleanly kills Python once the loop is broken.

if __name__ == "__main__":
    main()