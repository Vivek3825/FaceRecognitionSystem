"""
Main application window for Face Recognition System
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QApplication
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from pathlib import Path

from frontend.widgets import SidebarWidget, TopBarWidget
from frontend.pages import (
    DashboardPage, CameraMonitorPage, PersonSearchPage,
    RegistrationPage, ReportsPage, SettingsPage
)
from backend.src.multi_camera_manager import MultiCameraManager

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()

        self.manager = MultiCameraManager()
        self.camera_page = CameraMonitorPage(camera_manager=self.manager)
        
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
        content_layout.addWidget(self.sidebar)
        
        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("QStackedWidget { background-color: #0a0e27; }")
        
        # Create pages
        self.pages = {
            "dashboard": DashboardPage(),
            "camera": self.camera_page,
            "search": PersonSearchPage(),
            "registration": RegistrationPage(),
            "reports": ReportsPage(),
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
        """Apply QSS stylesheet to application"""
        stylesheet_path = Path(__file__).parent / "styles" / "dark_theme.qss"
        
        try:
            with open(stylesheet_path, 'r') as f:
                stylesheet = f.read()
                self.setStyleSheet(stylesheet)
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at {stylesheet_path}")
    
    def on_page_changed(self, page_name):
        """Handle page navigation"""
        self.show_page(page_name)
    
    def show_page(self, page_name):
        """Show a specific page"""
        if page_name in self.pages:
            page_index = list(self.pages.keys()).index(page_name)
            self.stacked_widget.setCurrentIndex(page_index)


def main():
    """Main entry point"""
    app = QApplication([])
    
    window = MainWindow()
    window.show()
    
    exit(app.exec())


if __name__ == "__main__":
    main()
