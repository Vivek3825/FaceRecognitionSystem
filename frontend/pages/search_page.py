"""
Person search page - find and identify persons
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFrame,
    QComboBox, QDateEdit, QCheckBox, QScrollArea
)
from PySide6.QtCore import Qt, QDate, QMimeData
from PySide6.QtGui import QFont, QDrag, QPixmap
from frontend.widgets import PersonCard, BaseCard, AlertWidget


class PersonSearchPage(QWidget):
    """Page for searching and identifying persons"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize person search UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel("Person Search")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        main_layout.addWidget(title)
        
        # Search section
        search_frame = BaseCard()
        search_layout = QVBoxLayout()
        search_layout.setSpacing(15)
        
        # Search mode tabs
        search_mode_layout = QHBoxLayout()
        
        mode_buttons = ["By Name", "By ID", "By Image", "By Camera"]
        self.mode_buttons = {}
        
        for mode in mode_buttons:
            btn = QPushButton(mode)
            btn.setMinimumHeight(35)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e2844;
                    border: none;
                    border-bottom: 3px solid transparent;
                    color: #a0a0a0;
                    font-weight: bold;
                }
                QPushButton:checked {
                    border-bottom: 3px solid #00bfff;
                    color: #00bfff;
                }
                QPushButton:hover {
                    background-color: #2a3452;
                }
            """)
            self.mode_buttons[mode] = btn
            search_mode_layout.addWidget(btn)
        
        self.mode_buttons["By Name"].setChecked(True)
        main_layout.addLayout(search_mode_layout)
        
        # Search input
        search_input_layout = QHBoxLayout()
        search_input_layout.setSpacing(10)
        
        # Name search
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter person name...")
        name_input.setMinimumHeight(40)
        search_input_layout.addWidget(name_input)
        
        # Search button
        search_btn = QPushButton("🔍 Search")
        search_btn.setMinimumHeight(40)
        search_btn.setMaximumWidth(120)
        search_btn.setStyleSheet("""
            QPushButton {
                background-color: #00bfff;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00d4ff;
            }
        """)
        search_input_layout.addWidget(search_btn)
        
        main_layout.addLayout(search_input_layout)
        
        # Image upload area (drag and drop)
        upload_frame = BaseCard()
        upload_layout = QVBoxLayout()
        upload_layout.setAlignment(Qt.AlignCenter)
        
        upload_icon = QLabel("📤")
        upload_icon.setFont(QFont("Arial", 40))
        upload_icon.setAlignment(Qt.AlignCenter)
        upload_layout.addWidget(upload_icon)
        
        upload_text = QLabel("Drag and drop an image here or click to upload")
        upload_text.setAlignment(Qt.AlignCenter)
        upload_text.setStyleSheet("color: #a0a0a0;")
        upload_layout.addWidget(upload_text)
        
        upload_subtext = QLabel("Supported formats: JPG, PNG, BMP")
        upload_subtext.setAlignment(Qt.AlignCenter)
        upload_subtext.setFont(QFont("Arial", 9))
        upload_subtext.setStyleSheet("color: #666666;")
        upload_layout.addWidget(upload_subtext)
        
        upload_btn = QPushButton("Choose Image")
        upload_btn.setMaximumWidth(150)
        upload_btn.setMinimumHeight(35)
        upload_layout.addWidget(upload_btn)
        
        upload_frame.setMinimumHeight(200)
        upload_frame.setLayout(upload_layout)
        main_layout.addWidget(upload_frame)
        
        # Advanced filters
        filters_title = QLabel("Filters")
        filters_title.setFont(QFont("Arial", 14, QFont.Bold))
        filters_title.setStyleSheet("color: #e0e0e0;")
        main_layout.addWidget(filters_title)
        
        filters_layout = QGridLayout()
        filters_layout.setSpacing(15)
        
        # Camera filter
        camera_label = QLabel("Camera:")
        camera_label.setStyleSheet("color: #a0a0a0;")
        filters_layout.addWidget(camera_label, 0, 0)
        
        camera_combo = QComboBox()
        camera_combo.addItems(["All Cameras", "Camera 1", "Camera 2", "Camera 3"])
        filters_layout.addWidget(camera_combo, 0, 1)
        
        # Date filter
        date_label = QLabel("Date Range:")
        date_label.setStyleSheet("color: #a0a0a0;")
        filters_layout.addWidget(date_label, 0, 2)
        
        date_edit = QDateEdit()
        date_edit.setDate(QDate.currentDate())
        filters_layout.addWidget(date_edit, 0, 3)
        
        # Confidence threshold
        confidence_label = QLabel("Min Confidence:")
        confidence_label.setStyleSheet("color: #a0a0a0;")
        filters_layout.addWidget(confidence_label, 1, 0)
        
        confidence_combo = QComboBox()
        confidence_combo.addItems(["All", "90%+", "95%+", "99%+"])
        filters_layout.addWidget(confidence_combo, 1, 1)

        filters_layout.setColumnStretch(4, 1)
        #filters_layout.addStretch(1, 4)
        main_layout.addLayout(filters_layout)
        
        # Results section
        results_title = QLabel("Search Results")
        results_title.setFont(QFont("Arial", 16, QFont.Bold))
        results_title.setStyleSheet("color: #e0e0e0;")
        main_layout.addWidget(results_title)
        
        # Results grid
        results_layout = QGridLayout()
        results_layout.setSpacing(15)
        
        # Mock search results
        results_data = [
            ("John Doe", "P001", 98, "Camera 1", "2 min ago"),
            ("Sarah Smith", "P002", 96, "Camera 2", "5 min ago"),
            ("Mike Johnson", "P003", 94, "Camera 1", "10 min ago"),
            ("Emily Brown", "P004", 92, "Camera 3", "15 min ago"),
            ("Alex Wilson", "P005", 89, "Camera 2", "20 min ago"),
            ("Lisa Anderson", "P006", 87, "Camera 1", "25 min ago"),
        ]
        
        for i, (name, pid, conf, camera, time) in enumerate(results_data):
            card = PersonCard(name, pid, conf, camera)
            results_layout.addWidget(card, i // 3, i % 3)
        
        # Add empty cards for visual balance
        for i in range(len(results_data), 6):
            empty_card = BaseCard()
            empty_card.setMinimumHeight(180)
            results_layout.addWidget(empty_card, i // 3, i % 3)
        
        main_layout.addLayout(results_layout)
        
        # Pagination
        pagination_layout = QHBoxLayout()
        pagination_layout.addStretch()
        
        prev_btn = QPushButton("← Previous")
        prev_btn.setMaximumWidth(100)
        pagination_layout.addWidget(prev_btn)
        
        page_label = QLabel("Page 1 of 10")
        page_label.setStyleSheet("color: #a0a0a0;")
        pagination_layout.addWidget(page_label)
        
        next_btn = QPushButton("Next →")
        next_btn.setMaximumWidth(100)
        pagination_layout.addWidget(next_btn)
        
        main_layout.addLayout(pagination_layout)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
