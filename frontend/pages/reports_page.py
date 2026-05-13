"""
Reports page - view logs, statistics, and export data
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QFrame, QComboBox, QDateEdit, QCheckBox, QHeaderView
)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QFont, QColor, QBrush
from frontend.widgets import BaseCard, StatsCard


class ReportsPage(QWidget):
    """Page for viewing reports and logs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize reports UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel("Reports & Analytics")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        main_layout.addWidget(title)
        
        # Summary cards
        summary_layout = QGridLayout()
        summary_layout.setSpacing(15)
        
        summary_stats = [
            ("Total Detections", 2847, "👁️", "Last 7 days"),
            ("Unique Persons", 156, "👥", "In database"),
            ("Alert Triggers", 42, "🚨", "This week"),
            ("System Uptime", 99.8, "⚡", "Percentage"),
        ]
        
        for i, (title_text, value, icon, subtitle) in enumerate(summary_stats):
            card = StatsCard(title_text, value, subtitle, icon)
            summary_layout.addWidget(card, i // 2, i % 2)
        
        main_layout.addLayout(summary_layout)
        
        # Filters section
        filters_title = QLabel("Filters")
        filters_title.setFont(QFont("Arial", 14, QFont.Bold))
        filters_title.setStyleSheet("color: #e0e0e0;")
        main_layout.addWidget(filters_title)
        
        filters_layout = QHBoxLayout()
        filters_layout.setSpacing(15)
        
        # Date range
        from_label = QLabel("From:")
        from_label.setStyleSheet("color: #a0a0a0;")
        filters_layout.addWidget(from_label)
        
        from_date = QDateEdit()
        from_date.setDate(QDate.currentDate().addDays(-7))
        filters_layout.addWidget(from_date)
        
        to_label = QLabel("To:")
        to_label.setStyleSheet("color: #a0a0a0;")
        filters_layout.addWidget(to_label)
        
        to_date = QDateEdit()
        to_date.setDate(QDate.currentDate())
        filters_layout.addWidget(to_date)
        
        # Report type
        report_label = QLabel("Report Type:")
        report_label.setStyleSheet("color: #a0a0a0;")
        filters_layout.addWidget(report_label)
        
        report_combo = QComboBox()
        report_combo.addItems(["All Events", "Detections", "Alerts", "System Events"])
        filters_layout.addWidget(report_combo)
        
        # Camera filter
        camera_label = QLabel("Camera:")
        camera_label.setStyleSheet("color: #a0a0a0;")
        filters_layout.addWidget(camera_label)
        
        camera_combo = QComboBox()
        camera_combo.addItems(["All Cameras", "Camera 1", "Camera 2", "Camera 3"])
        filters_layout.addWidget(camera_combo)
        
        filters_layout.addStretch()
        
        # Export button
        export_btn = QPushButton("📊 Export")
        export_btn.setMaximumWidth(100)
        export_btn.setMinimumHeight(35)
        filters_layout.addWidget(export_btn)
        
        main_layout.addLayout(filters_layout)
        
        # Activity logs table
        logs_title = QLabel("Activity Logs")
        logs_title.setFont(QFont("Arial", 14, QFont.Bold))
        logs_title.setStyleSheet("color: #e0e0e0;")
        main_layout.addWidget(logs_title)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Event Type", "Person", "Camera", "Status", "Details"
        ])
        self.table.setMinimumHeight(300)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0a0e27;
                color: #e0e0e0;
                gridline-color: #1e2233;
                border: 1px solid #1e2233;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
                background-color: #0a0e27;
            }
            QTableWidget::item:selected {
                background-color: #00bfff;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #141829;
                color: #e0e0e0;
                padding: 8px;
                border: none;
                border-right: 1px solid #1e2233;
                font-weight: bold;
            }
        """)
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        
        # Mock data
        mock_events = [
            ("2024-05-11 10:45:23", "Face Detected", "John Doe", "Camera 1", "✓ Matched", "ID: P001"),
            ("2024-05-11 10:42:15", "Face Detected", "Unknown", "Camera 2", "⏳ Processing", "New face"),
            ("2024-05-11 10:38:42", "Alert", "Unknown", "Camera 3", "🚨 Alert", "Watchlist match"),
            ("2024-05-11 10:35:10", "Face Detected", "Sarah Smith", "Camera 1", "✓ Matched", "ID: P002"),
            ("2024-05-11 10:30:55", "System", "N/A", "System", "✓ OK", "Camera sync completed"),
            ("2024-05-11 10:28:30", "Face Detected", "Mike Johnson", "Camera 2", "✓ Matched", "ID: P003"),
            ("2024-05-11 10:25:12", "Error", "N/A", "Camera 4", "❌ Error", "Connection lost"),
            ("2024-05-11 10:20:45", "Face Detected", "Emily Brown", "Camera 3", "✓ Matched", "ID: P004"),
        ]
        
        self.table.setRowCount(len(mock_events))
        
        for row, (timestamp, event_type, person, camera, status, details) in enumerate(mock_events):
            # Timestamp
            self.table.setItem(row, 0, QTableWidgetItem(timestamp))
            
            # Event type
            event_item = QTableWidgetItem(event_type)
            if event_type == "Alert":
                event_item.setForeground(QBrush(QColor("#ffb300")))
            elif event_type == "Error":
                event_item.setForeground(QBrush(QColor("#ff4444")))
            self.table.setItem(row, 1, event_item)
            
            # Person
            self.table.setItem(row, 2, QTableWidgetItem(person))
            
            # Camera
            self.table.setItem(row, 3, QTableWidgetItem(camera))
            
            # Status
            status_item = QTableWidgetItem(status)
            if "Matched" in status:
                status_item.setForeground(QBrush(QColor("#4caf50")))
            elif "Processing" in status:
                status_item.setForeground(QBrush(QColor("#ffb300")))
            elif "Alert" in status:
                status_item.setForeground(QBrush(QColor("#ffb300")))
            elif "Error" in status:
                status_item.setForeground(QBrush(QColor("#ff4444")))
            self.table.setItem(row, 4, status_item)
            
            # Details
            self.table.setItem(row, 5, QTableWidgetItem(details))
        
        main_layout.addWidget(self.table)
        
        # Pagination
        pagination_layout = QHBoxLayout()
        pagination_layout.addStretch()
        
        prev_btn = QPushButton("← Previous")
        prev_btn.setMaximumWidth(100)
        pagination_layout.addWidget(prev_btn)
        
        page_label = QLabel("Page 1 of 50")
        page_label.setStyleSheet("color: #a0a0a0;")
        pagination_layout.addWidget(page_label)
        
        next_btn = QPushButton("Next →")
        next_btn.setMaximumWidth(100)
        pagination_layout.addWidget(next_btn)
        
        main_layout.addLayout(pagination_layout)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
