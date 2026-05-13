"""
Dashboard page - main overview and statistics
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from frontend.widgets import StatsCard, AlertWidget, BaseCard


class DashboardPage(QWidget):
    """Main dashboard page with statistics and overview"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        # Update stats periodically (mock data simulation)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(5000)
    
    def init_ui(self):
        """Initialize dashboard UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel("Dashboard")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        main_layout.addWidget(title)
        
        # Statistics cards grid
        stats_layout = QGridLayout()
        stats_layout.setSpacing(15)
        
        # Create statistics cards
        stats = [
            ("Active Cameras", 12, "📷", "All systems operational"),
            ("Persons Detected", 247, "👥", "Last 24 hours"),
            ("Watchlist Alerts", 3, "🚨", "Requires attention"),
            ("System Health", 98, "💚", "Excellent")
        ]
        
        self.stats_cards = {}
        for i, (title, value, icon, subtitle) in enumerate(stats):
            card = StatsCard(title, value, subtitle, icon)
            self.stats_cards[title] = (card, value)
            stats_layout.addWidget(card, i // 2, i % 2)
        
        main_layout.addLayout(stats_layout)
        
        # Alerts section
        alerts_title = QLabel("Recent Alerts")
        alerts_title.setFont(QFont("Arial", 16, QFont.Bold))
        alerts_title.setStyleSheet("color: #e0e0e0;")
        main_layout.addWidget(alerts_title)
        
        # Alert cards
        alerts_layout = QVBoxLayout()
        alerts_layout.setSpacing(10)
        
        alerts_data = [
            ("Security Alert", "Unknown person detected at Gate A", "warning"),
            ("Detection Complete", "Face recognition scan completed successfully", "success"),
            ("Camera Status", "Camera 5 requires maintenance", "info"),
        ]
        
        for title, msg, severity in alerts_data:
            alert = AlertWidget(title, msg, severity)
            alerts_layout.addWidget(alert)
        
        alerts_layout.addStretch()
        main_layout.addLayout(alerts_layout)
        
        # Activity timeline section
        timeline_title = QLabel("Recent Detections")
        timeline_title.setFont(QFont("Arial", 16, QFont.Bold))
        timeline_title.setStyleSheet("color: #e0e0e0;")
        main_layout.addWidget(timeline_title)
        
        # Timeline cards
        timeline_layout = QVBoxLayout()
        timeline_layout.setSpacing(8)
        
        detections = [
            ("John Doe", "Camera 1", "10:45 AM", "Matched"),
            ("Sarah Smith", "Camera 3", "10:42 AM", "Matched"),
            ("Unknown Person", "Camera 2", "10:38 AM", "Processing"),
        ]
        
        for person, camera, time, status in detections:
            timeline_card = self.create_timeline_item(person, camera, time, status)
            timeline_layout.addWidget(timeline_card)
        
        timeline_layout.addStretch()
        main_layout.addLayout(timeline_layout)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
    
    def create_timeline_item(self, person, camera, time, status):
        """Create a timeline item card"""
        card = BaseCard()
        card.setMinimumHeight(50)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(15)
        
        # Avatar
        avatar = QLabel("👤")
        avatar.setFont(QFont("Arial", 16))
        avatar.setMaximumWidth(40)
        layout.addWidget(avatar)
        
        # Person info
        info_layout = QVBoxLayout()
        
        person_label = QLabel(person)
        person_label.setFont(QFont("Arial", 11, QFont.Bold))
        person_label.setStyleSheet("color: #e0e0e0;")
        info_layout.addWidget(person_label)
        
        details_label = QLabel(f"{camera} • {time}")
        details_label.setFont(QFont("Arial", 9))
        details_label.setStyleSheet("color: #a0a0a0;")
        info_layout.addWidget(details_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Status badge
        status_colors = {
            "Matched": "#4caf50",
            "Processing": "#ffb300",
            "Error": "#ff4444"
        }
        status_color = status_colors.get(status, "#00bfff")
        
        status_label = QLabel(f"● {status}")
        status_label.setFont(QFont("Arial", 10, QFont.Bold))
        status_label.setStyleSheet(f"color: {status_color};")
        layout.addWidget(status_label)
        
        card.setLayout(layout)
        return card
    
    def update_stats(self):
        """Update statistics (mock data)"""
        import random
        
        # Simulate changing values
        for title, (card, base_value) in self.stats_cards.items():
            variation = random.randint(-2, 5)
            new_value = max(0, base_value + variation)
            
            # This would need to be modified in StatsCard to support value updates
            # For now, we'll just keep the base implementation
