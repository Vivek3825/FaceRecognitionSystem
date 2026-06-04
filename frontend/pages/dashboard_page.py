"""
Dashboard page - main overview and statistics
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame, 
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QColor
from pathlib import Path
import sys

# Add backend to path dynamically
backend_path = Path(__file__).resolve().parents[2] / 'backend' / 'src'
sys.path.insert(0, str(backend_path))

from backend.src.stats_manager import StatsManager
from backend.src.dataset_manager import RemovePerson


class RemovePersonWorker(QThread):
    """Background thread for removing a person to keep UI responsive"""
    worker_finished = Signal(bool, str) 
    
    def __init__(self, person_id, dataset_path):
        super().__init__()
        self.person_id = person_id
        self.dataset_path = dataset_path
    
    def run(self):
        try:
            RemovePerson(personID=self.person_id, dataset_path=self.dataset_path)
            self.worker_finished.emit(True, f"Successfully removed person {self.person_id} and all associated data.")
        except Exception as e:
            self.worker_finished.emit(False, f"Error removing person: {str(e)}")


class DashboardPage(QWidget):
    """Main dashboard page with statistics and management overview"""
    
    def __init__(self, parent=None, dataset_path=None):
        super().__init__(parent)
        self.dataset_path = dataset_path or (Path(__file__).resolve().parents[2] / 'backend' / 'dataset')
        
        self.stats_manager = StatsManager(self.dataset_path)
        self.remove_thread = None
        self.progress_dialog = None
        self.stat_labels = {} 
        
        self.init_ui()
        
        # Auto-refresh stats every 10 seconds
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_stats)
        self.refresh_timer.start(10000)
    
    def init_ui(self):
        # Main Layout (Top to Bottom)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 30)
        main_layout.setSpacing(20)
        
        # Title Header
        title = QLabel("System Statistics & Database Info")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        main_layout.addWidget(title)
        
        stats = self.stats_manager.get_all_statistics()
        
        # ==========================================
        # TOP SECTION: STATS CARDS
        # ==========================================
        stats_grid = QGridLayout()
        stats_grid.setSpacing(20)
        
        # Row 1
        stats_grid.addWidget(self.create_stats_card("Total Persons", stats['person_count'], "👥", "#00bfff"), 0, 0)
        stats_grid.addWidget(self.create_stats_card("Total Embeddings", stats['embeddings_count'], "🧠", "#ff6b6b"), 0, 1)
        stats_grid.addWidget(self.create_stats_card("Unique Embeddings", stats['unique_embeddings'], "🔗", "#4ecdc4"), 0, 2)
        
        # Row 2
        stats_grid.addWidget(self.create_stats_card("Original Images", stats['total_images'], "📷", "#ffd93d"), 1, 0)
        stats_grid.addWidget(self.create_stats_card("Face Images", stats['face_images'], "👤", "#6bcf7f"), 1, 1)
        db_size = stats['database_info'].get('total_size', '0 B')
        stats_grid.addWidget(self.create_stats_card("Database Size", db_size, "💾", "#a78bfa"), 1, 2)
        
        main_layout.addLayout(stats_grid)
        
        # ==========================================
        # BOTTOM SECTION: TABLE
        # ==========================================
        table_section = self.create_persons_section(stats['all_persons'])
        main_layout.addWidget(table_section, stretch=1)
    
    def create_stats_card(self, title, value, icon, color):
        """Create a cleanly formatted statistics card"""
        card = QFrame()
        card.setObjectName("StatCard")
        card.setFixedHeight(120)
        
        card.setStyleSheet("""
            QFrame#StatCard {
                background-color: #141829;
                border: 1px solid #2a2e45;
                border-radius: 12px;
            }
            QFrame#StatCard:hover {
                border: 1px solid #4ecdc4;
                background-color: #1a1f35;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 15, 20, 15)
        
        # Top Row: Icon + Title
        top_layout = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Segoe UI", 18))
        icon_label.setStyleSheet("background: transparent; border: none;")
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title_label.setStyleSheet("color: #8c9bb5; background: transparent; border: none;")
        
        top_layout.addWidget(icon_label)
        top_layout.addWidget(title_label)
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # Bottom Row: Value
        value_label = QLabel(str(value))
        value_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        value_label.setStyleSheet(f"color: {color}; background: transparent; border: none;")
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        
        layout.addStretch()
        layout.addWidget(value_label)
        
        self.stat_labels[title] = value_label
        return card
    
    def create_persons_section(self, persons_list):
        """Create the data table panel"""
        section_frame = QFrame()
        section_frame.setObjectName("TableCard")
        section_frame.setStyleSheet("""
            QFrame#TableCard { 
                background-color: #141829; 
                border: 1px solid #2a2e45; 
                border-radius: 12px; 
            }
        """)
        
        layout = QVBoxLayout(section_frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header & Button
        header_layout = QHBoxLayout()
        title = QLabel("Persons & Management")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #ffffff; background: transparent; border: none;")
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton { 
                background-color: #00bfff; color: #000000; font-weight: bold;
                border: none; border-radius: 6px; padding: 6px 15px; 
            }
            QPushButton:hover { background-color: #00d4ff; }
        """)
        refresh_btn.clicked.connect(self.refresh_stats)
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(refresh_btn)
        layout.addLayout(header_layout)
        
        # Table UI configuration
        self.persons_table = QTableWidget()
        self.persons_table.setFocusPolicy(Qt.NoFocus)
        self.persons_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.persons_table.setStyleSheet("""
            QTableWidget { 
                background-color: transparent; 
                color: #e2e8f0; 
                border: none; 
                gridline-color: #2a2e45; 
            }
            QTableWidget::item { 
                padding: 10px; 
                border-bottom: 1px solid #2a2e45; 
            }
            QTableWidget::item:selected { 
                background-color: #1e253c; 
            }
            QHeaderView::section { 
                background-color: #0f1221; 
                color: #8c9bb5; 
                padding: 10px; 
                border: none; 
                font-weight: bold;
                border-bottom: 2px solid #2a2e45;
            }
        """)
        
        self.persons_table.setColumnCount(5)
        self.persons_table.setHorizontalHeaderLabels(["No.", "Name", "ID", "Img Count", "Action"])
        self.persons_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.persons_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.persons_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.persons_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.persons_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Fixed)
        self.persons_table.setColumnWidth(2, 130)
        self.persons_table.setColumnWidth(3, 130)
        self.persons_table.setColumnWidth(4, 130)
        
        self.persons_table.verticalHeader().setVisible(False)
        self.persons_table.setShowGrid(False)
        self.persons_table.verticalHeader().setDefaultSectionSize(70) # Sets all rows to 70px tall
        self.populate_persons_table(persons_list)
        layout.addWidget(self.persons_table)
        
        return section_frame
    
    def populate_persons_table(self, persons_list):
        self.persons_table.setRowCount(0)
        if not persons_list:
            return
        
        unique_persons = {}
        for person in persons_list:
            person_id = str(person.get('ID', 'Unknown'))
            name = str(person.get('Name', 'Unknown'))
            if person_id not in unique_persons:
                unique_persons[person_id] = {'name': name, 'count': 0}
            unique_persons[person_id]['count'] += 1
        
        for row, (person_id, data) in enumerate(unique_persons.items()):
            self.persons_table.insertRow(row)
            
            def create_item(text, color):
                item = QTableWidgetItem(str(text))
                item.setForeground(QColor(color))
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
                return item
                
            self.persons_table.setItem(row, 0, create_item(row + 1, "#ffffff"))
            self.persons_table.setItem(row, 1, create_item(data['name'], "#00bfff"))
            self.persons_table.setItem(row, 2, create_item(person_id, "#ffd93d"))
            self.persons_table.setItem(row, 3, create_item(data['count'], "#6bcf7f"))
            
            # Action Button
            btn_container = QWidget()
            btn_layout = QHBoxLayout(btn_container)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            
            remove_btn = QPushButton("🗑️ Remove")
            remove_btn.setMinimumWidth(90)
            remove_btn.setCursor(Qt.PointingHandCursor)
            remove_btn.setStyleSheet("""
                QPushButton { 
                    background-color: rgba(255, 68, 68, 0.1); 
                    color: #ff4444; 
                    border: 1px solid #ff4444; 
                    border-radius: 6px; 
                    padding: 6px 12px; /* Increased padding slightly */
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #ff4444; color: white; }
            """)
            remove_btn.clicked.connect(lambda checked, pid=person_id: self.trigger_remove_person(pid))
            
            btn_layout.addWidget(remove_btn)
            btn_layout.setAlignment(Qt.AlignCenter)
            self.persons_table.setCellWidget(row, 4, btn_container)
    
    def trigger_remove_person(self, person_id):
        # 1. Protection for Default Person (P001)
        if str(person_id) == "P001":
            QMessageBox.critical(
                self, 
                "Action Denied", 
                "You cannot remove the default user (P001) from the database."
            )
            return

        # 2. Strict Confirmation Dialog
        reply = QMessageBox.warning(
            self, 
            "Confirm Permanent Deletion",
            f"Are you absolutely sure you want to completely remove Person ID: {person_id}?\n\n"
            "⚠️ This action is IRREVERSIBLE. It will permanently delete their images, "
            "extracted face features, and all database records.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 3. Disable the table so the user cannot click buttons in the background
            self.persons_table.setEnabled(False)
            
            self.progress_dialog = QMessageBox(self)
            self.progress_dialog.setWindowTitle("Deleting Data")
            self.progress_dialog.setText(f"Removing {person_id} from dataset...\nPlease wait.")
            self.progress_dialog.setStandardButtons(QMessageBox.NoButton)
            # Make the window strictly modal so it stays on top and prevents background clicks
            self.progress_dialog.setWindowModality(Qt.ApplicationModal) 
            self.progress_dialog.show()
            
            self.remove_thread = RemovePersonWorker(person_id, self.dataset_path)
            self.remove_thread.worker_finished.connect(self.on_remove_finished)
            self.remove_thread.start()
    
    def on_remove_finished(self, success, message):
        # 4. Perfectly clean up the modal dialog from memory
        if self.progress_dialog:
            self.progress_dialog.accept() # Safely closes the QDialog
            self.progress_dialog.deleteLater() # Flags it for immediate memory cleanup
            self.progress_dialog = None
            
        # Re-enable the table
        self.persons_table.setEnabled(True)
            
        if success:
            QMessageBox.information(self, "Success", message)
            self.refresh_stats()
        else:
            QMessageBox.critical(self, "Error", message)
    
    def refresh_stats(self):
        stats = self.stats_manager.get_all_statistics()
        
        mapping = {
            "Total Persons": stats['person_count'],
            "Total Embeddings": stats['embeddings_count'],
            "Unique Embeddings": stats['unique_embeddings'],
            "Original Images": stats['total_images'],
            "Face Images": stats['face_images'],
            "Database Size": stats['database_info'].get('total_size', '0 B')
        }
        
        for title, value in mapping.items():
            if title in self.stat_labels:
                self.stat_labels[title].setText(str(value))
        
        self.populate_persons_table(stats['all_persons'])


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    page = DashboardPage()
    page.resize(1200, 700)
    page.show()
    sys.exit(app.exec())