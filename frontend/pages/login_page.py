import time
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QLineEdit, QApplication, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QCursor
from backend.src.login_setup import FirebaseAuth

class LoginPage(QDialog):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.setWindowTitle("System Login - Face Recognition")
        # Instead of a tiny fixed size, we give it a large default size
        self.resize(1200, 800)
        
        # Store the logged-in user's email to pass to MainWindow later
        self.logged_in_email = None 
        
        # Initialize Firebase
        self.db = FirebaseAuth()
        
        # Build the User Interface
        self.init_ui()

    def init_ui(self):
        # 1. Main Background Layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)  # Centers the login card on the screen
        
        # Set a dark background for the whole screen
        self.setStyleSheet("""
            QDialog {
                background-color: #0a0e27; /* Matches your app's stacked widget background */
            }
        """)

        # 2. Create the "Login Card" (The box in the middle)
        card = QFrame()
        card.setFixedSize(450, 480)
        card.setObjectName("LoginCard")
        # Style the card to look like a modern floating container
        card.setStyleSheet("""
            QFrame#LoginCard {
                background-color: #1c2340; 
                border-radius: 15px;
                border: 1px solid #2a3456;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit {
                background-color: #0a0e27;
                color: #ffffff;
                border: 1px solid #3a4773;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QLineEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)

        # 3. Layout INSIDE the card
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(15)

        # Title
        title_label = QLabel("System Access")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)

        # Email Input
        email_label = QLabel("Email:")
        email_label.setFont(QFont("Arial", 10))
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Enter registered email")
        self.email_input.setMinimumHeight(40)

        # Password Input
        password_label = QLabel("Password:")
        password_label.setFont(QFont("Arial", 10))
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(40)

        # Status Message Label (Hidden by default)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.hide()

        # Buttons Layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # Login Button
        login_button = QPushButton("Login")
        login_button.setMinimumHeight(45)
        login_button.setCursor(QCursor(Qt.PointingHandCursor))
        login_button.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd; 
                color: white; 
                font-size: 14px;
                font-weight: bold; 
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #0b5ed7; }
        """)
        login_button.clicked.connect(self.handle_login)

        # Sign Up Button
        signup_button = QPushButton("Sign Up")
        signup_button.setMinimumHeight(45)
        signup_button.setCursor(QCursor(Qt.PointingHandCursor))
        signup_button.setStyleSheet("""
            QPushButton {
                background-color: #198754; 
                color: white; 
                font-size: 14px;
                font-weight: bold; 
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #157347; }
        """)
        signup_button.clicked.connect(self.handle_signup)

        button_layout.addWidget(login_button)
        button_layout.addWidget(signup_button)

        # Forgot Password Button
        forgot_pw_button = QPushButton("Forgot Password?")
        forgot_pw_button.setCursor(QCursor(Qt.PointingHandCursor))
        forgot_pw_button.setStyleSheet("""
            QPushButton {
                color: #8bb4f7; 
                background: transparent;
                border: none; 
                text-decoration: underline;
                font-size: 12px;
            }
            QPushButton:hover { color: #ffffff; }
        """)
        forgot_pw_button.clicked.connect(self.handle_forgot_password)

        # Assemble the card
        card_layout.addWidget(title_label)
        card_layout.addSpacing(20)
        card_layout.addWidget(email_label)
        card_layout.addWidget(self.email_input)
        card_layout.addSpacing(5)
        card_layout.addWidget(password_label)
        card_layout.addWidget(self.password_input)
        card_layout.addSpacing(5)
        card_layout.addWidget(self.status_label)
        card_layout.addSpacing(10)
        card_layout.addLayout(button_layout)
        card_layout.addSpacing(15)
        card_layout.addWidget(forgot_pw_button)

        main_layout.addWidget(card)
        self.setLayout(main_layout)

    def show_message(self, text, is_error=True):
        color = "#ff6b6b" if is_error else "#20c997" # Adjusted colors for dark theme
        self.status_label.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold; background: transparent;")
        self.status_label.setText(text)
        self.status_label.show()

    def handle_login(self):
        email = self.email_input.text().strip()
        password = self.password_input.text().strip()
        
        if not email or not password:
            self.show_message("Please enter both email and password")
            return

        self.show_message("Authenticating...", is_error=False)
        QApplication.processEvents()

        #time.sleep(3)  # Simulate network delay

        user = self.db.sign_in(email, password)
        if user:
            self.logged_in_email = email
            self.accept()
        else:
            self.show_message("Invalid email or password")
            self.password_input.clear()

    def handle_signup(self):
        email = self.email_input.text().strip()
        password = self.password_input.text().strip()
        
        if not email or not password:
            self.show_message("Please enter both email and password")
            return
            
        if len(password) < 6:
            self.show_message("Password must be at least 6 characters")
            return

        self.show_message("Creating account...", is_error=False)
        QApplication.processEvents()

        user = self.db.sign_up(email, password)
        if user:
            self.show_message("Account created! You can now log in.", is_error=False)
            self.password_input.clear()
        else:
            self.show_message("Error creating account. Email may already be in use.")

    def handle_forgot_password(self):
        email = self.email_input.text().strip()
        
        if not email:
            self.show_message("Enter your email above to reset password")
            return

        self.show_message("Sending reset link...", is_error=False)
        QApplication.processEvents()

        success = self.db.reset_password(email)
        if success:
            self.show_message("Password reset link sent to your email!", is_error=False)
        else:
            self.show_message("Error sending reset link. Is the email registered?")
            