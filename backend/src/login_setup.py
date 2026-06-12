import pyrebase
import json
from pathlib import Path
import sys

class FirebaseAuth:
    def __init__(self):

        config_path = Path(__file__).parent.parent / "login_credentials.json"

        try:
            # Safely open and read the JSON file
            with open(config_path, 'r') as file:
                self.firebaseConfig = json.load(file)
                
        except FileNotFoundError:
            print(f"❌ CRITICAL ERROR: Could not find credentials at {config_path}")
            print("Please ensure 'login_credentials.json' exists in your backend folder.")
            sys.exit(1) # Stop the application safely if credentials are missing
        except json.JSONDecodeError:
            print(f"❌ CRITICAL ERROR: 'login_credentials.json' is not formatted correctly.")
            sys.exit(1)

        self.firebase = pyrebase.initialize_app(self.firebaseConfig)
        self.auth = self.firebase.auth()

    def sign_in(self, email, password):
        try:
            user = self.auth.sign_in_with_email_and_password(email, password)
            return user
        except Exception as e:
            print(f"Error signing in: {e}")
            return None
        
    def sign_up(self, email, password): 
        try:
            user = self.auth.create_user_with_email_and_password(email, password)
            return user
        except Exception as e:
            print(f"Error signing up: {e}")
            return None

    def reset_password(self, email):
        """Sends a password reset link to the provided email."""
        try:
            self.auth.send_password_reset_email(email)
            return True
        except Exception as e:
            print(f"Error sending reset email: {e}")
            return False

if __name__ == "__main__":
    auth = FirebaseAuth()
    email = "vivekavhad@gmail.com"
    password = "Vivek@123"
    user = auth.sign_in(email, password)
    if user:
        print("Successfully signed in!")
    else:        
        print("Failed to sign in.")
    