#!/usr/bin/env python3
"""
Multi-Camera Face Recognition System - Clean Version
Simple launcher for the face recognition system

Usage: python3 main.py

Author: Cleaned up on August 26, 2025
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from multi_camera_gui import MultiCameraGUI

def main():
    """Main entry point"""
    print("🎯 Multi-Camera Face Recognition System")
    print("📅 Clean Version - August 26, 2025")
    print("🔧 Features: Multi-camera, Face recognition, Person search")
    print("-" * 50)
    
    try:
        # Create and run the GUI
        app = MultiCameraGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
