#!/usr/bin/env python3
"""
Launch the Face Recognition Surveillance System Frontend Application
"""
import sys
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir))

from frontend.main_window import main

if __name__ == "__main__":  # __name__ is a built in python function, work as a guard. 
    main()

"""
Without the guard, main() executes immediately on import in another file— launching your entire surveillance GUI unexpectedly. 
That's a side effect nobody asked for.
With the guard, main() only runs when you explicitly execute python run_app.py.
"""