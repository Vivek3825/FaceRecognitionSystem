#!/usr/bin/env python3
"""
Simple API Server for Face Recognition System
Uses the person_registration.py with proper integration
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.person_registration import PersonRegistrationSystem

app = Flask(__name__)
CORS(app)

# Initialize registration system
registration_system = PersonRegistrationSystem()

@app.route('/')
def health_check():
    return jsonify({'status': 'Face Recognition API is running'})

@app.route('/api/cameras')
def get_cameras():
    """Get available cameras"""
    cameras = []
    for i in range(3):  # Check first 3 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({'id': i, 'name': f'Camera {i}'})
            cap.release()
    return jsonify(cameras)

@app.route('/api/register-person', methods=['POST'])
def register_person():
    """Register a new person"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        images = data.get('images', {})
        
        if not name:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
        
        # Convert images list to dict if needed
        if isinstance(images, list):
            if len(images) != 3:
                return jsonify({'success': False, 'error': 'Three images required (front, left, right)'}), 400
            images = {
                'front': images[0],
                'left': images[1], 
                'right': images[2]
            }
        
        # Check for required angles
        required_angles = ['front', 'left', 'right']
        for angle in required_angles:
            if angle not in images:
                return jsonify({'success': False, 'error': f'Missing {angle} image'}), 400
        
        # Register the person
        result = registration_system.register_person(name, images)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-next-id')
def get_next_id():
    """Get the next person ID"""
    try:
        next_id = registration_system.get_next_person_id()
        return jsonify({'next_id': next_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-consistency')
def verify_consistency():
    """Check data consistency"""
    try:
        report = registration_system.verify_data_consistency()
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/frontend')
def serve_frontend():
    """Serve the frontend"""
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'Frontend')
    return send_from_directory(frontend_path, 'index.html')

@app.route('/frontend/<path:filename>')
def serve_frontend_files(filename):
    """Serve frontend static files"""
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'Frontend')
    return send_from_directory(frontend_path, filename)

if __name__ == '__main__':
    print("🚀 Starting Face Registration API...")
    print("📡 Server available at: http://localhost:5000")
    print("🌐 Frontend available at: http://localhost:5000/frontend")
    print("─" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)