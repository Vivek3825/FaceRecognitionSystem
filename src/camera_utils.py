# src/camera_utils.py
# NEW FILE: Utility functions for camera management
# Purpose: Helper functions for camera detection and configuration
# Author: Added on August 25, 2025

import cv2
import platform
import subprocess
from typing import List, Dict, Tuple

def detect_available_cameras(max_cameras: int = 10) -> List[Dict]:
    """
    NEW FUNCTION: Detect all available cameras on the system
    
    Args:
        max_cameras: Maximum number of cameras to check (default: 10)
        
    Returns:
        List of dictionaries with camera information
    """
    available_cameras = []
    
    print("🔍 Scanning for available cameras...")
    
    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            # Test if camera actually works
            ret, frame = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                camera_info = {
                    'id': camera_id,
                    'name': f"Camera {camera_id}",
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'status': 'Available'
                }
                
                # Try to identify camera type
                if camera_id == 0:
                    camera_info['name'] = "Built-in Camera"
                    camera_info['type'] = "built-in"
                else:
                    camera_info['name'] = f"External Camera {camera_id}"
                    camera_info['type'] = "external"
                
                available_cameras.append(camera_info)
                print(f"✅ Found: {camera_info['name']} - {camera_info['resolution']} @ {fps}fps")
            
            cap.release()
        
    print(f"📹 Total cameras found: {len(available_cameras)}")
    return available_cameras

def get_camera_backend_info() -> Dict:
    """
    NEW FUNCTION: Get information about available camera backends
    
    Returns:
        Dictionary with backend information
    """
    backends = {}
    
    # Check OpenCV backends
    try:
        backends['opencv_version'] = cv2.__version__
        backends['available_backends'] = []
        
        # Common backends to check
        backend_list = [
            ('DirectShow', cv2.CAP_DSHOW),
            ('V4L2', cv2.CAP_V4L2),
            ('GStreamer', cv2.CAP_GSTREAMER),
            ('FFMPEG', cv2.CAP_FFMPEG)
        ]
        
        for name, backend in backend_list:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                backends['available_backends'].append(name)
                cap.release()
                
    except Exception as e:
        print(f"⚠️ Error checking backends: {e}")
    
    return backends

def configure_camera_optimal_settings(cap: cv2.VideoCapture, camera_id: int) -> bool:
    """
    NEW FUNCTION: Configure camera with optimal settings for face recognition
    
    Args:
        cap: OpenCV VideoCapture object
        camera_id: Camera index
        
    Returns:
        True if configuration successful, False otherwise
    """
    try:
        # Set optimal resolution for face recognition
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set frame rate
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Set auto-exposure and auto-focus if available
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto-exposure
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable auto-focus
        
        # Verify settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📷 Camera {camera_id} configured: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
        
    except Exception as e:
        print(f"⚠️ Error configuring camera {camera_id}: {e}")
        return False

def get_phone_camera_instructions() -> str:
    """
    NEW FUNCTION: Get instructions for connecting phone as camera
    
    Returns:
        String with detailed instructions
    """
    instructions = """
📱 PHONE CAMERA SETUP INSTRUCTIONS
==================================

Option 1: USB Connection (Android)
----------------------------------
1. Enable USB Debugging on your phone:
   - Go to Settings > About Phone
   - Tap "Build Number" 7 times to enable Developer Options
   - Go to Settings > Developer Options
   - Enable "USB Debugging"

2. Connect phone via USB cable

3. Install DroidCam or similar app:
   - Download DroidCam from Play Store
   - Install DroidCam client on computer
   - Follow app instructions to connect

Option 2: WiFi Connection (Android/iPhone)
-----------------------------------------
1. Install IP Webcam app (Android) or EpocCam (iPhone)
2. Connect phone and computer to same WiFi network
3. Start camera server on phone
4. Note the IP address shown in app
5. Use IP camera URL in application

Option 3: Direct USB (Some Android phones)
-----------------------------------------
1. Connect phone via USB
2. Enable "USB Camera" or "Webcam" mode if available
3. Phone should appear as additional camera device

Common Camera IDs:
- 0: Built-in laptop camera
- 1: First external/phone camera
- 2: Second external camera
- etc.

Troubleshooting:
- Try different camera IDs (0, 1, 2, 3...)
- Check if phone is recognized by system
- Restart both phone and computer if needed
- Ensure no other apps are using the camera
"""
    return instructions

def test_camera_connection(camera_id: int, duration: int = 5) -> Dict:
    """
    NEW FUNCTION: Test camera connection and quality
    
    Args:
        camera_id: Camera index to test
        duration: Test duration in seconds
        
    Returns:
        Dictionary with test results
    """
    test_results = {
        'camera_id': camera_id,
        'success': False,
        'frames_captured': 0,
        'average_fps': 0,
        'resolution': None,
        'errors': []
    }
    
    try:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            test_results['errors'].append("Camera failed to open")
            return test_results
        
        # Configure camera
        configure_camera_optimal_settings(cap, camera_id)
        
        # Test capture
        import time
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if test_results['resolution'] is None:
                    h, w = frame.shape[:2]
                    test_results['resolution'] = f"{w}x{h}"
            else:
                test_results['errors'].append("Failed to read frame")
                break
            
            time.sleep(0.033)  # ~30 FPS
        
        elapsed = time.time() - start_time
        test_results['frames_captured'] = frame_count
        test_results['average_fps'] = frame_count / elapsed if elapsed > 0 else 0
        test_results['success'] = frame_count > 0
        
        cap.release()
        
        print(f"📊 Camera {camera_id} test: {frame_count} frames, {test_results['average_fps']:.1f} FPS")
        
    except Exception as e:
        test_results['errors'].append(f"Test error: {str(e)}")
    
    return test_results

def create_camera_config_file(cameras_info: List[Dict], filename: str = "camera_config.txt"):
    """
    NEW FUNCTION: Create a configuration file with camera information
    
    Args:
        cameras_info: List of camera information dictionaries
        filename: Output filename
    """
    try:
        with open(filename, 'w') as f:
            f.write("CAMERA CONFIGURATION\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {__import__('datetime').datetime.now()}\n\n")
            
            for camera in cameras_info:
                f.write(f"Camera ID: {camera['id']}\n")
                f.write(f"Name: {camera['name']}\n")
                f.write(f"Type: {camera.get('type', 'unknown')}\n")
                f.write(f"Resolution: {camera['resolution']}\n")
                f.write(f"FPS: {camera['fps']}\n")
                f.write(f"Status: {camera['status']}\n")
                f.write("-" * 30 + "\n")
            
            f.write("\nUSAGE NOTES:\n")
            f.write("- Use Camera ID in the GUI to add cameras\n")
            f.write("- Built-in cameras are usually more stable\n")
            f.write("- External cameras may need specific drivers\n")
        
        print(f"📄 Camera configuration saved to: {filename}")
        
    except Exception as e:
        print(f"⚠️ Error saving camera config: {e}")

# NEW MAIN EXECUTION BLOCK FOR TESTING
if __name__ == "__main__":
    """
    Standalone camera utility for testing and configuration
    """
    print("🔧 Camera Utility Tool")
    print("=" * 40)
    
    # Detect available cameras
    cameras = detect_available_cameras()
    
    if cameras:
        print("\n📋 Creating camera configuration file...")
        create_camera_config_file(cameras)
        
        print("\n🧪 Testing camera connections...")
        for camera in cameras:
            test_results = test_camera_connection(camera['id'], 3)
            if test_results['success']:
                print(f"✅ Camera {camera['id']}: Working properly")
            else:
                print(f"❌ Camera {camera['id']}: Issues detected")
                for error in test_results['errors']:
                    print(f"   - {error}")
    else:
        print("❌ No cameras detected!")
        print("\n📱 Phone Camera Setup:")
        print(get_phone_camera_instructions())
    
    # Backend information
    print("\n🔍 Camera Backend Information:")
    backend_info = get_camera_backend_info()
    print(f"OpenCV Version: {backend_info.get('opencv_version', 'Unknown')}")
    print(f"Available Backends: {', '.join(backend_info.get('available_backends', []))}")
