# AeroSecure - Complete Face Recognition System

<div align="center">

**A comprehensive face recognition system with futuristic web interface designed for airport security operations**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>



---

## 📋 Quick Navigation

- [🚀 Quick Start](#-quick-start) - Get running in 2 minutes
- [✨ Features](#-features) - What the system can do
- [🏗️ Architecture](#️-architecture) - How it's built
- [ Setup & Installation](#-setup--installation) - Getting started
- [📖 Usage Guide](#-usage-guide) - How to use
- [🛠️ Configuration](#️-configuration) - Customization
- [🚀 Deployment](#-deployment) - Production setup



## 🚀 Quick Start

### **Option 1: Full System (Recommended)**
```bash
# Clone repository
git clone https://github.com/Vivek3825/FaceRecognitionSystem.git
cd FaceRecognitionSystem

# Setup backend
cd Backend/
pip install -r requirements.txt
python main.py  # Multi-camera GUI

# Setup frontend (new terminal)
cd Frontend/
python -m http.server 8000
# Open http://localhost:8000
```

### **Option 2: Backend Only**
```bash
cd Backend/
python main.py  # Multi-camera GUI
```

### **Option 3: Frontend Only**
```bash
cd Frontend/
python -m http.server 8000
# Open http://localhost:8000
```

---

## ✨ Features

### 🌐 **Frontend (AeroSecure Web Interface)**
- **Modern Dashboard** - Real-time statistics and security alerts
- **Camera Monitor** - Multi-camera surveillance with live detection
- **Person Search** - Photo-based and name-based search capabilities
- **Personnel Management** - Add new persons with facial capture
- **Futuristic Design** - Dark theme with advanced animations
- **Responsive Layout** - Works on desktop, tablet, and mobile
- **Real-time Updates** - Live data visualization and notifications

### 🔧 **Backend (Recognition Engine)**
- **Multi-Camera Support** - Process multiple cameras simultaneously
- **Real-time Recognition** - Live face detection and identification
- **High Accuracy** - >95% recognition accuracy with confidence scoring
- **Person Search** - Find person location across all cameras
- **Fast Registration** - ~15 seconds vs ~3 minutes (10x improvement)
- **Smart Processing** - Incremental data processing
- **Performance Optimized** - GPU acceleration and threading support
- **Database Management** - Efficient embedding storage and retrieval



## 🏗️ Architecture

### **System Overview**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (Web UI)      │◄──►│   (API Server)  │◄──►│   (Embeddings)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  • Dashboard    │    │  • Multi-Camera │    │  • Face Data    │
│  • Camera View  │    │  • Recognition  │    │  • Embeddings   │
│  • Person Mgmt  │    │  • Person Search│    │  • Metadata     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Technology Stack**
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | HTML5/CSS3/JavaScript | Modern web interface |
| **Backend** | Python 3.12+ | Face recognition engine |
| **AI Models** | MTCNN + InceptionResnetV1 | Face detection & recognition |
| **Database** | NPZ + CSV | Embedding storage |
| **GUI** | Tkinter | Desktop interface |
| **API** | Flask | REST API server |

### **Performance Improvements**
- **Before**: Full reprocessing → ~3 minutes
- **After**: Incremental processing → ~15 seconds
- **Improvement**: 12x faster registration

---

## 🔧 Setup & Installation

### **System Requirements**
- **Python** 3.12+
- **RAM** 8GB+ (4GB minimum)
- **Storage** 2GB+ free space
- **Camera** Any USB/built-in camera
- **Browser** Modern web browser (Chrome/Firefox recommended)

### **Step 1: Clone Repository**
```bash
git clone https://github.com/Vivek3825/FaceRecognitionSystem.git
cd FaceRecognitionSystem
```

### **Step 2: Setup Backend**
```bash
cd Backend/
pip install -r requirements.txt

# Process existing data (first time only)
python src/detect_faces.py
python src/extract_features.py
```

### **Step 3: Start System**
```bash
# Backend: Multi-camera system
python main.py

# Frontend: Web interface (new terminal)
cd Frontend/
python -m http.server 8000
# Open http://localhost:8000
```

---

## 📖 Usage Guide

### **Backend (Multi-Camera System)**
```bash
cd Backend/
python main.py
```
**Features:**
- Multi-camera view with live recognition
- Person search functionality  
- Real-time face detection and identification
- Parallel processing for multiple cameras

### **Frontend (Web Interface)**
```bash
cd Frontend/
python -m http.server 8000
# Open http://localhost:8000
```
**Features:**
- Modern dashboard with statistics
- Camera monitoring interface
- Person registration with photo capture
- Search functionality (photo/name based)

### **Person Registration Process**
1. **Enter Name** - Type person's full name
2. **Camera Access** - Allow camera permissions  
3. **Capture Photos** - Take front, left, right angle photos
4. **Submit** - System processes in ~15 seconds
5. **Success** - Person registered and ready for recognition

---

## ⚡ Performance Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Registration Time** | ~3 minutes | ~15 seconds | **12x faster** |
| **Processing Method** | Full reprocessing | Incremental only | **Smart** |
| **Memory Usage** | ~500MB | ~200MB | **60% less** |
| **Recognition Accuracy** | >95% | >95% | **Maintained** |

| **Data Processed** | All 45 people (135 images) | New person only (3 images) | **45x less data** |
| **System Response** | Blocked during processing | Responsive | **Better UX** |

### **Technical Improvements**
- **Incremental Processing** - Only processes new person data
- **Smart Rollback** - Failed registrations don't corrupt data
- **Path Consistency** - Unified relative paths across all CSV files
- **Concurrent Processing** - Non-blocking registration workflow

---

## 📁 Project Structure

```
FaceRecognitionSystem/
├── Frontend/                    # Web Interface
│   ├── index.html              # Main application
│   ├── html_section_files/     # Modular page sections
│   │   ├── dashboard.html      # Dashboard page
│   │   ├── camera_monitor.html # Camera monitoring
│   │   ├── person_search.html  # Person search
│   │   └── add_person.html     # Person registration
│   ├── styles/                 # CSS styling
│   │   ├── main.css           # Core styles
│   │   └── animations.css     # UI animations
│   └── js/                    # JavaScript
│       ├── main.js            # Application logic
│       └── section_loader.js  # Dynamic page loading
└── Backend/                   # Recognition Engine
    ├── main.py               # Multi-camera launcher
    ├── src/                  # Core modules
    │   ├── detect_faces.py   # Face detection
    │   ├── extract_features.py # Feature extraction
    │   ├── multi_camera_manager.py # Multi-camera engine
    │   └── multi_camera_gui.py # GUI interface
    └── dataset/              # Face database
        ├── images/           # Original photos
        ├── faces/           # Processed faces
        └── embeddings/      # Face embeddings
```

---

## 🛠️ Configuration

### **Camera Settings**
```python
# Adjust camera detection in main.py
cameras = []
for i in range(3):  # Check first 3 cameras
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        cameras.append({'id': i, 'name': f'Camera {i}'})
```

### **Performance Tuning**
```python
# For better performance (lower resource usage)
DETECTION_FPS = 10
FACE_DETECTION_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.6

# For better accuracy (higher resource usage)
DETECTION_FPS = 20
FACE_DETECTION_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.8
```

---

## 🆘 Troubleshooting

### **Common Issues**
| Problem | Solution |
|---------|----------|
| **Camera not detected** | Check USB connection, try different camera ID |
| **Low FPS** | Reduce detection threshold, close other apps |
| **No faces detected** | Improve lighting, check camera angle |
| **Wrong recognition** | Add more training images, retrain |
| **High CPU usage** | Enable GPU acceleration, reduce FPS |
| **GUI freezing** | Restart application, check thread issues |

### **Debug Mode**
```bash
# Start with debug info
cd Backend/
python main.py
# Watch console for detailed logs
```

### **Reset System**
```bash
# If system gets corrupted, rebuild embeddings
cd Backend/
python src/detect_faces.py
python src/extract_features.py
```

---

## 🚀 Deployment

### **Development**
```bash
# Backend
cd Backend && python main.py

# Frontend  
cd Frontend && python -m http.server 8000
```

### **Production**
```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 simple_api:app
```

### **Docker**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY Backend/ .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]
```

---

## 🎯 Performance Metrics

| Metric | Value | Notes |
|--------|-------|--------|
| **Recognition Accuracy** | >95% | On known faces |
| **Processing Speed** | <100ms | Per face detection |
| **Multi-Camera FPS** | 10-15 | Per camera |
| **Memory Usage** | ~200MB | Per camera |
| **Startup Time** | ~3-5 seconds | Model loading |
| **Database Size** | ~50MB | 1000 faces |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

---

## 🎉 Acknowledgments

- **MTCNN** for face detection
- **InceptionResnetV1** for face recognition
- **OpenCV** for computer vision
- **PyTorch** for deep learning
- Airport security teams for requirements and feedback

---

<div align="center">

**Made with ❤️ for Airport Security Teams Worldwide**

⭐ **Star this repo if you found it helpful!** ⭐

</div>
