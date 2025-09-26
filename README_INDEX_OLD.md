# AeroSecure - Project Index & Quick Reference

## 📂 Project Structure Overview

```
FaceRecognitionSystem/
├── 📖 README.md                    # Main documentation (you are here!)
├── 📋 README_INDEX.md             # This index file
├── 📄 Complete Face Recognition System - Project Flow & Implementation Guide.pdf
├── 🌐 Frontend/                   # Web Interface (AeroSecure)
│   ├── index.html                 # Main web application
│   ├── styles/                    # CSS styling
│   │   ├── main.css              # Core styles
│   │   ├── animations.css        # Advanced animations
│   │   └── camera-monitor.css    # Camera specific styles
│   ├── js/                       # JavaScript functionality
│   │   ├── main.js              # Application logic
│   │   └── animations.js        # Animation controller
│   └── README.md                 # Frontend documentation
└── ⚙️ Backend/                   # Recognition Engine
    ├── main.py                   # Main launcher
    ├── requirements.txt          # Dependencies
    ├── camera_config_clean.ini   # Camera configuration
    ├── src/                      # Source code modules
    │   ├── detect_faces.py       # Face detection
    │   ├── extract_features.py   # Feature extraction
    │   ├── match_face.py         # Face matching
    │   ├── real_time_recognition.py    # Single camera
    │   ├── multi_camera_manager.py     # Multi-camera engine
    │   └── multi_camera_gui.py         # GUI interface
    └── dataset/                  # Face database
        ├── images/               # Raw person photos
        ├── faces/               # Processed face crops
        └── embeddings/          # Face embeddings
            ├── embeddings.csv   # Metadata
            └── all_embeddings.npz # Vectors
```

---

## 🚀 Quick Start Options

### **Option 1: Web Interface Only**
```bash
cd Frontend/
python -m http.server 8000
# Open http://localhost:8000
```

### **Option 2: Face Recognition Only**  
```bash
cd Backend/
python main.py
```

### **Option 3: Full System**
```bash
# Terminal 1: Backend
cd Backend/ && python main.py

# Terminal 2: Frontend  
cd Frontend/ && python -m http.server 8000
```

---

## 📚 Documentation Quick Links

| Component | Documentation | Purpose |
|-----------|---------------|---------|
| 🎯 **Main Guide** | [README.md](README.md) | Complete system documentation |
| 🌐 **Frontend** | [Frontend/README.md](Frontend/README.md) | Web interface guide |
| ⚙️ **Backend** | [Backend/ modules](Backend/src/) | Recognition engine code |
| 📄 **PDF Guide** | [Implementation Guide.pdf](Complete%20Face%20Recognition%20System%20-%20Project%20Flow%20&%20Implementation%20Guide.pdf) | Detailed project flow |

---

## 🔧 Key Files by Function

### **🚀 Getting Started**
- `README.md` - Start here for complete setup
- `Backend/main.py` - Launch face recognition system
- `Frontend/index.html` - Open web interface
- `Backend/requirements.txt` - Install dependencies

### **⚙️ Configuration**
- `Backend/camera_config_clean.ini` - Camera settings
- `Backend/dataset/` - Face database location
- `Frontend/styles/main.css` - UI customization

### **🧠 Core Algorithms**
- `Backend/src/detect_faces.py` - MTCNN face detection
- `Backend/src/extract_features.py` - InceptionResnetV1 embeddings
- `Backend/src/match_face.py` - Face matching logic
- `Backend/src/multi_camera_manager.py` - Multi-camera processing

### **🎨 User Interface**
- `Frontend/js/main.js` - Web app functionality
- `Backend/src/multi_camera_gui.py` - Desktop GUI
- `Frontend/styles/animations.css` - Advanced animations

---

## 🏃‍♂️ Common Workflows

### **First Time Setup**
1. 📖 Read [README.md](README.md) - Complete setup guide
2. 📦 Install dependencies: `pip install -r Backend/requirements.txt`
3. 📷 Add face images to `Backend/dataset/images/`
4. 🔄 Process faces: `python Backend/src/detect_faces.py`
5. 🧠 Generate embeddings: `python Backend/src/extract_features.py`
6. 🚀 Launch system: `python Backend/main.py`

### **Adding New People**
1. 📷 Add photos to `Backend/dataset/images/`
2. 🔄 Run `python Backend/src/detect_faces.py`
3. 🧠 Run `python Backend/src/extract_features.py`
4. ♻️ Restart system

### **Frontend Development**
1. 🌐 Navigate to [Frontend/](Frontend/)
2. 📖 Read [Frontend/README.md](Frontend/README.md)
3. 🚀 Start server: `python -m http.server 8000`
4. 🔧 Edit files in `Frontend/styles/` or `Frontend/js/`

---

## 🆘 Quick Troubleshooting

| Issue | Quick Fix | Detailed Help |
|-------|-----------|---------------|
| 📷 **Camera not working** | Check USB, try different camera ID | [README.md#troubleshooting](README.md#-troubleshooting) |
| 🐌 **Slow performance** | Reduce FPS, close other apps | [README.md#configuration](README.md#️-configuration) |
| ❌ **No faces detected** | Better lighting, check angles | [README.md#troubleshooting](README.md#-troubleshooting) |
| 🔄 **Need to restart** | Kill Python processes, restart | [README.md#usage-guide](README.md#-usage-guide) |

---

## 📊 System Requirements

- **Python**: 3.12+ 
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 5GB+ free space
- **Cameras**: USB webcams or built-in camera
- **GPU**: Optional (for acceleration)

---

## 🎯 Feature Summary

### **🌐 Frontend (AeroSecure Web)**
- Modern dashboard with live stats
- Multi-camera monitoring
- Person search (photo + name)
- Add new personnel
- Futuristic airport security theme

### **⚙️ Backend (Recognition Engine)**
- Real-time face recognition
- Multi-camera support
- High accuracy (>95%)
- Person search across cameras
- Tkinter desktop GUI

---

## 🔄 Version Information

- **Current Version**: v5.0
- **Frontend**: Complete AeroSecure web interface
- **Backend**: Multi-camera recognition system
- **Documentation**: Cleaned and organized
- **Next**: API integration between frontend/backend

---

## 🤝 Contributing

Want to contribute? Check these files:
- [README.md](README.md) - Main project info
- [Frontend/README.md](Frontend/README.md) - Frontend specific
- Issues and PRs welcome on GitHub!

---

<div align="center">

**📖 For complete documentation, see [README.md](README.md)**

**🌐 For web interface details, see [Frontend/README.md](Frontend/README.md)**

**Made with ❤️ for Airport Security**

</div>