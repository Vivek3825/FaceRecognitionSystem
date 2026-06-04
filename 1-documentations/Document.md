# Face Recognition Surveillance System

**A comprehensive desktop application for real-time multi-camera face detection, recognition, and person registration built with Python, PySide6, and advanced deep learning models.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [How It Works](#how-it-works)
- [Application Architecture](#application-architecture)
- [Main Screens & UI Components](#main-screens--ui-components)
- [Important Buttons & Controls](#important-buttons--controls)
- [Key Logic & Implementation Highlights](#key-logic--implementation-highlights)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)

---

## Project Overview

This is a **professional-grade face recognition surveillance system** built as a desktop application. It provides a modern PySide6-based GUI that integrates with powerful backend services for:

- **Real-time multi-camera monitoring** with live face detection overlay
- **Person registration** with 3-angle face capture (front, left, right)
- **Face recognition** using deep learning embeddings (FaceNet with InceptionResnet-V1)
- **Dataset management** with automatic validation and consistency checking
- **Comprehensive statistics** and database health monitoring

The system uses:
- **Frontend**: PySide6 (Qt for Python) with a modern dark theme
- **Camera Management & Image Processing**: OpenCV (cv2) for frame capture, resizing, preprocessing, and drawing overlays
- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Recognition**: FaceNet InceptionResnet-V1 with embeddings
- **Backend**: Python with NumPy, Pandas, PyTorch
- **Database**: CSV files with NumPy embeddings (NPZ format)

---

## Core Features

### 1. **Real-Time Camera Monitoring**
- Multi-camera support with simultaneous stream processing
- Configurable grid layouts (2×2, 1×4, 2×3)
- Individual camera feed enlargement with detailed info
- Live detection overlay showing recognized faces and confidence scores

### 2. **Person Registration System**
- Capture 3 face angles per person (front, left, right) for robust recognition
- Automatic face detection and validation before storing
- Incremental embedding generation for new registrations
- Consistent CSV/image file management

### 3. **Face Recognition Engine**
- Cosine similarity-based face matching against known embeddings
- Confidence threshold filtering (default: 85%)
- Real-time embedding generation for captured faces
- Platform-aware camera backend selection (Windows/Linux/macOS)

### 4. **Database & Dataset Management**
- Automatic dataset consistency validation on startup
- Self-healing mechanism for orphaned CSV rows or missing images
- Comprehensive statistics: person count, embedding count, image counts, database size
- Safe person deletion with cascading removal of all related data

### 5. **System Statistics Dashboard**
- Total registered persons, embeddings, and images at-a-glance
- Database size and file structure information
- Persons table with removal capability
- Live data refresh every 10 seconds

---

## How It Works

### Application Flow

```
run_frontend.py (Entry Point)
    ↓
frontend/main_window.py (MainWindow)
    ├── TopBarWidget (Time & system info)
    ├── SidebarWidget (Navigation)
    └── QStackedWidget (Page switching)
        ├── DashboardPage (Statistics)
        ├── CameraMonitorPage (Live monitoring)
        ├── RegistrationPage (Person registration)
        └── SettingsPage (Configuration)
```

### Backend Processing

1. **Dataset Manager** initializes on startup, validates all CSV files and image folders
2. **Multi-Camera Manager** loads known face embeddings from the NPZ database
3. **OpenCV (cv2)** captures raw video frames from camera hardware using platform-aware backends:
   - Reads frames continuously: `ret, frame = cap.read()`
   - Resizes frames: `cv2.resize(frame, (640, 480))`
   - Preprocesses images for model input (normalization, format conversion)
4. **Camera Stream** threads continuously capture frames in the background using OpenCV queues
5. **Face Detection** (MTCNN) identifies faces in each frame
6. **Face Recognition** computes embeddings and matches against known faces using cosine similarity
7. **OpenCV** draws detection results: bounding boxes, person names, confidence scores
8. **Results** are displayed live with overlays and sent to UI for rendering

---

## Application Architecture

### Frontend Architecture

```
frontend/
├── main_window.py          # Application entry, window management, page routing
├── config.py               # Central configuration (colors, sizes, thresholds)
├── utils.py                # Shared utilities and helpers
├── pages/                  # Page/Screen implementations
│   ├── dashboard_page.py   # Statistics & person management (READ-ONLY)
│   ├── camera_page.py      # Multi-camera monitoring & real-time detection
│   ├── registration_page.py # Person registration workflow
│   └── settings_page.py    # Configuration & camera setup
└── widgets/                # Reusable UI components
    ├── sidebar.py          # Navigation sidebar with active state tracking
    ├── topbar.py           # Top bar with datetime & system info
    ├── cards.py            # Statistics cards & info panels
    └── overlays.py         # Alerts, loading indicators, dialogs
```

### Backend Architecture

```
backend/
├── src/
│   ├── dataset_manager.py          # Dataset validation & auto-repair
│   ├── multi_camera_manager.py     # Camera control & face recognition
│   ├── person_registration.py      # Registration workflow & CSV management
│   └── stats_manager.py            # Statistics computation (READ-ONLY)
└── dataset/
    ├── info.csv                    # Master person registry (ID, Name, Image Path)
    ├── face_info.csv               # Face metadata & quality info
    ├── images/                     # Original captured images (front/left/right)
    ├── faces/                      # Cropped/normalized face images
    └── embeddings/
        ├── all_embeddings.npz      # FaceNet embeddings (binary format)
        └── embeddings.csv          # Embedding metadata (ID, Name, Timestamp)
```

---

## Main Screens & UI Components

### 1. **Dashboard Page** - System Overview & Management

**Purpose**: Central hub for monitoring system health and managing person records.

**Layout**:
```
┌─────────────────────────────────────────────────────┐
│  System Statistics & Database Info              [+] │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  │ 👥           │  │ 🧠           │  │ 🔗           │
│  │ Total        │  │ Total        │  │ Unique       │
│  │ Persons      │  │ Embeddings   │  │ Embeddings   │
│  │ [Count]      │  │ [Count]      │  │ [Count]      │
│  └──────────────┘  └──────────────┘  └──────────────┘
│
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  │ 📷           │  │ 👤           │  │ 💾           │
│  │ Original     │  │ Face         │  │ Database     │
│  │ Images       │  │ Images       │  │ Size         │
│  │ [Count]      │  │ [Count]      │  │ [Size]       │
│  └──────────────┘  └──────────────┘  └──────────────┘
│
│  ┌──────────────────────────────────────────────────┐
│  │ Registered Persons Table                         │
│  ├──────────────────────────────────────────────────┤
│  │ Sr.No | Name         | ID    | Images | Actions  │
│  │ 1     | John Doe     | P001  | 3      | Remove   │
│  │ 2     | Sarah Smith  | P002  | 3      | Remove   │
│  │ ...                                              │
│  └──────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────┘
```

**Statistics Cards** (6 cards in 2×3 grid):
- **Total Persons** (👥): Count of unique registered people
- **Total Embeddings** (🧠): Total embedding vectors in database
- **Unique Embeddings** (🔗): Count of unique embeddings by person
- **Original Images** (📷): Total raw image captures
- **Face Images** (👤): Processed cropped face images
- **Database Size** (💾): Total storage used by dataset folder

**Persons Table**:
- Lists all registered persons with Sr.No, Name, ID, and image count
- **Remove** button: Safely deletes person and all associated data (runs in background thread)
- Auto-refreshes every 10 seconds with live data

**Important Detail**: Dashboard is **READ-ONLY** for data integrity. Deletions trigger `RemovePersonWorker` thread to keep UI responsive.

---

### 2. **Camera Monitor Page** - Real-Time Multi-Camera Surveillance

**Purpose**: Live monitoring of multiple camera feeds with face detection overlay.

**Layout**:
```
┌─────────────────────────────────────────────────────┐
│ Camera Monitor                                      │
├─────────────────┬───────────────────────────────────┤
│ LEFT SECTION    │ RIGHT SECTION                     │
├─────────────────┼───────────────────────────────────┤
│                 │                                   │
│ View Mode: [▼]  │  Selected Camera                  │
│ [Load] [Start]  │  ┌───────────────────────────┐    │
│ [Stop]          │  │                           │    │
│                 │  │  ENLARGED FEED (800×420)  │    │
│ ┌─────────────┐ │  │                           │    │
│ │ CAM 1       │ │  │ Shows selected camera's   │    │
│ │ (thumbnail) │ │  │ live feed + overlay       │    │
│ │             │ │  │                           │    │
│ └─────────────┘ │  └───────────────────────────┘    │
│ ┌─────────────┐ │                                   │
│ │ CAM 2       │ │  Camera Details:                  │
│ │ (thumbnail) │ │  Name: Gate A - Entrance          │
│ │             │ │  FPS: 30                          │
│ └─────────────┘ │  Status: Active                   │
│                 │                                   │
│ Live Detections │                                   │
│ ├─ John Doe     │                                   │
│ │  Conf: 96%    │                                   │
│ ├─ Sarah Smith  │                                   │
│ │  Conf: 92%    │                                   │
│                 │                                   │
└─────────────────┴───────────────────────────────────┘
```

**Control Buttons**:
- **Load**: Initializes `MultiCameraManager`, loads embeddings, prepares AI models (MTCNN & InceptionResnet-V1)
- **Start**: Begins camera capture threads; displays live video streams
- **Stop**: Stops all camera threads and releases hardware resources
- **View Mode Combo**: Switch between 2×2, 1×4, or 2×3 grid layouts dynamically

**Grid Container**:
- Fixed 800×420 frame showing camera thumbnails
- Click any thumbnail to select it → displays enlarged in right panel
- Clicking updates "Selected Camera" panel with detail view

**Detection List**:
- Real-time updates showing recognized faces from all cameras
- Format: `[Camera Name] Name - Confidence%`
- Updates dynamically as faces are detected/matched

**Selected Camera Panel** (Right):
- **Enlarged Feed**: High-quality view of selected camera (800×420)
- **Camera Details**: Name, FPS, status, resolution
- Shows bounding boxes with names and confidence scores for detected faces

**Behind the Scenes**:
- Each camera runs in its own thread (`CameraStream._capture_loop()`)
- Frames stored in queue (max size 2) to avoid memory bloat
- MTCNN detects faces in parallel across all streams
- Embeddings computed and matched against `known_embeddings` using cosine similarity

---

### 3. **Registration Page** - Person Registration Workflow

**Purpose**: Guided workflow to register new people with 3-angle face captures.

**Layout**:
```
┌───────────────────────────────────────────────────┐
│ Person Registration                               │
├──────────────────┬────────────────────────────────┤
│ LEFT SECTION     │ RIGHT SECTION                  │
├──────────────────┼────────────────────────────────┤
│                  │                                │
│ Name Input:      │  Live Camera Preview           │
│ [_____________]  │  ┌────────────────────────┐    │
│                  │  │                        │    │
│ Camera ID:       │  │   Video Stream (480p)  │    │
│ [  ] [Start/Stop]│  │   with face box overlay│    │
│                  │  │                        │    │
│ [Capture Front]  │  └────────────────────────┘    │
│ [Capture Left]   │                                │
│ [Capture Right]  │  Progress Checklist:           │
│                  │  ☐ Front (0%)                  │
│ [Clear Form]     │  ☐ Left   (0%)                 │
│ [Commit]         │  ☐ Right  (0%)                 │
│                  │                                │
│ Status Messages: │  (Shows which angles captured) │
│ [Messages...]    │                                │
│                  │                                │
└──────────────────┴────────────────────────────────┘
```

**Input Fields & Controls**:

1. **Name Input**: Text field for person's name (stored in info.csv)

2. **Camera ID Input**: Integer camera index (typically 0, 1, 2...)
   - **Start/Stop Camera Button**: Toggles camera stream on/off
   - Validates camera accessibility before capture

3. **Capture Buttons** (3 buttons):
   - **[Capture Front]**: Snaps front-facing photo, auto-detects face
   - **[Capture Left]**: Snaps left profile photo
   - **[Capture Right]**: Snaps right profile photo
   - Images stored in temporary buffer until commit
   - Progress checklist updates in real-time

4. **[Clear Form]**: Resets all inputs, clears captured images, resets progress

5. **[Commit]**: 
   - Validates all 3 angles are captured
   - Saves images to `backend/dataset/images/`
   - Updates `info.csv` with person details
   - Runs face detection on all 3 images
   - Generates embeddings and updates `all_embeddings.npz`
   - Assigns auto-generated ID (P001, P002, etc.)

**Progress Checklist**:
- Shows visual checkmarks for each captured angle
- Blocks commit until all 3 angles are captured
- Prevents accidental re-submission

**Live Camera Preview**:
- Right panel shows real-time video stream
- Face detection visualization (green bounding box around detected faces)
- Quality feedback (helps user position face for capture)

**Behind the Scenes**:
- `PersonRegistrationSystem.save_images()`: Base64 decoding + PIL image save
- `run_face_detection_incremental()`: MTCNN on new images only (not full dataset)
- Embeddings generated via InceptionResnet-V1
- CSV rows auto-increment for consistency

---

### 4. **Settings Page** - Camera Configuration

**Purpose**: Configure which cameras are active and assign friendly display names.

**Two-Step Workflow**:

#### Step 1: Define Active Camera IDs
- **Input Field**: Enter comma-separated camera device IDs (e.g., `0, 1, 2`)
- **Button**: "🔄 Update Camera List"
- Saves the active cameras to the system

#### Step 2: Assign Display Names (Optional)
- **Table with 3 columns**:
  - **Column 1**: Camera ID (from Step 1)
  - **Column 2**: Current Name (from `camera_config.ini`, defaults to "Default Camera X")
  - **Column 3**: Input field to enter new name
- **Button**: "💾 Update/Overwrite Camera Names"
- Saves display names to `backend/camera_config.ini` for currently active cameras only

**Configuration File** (`backend/camera_config.ini`):
```ini
[Display Names]
0 = Gate A - Entrance
2 = Lobby - Main Hall
```
(Only contains entries for configured camera IDs)

---

## Important Buttons & Controls

### Global Buttons (Top Bar)

**Top Bar Widget**:
- **DateTime Display**: Live system time (updates every 1 second)
- **System Status Indicator**: Shows connection status to backend
- **Minimize/Maximize/Close**: Standard window controls

### Navigation (Sidebar)

**Page Navigation Buttons**:
- **Dashboard**: Home screen with statistics
- **Camera Monitor**: Multi-camera surveillance
- **Registration**: Person enrollment workflow
- **Settings**: System configuration
- **[Logout]** (bottom): Closes application (future expansion)

**Active State**: Highlighted button shows current page with left cyan border + color change

### Dashboard Buttons

| Button | Action | Purpose |
|--------|--------|---------|
| **Refresh Stats** | Manual refresh | Forces immediate stats recompute from CSV |
| **Remove** (per person) | Delete person + data | Cascading deletion, runs in background thread |

### Camera Monitor Buttons

| Button | Function | Details |
|--------|----------|---------|
| **Load** | Initialize system | Loads embeddings, prepares MTCNN & InceptionResnet models |
| **Start** | Begin capture | Starts all camera threads, displays frames |
| **Stop** | End capture | Stops threads, releases camera hardware |
| **View Mode** (Dropdown) | Switch layout | 2×2, 1×4, 2×3 grid options |

### Registration Buttons

| Button | Action | Effect |
|--------|--------|--------|
| **Start/Stop Camera** | Toggle preview | Opens/closes camera, displays live stream |
| **Capture Front** | Snap front image | Stores in buffer, updates checklist |
| **Capture Left** | Snap left image | Stores in buffer, updates checklist |
| **Capture Right** | Snap right image | Stores in buffer, updates checklist |
| **Clear Form** | Reset workflow | Clears inputs + buffers, resets checklist |
| **Commit** | Save person | Writes to CSV, generates embeddings, assigns ID |

### Settings Buttons

| Button | Action | Effect |
|--------|--------|--------|
| **Update** | Save camera list | Writes to config.ini, applies on restart |
| **Reset to Defaults** | Restore preset | Reverts to factory camera list |

---

## Key Logic & Implementation Highlights

### 1. **DatasetManager - Automatic Consistency Validation**

**Location**: `backend/src/dataset_manager.py`

**Problem Solved**: CSV rows can get orphaned or images can be missing due to failed operations, crashes, or manual edits.

**Solution - Validation Logic**:
```python
def validate_information(self):
    # Check 1: CSV file consistency
    # All IDs in face_info.csv should match info.csv and embeddings.csv
    if len(face_info_df) != len(info_df) or len(info_df) != len(embeddings_df):
        raise ValidationError(101)
    
    # Check 2: Folder consistency
    # Count of files in images/ and faces/ must match CSV row count
    if len(image_files) != len(info_df):
        raise ValidationError(104)
```

**Auto-Repair Mechanism**:
```python
def fix_issues(self, error_codes):
    if 101 in errors:  # CSV mismatch
        # Keep only rows where ID exists in info.csv (source of truth)
        self.face_info_df = self.face_info_df[self.face_info_df['ID'].isin(valid_ids)]
    
    if 104 in errors:  # Missing images
        # Remove image files for IDs not in CSV
        for img_file in images/:
            person_id = extract_id_from_filename(img_file)
            if person_id not in valid_ids:
                os.remove(img_file)
```

**Result**: App starts with guaranteed data consistency—no crashes from file mismatches.

---

### 2. **MultiCameraManager - Parallel Face Recognition**

**Location**: `backend/src/multi_camera_manager.py`

**Architecture**:
```
MultiCameraManager (Main Thread)
├── CameraStream 1 (Thread 1: Capture loop)
├── CameraStream 2 (Thread 2: Capture loop)
├── CameraStream N (Thread N: Capture loop)
└── Main UI Thread: Detection processing
```

**Thread-Safe Camera Capture**:
```python
class CameraStream:
    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put(frame, block=False)  # Non-blocking add
                except queue.Full:
                    # Queue full, drop oldest frame to prevent lag
                    self.frame_queue.get(block=False)
```

**Key Benefits**:
- Each camera captures independently (doesn't block others)
- Queue prevents memory bloat (max 2 frames per camera)
- Oldest frame dropped if processing is slow (maintains responsiveness)

**Face Recognition Pipeline**:
```python
for camera_name, camera in self.cameras.items():
    frame = camera.get_frame()
    
    # Detect faces in frame
    faces, _ = self.mtcnn.detect(frame)
    
    if faces is not None:
        for face_box in faces:
            # Crop face region
            face_img = frame[y1:y2, x1:x2]
            
            # Generate embedding
            embedding = self.model(preprocess(face_img))
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            # Match against known embeddings
            similarities = cosine_similarity([embedding], self.known_embeddings)
            best_match_idx = np.argmax(similarities)
            confidence = similarities[0][best_match_idx] * 100
            
            if confidence >= THRESHOLD:
                name = self.known_names[best_match_idx]
                self.recognition_results[camera_name].append({
                    'name': name,
                    'confidence': confidence,
                    'box': face_box
                })
```

---

### 3. **PersonRegistrationSystem - Incremental Registration**

**Location**: `backend/src/person_registration.py`

**Workflow**:
```
Step 1: Save Images (3 angles to disk)
    ↓
Step 2: Update info.csv (add person metadata)
    ↓
Step 3: Run Face Detection (MTCNN on new images only)
    ↓
Step 4: Generate Embeddings (InceptionResnet-V1)
    ↓
Step 5: Append to embeddings CSV & NPZ
```

**ID Generation Strategy** (Consistent):
```python
def get_next_person_id(self):
    # Find max ID number from existing CSV
    max_id = 0
    for row in csv_data:
        if row['ID'] starts with 'P':
            num = int(row['ID'][1:])
            max_id = max(max_id, num)
    
    return f"P{(max_id + 1):03d}"  # P001, P002, P003...
```

**Image Naming** (3 angles per person):
```
Person number 3, captured angles:
├── p3_front.jpeg  → info.csv entry 1
├── p3_left.jpeg   → info.csv entry 2
└── p3_right.jpeg  → info.csv entry 3
```

**CSV Entry** (One row per image):
```csv
Sr No, Name,       ID,   Image Path
1,     John Doe,   P001, dataset/images/p1_front.jpeg
2,     John Doe,   P001, dataset/images/p1_left.jpeg
3,     John Doe,   P001, dataset/images/p1_right.jpeg
```

**Embedding Storage**:
```python
# all_embeddings.npz (binary format for efficiency)
npz_file = {
    "P001_front": [0.123, 0.456, ...],  # 512-dim vector
    "P001_left":  [0.124, 0.457, ...],
    "P001_right": [0.125, 0.458, ...]
}

# embeddings.csv (human-readable metadata)
ID,    Name,       Embedding Key,  Timestamp
P001,  John Doe,   P001_front,     2024-06-04 10:30:45
P001,  John Doe,   P001_left,      2024-06-04 10:30:46
P001,  John Doe,   P001_right,     2024-06-04 10:30:47
```

---

### 4. **StatsManager - Read-Only Statistics Computation**

**Location**: `backend/src/stats_manager.py`

**Design**: Never modifies data—only reads and computes metrics for dashboard display.

**Key Metrics Computed**:
```python
def get_all_statistics(self):
    return {
        'person_count': len(info_df['ID'].unique()),      # Unique persons
        'embeddings_count': len(embeddings_df),            # Total embeddings
        'unique_embeddings': len(embeddings_df['ID'].unique()),
        'total_images': len(info_df),                      # Total image entries
        'face_images': len([f for f in faces_dir if is_file(f)]),
        'database_info': {
            'total_size': format_bytes(size_sum),
            'folders': {...},
            'last_modified': timestamp
        }
    }
```

**File Size Calculation** (Efficient):
```python
def get_database_info(self):
    # Calculate sizes for each folder
    faces_size = sum(os.path.getsize(f) for f in faces_dir)
    images_size = sum(os.path.getsize(f) for f in images_dir)
    embeddings_size = os.path.getsize(all_embeddings_npz)
    
    return {
        'folders': {
            'faces': {'count': 123, 'size': '45.2 MB'},
            'images': {'count': 41, 'size': '52.1 MB'},
            'embeddings': {'count': 2, 'size': '8.3 MB'}
        }
    }
```

---

### 5. **MainWindow - Page Navigation with QStackedWidget**

**Location**: `frontend/main_window.py`

**Pattern Used**: QStackedWidget (efficient page switching without destruction/recreation)

```python
class MainWindow(QMainWindow):
    def __init__(self):
        # Create pages
        self.pages = {
            "dashboard": DashboardPage(),
            "camera": CameraMonitorPage(),
            "registration": RegistrationPage(),
            "settings": SettingsPage(),
        }
        
        # Add all to stacked widget (all in memory)
        self.stacked_widget = QStackedWidget()
        for key, page in self.pages.items():
            self.stacked_widget.addWidget(page)
    
    def show_page(self, page_name):
        # Instant switching (no loading)
        page_index = list(self.pages.keys()).index(page_name)
        self.stacked_widget.setCurrentIndex(page_index)
```

**Advantages**:
- Instant page switching (no lag)
- All pages run in background (e.g., stats refresh continues)
- Memory efficient for small number of pages

---

### 6. **Threading for Responsive UI**

**Worker Thread Pattern** (Dashboard → Remove Person):

```python
class RemovePersonWorker(QThread):
    worker_finished = Signal(bool, str)
    
    def run(self):
        # Long operation on background thread
        RemovePerson(personID, dataset_path)
        self.worker_finished.emit(True, "Success message")

# In dashboard:
self.remove_thread = RemovePersonWorker(person_id, dataset_path)
self.remove_thread.worker_finished.connect(self.on_remove_complete)
self.remove_thread.start()  # Non-blocking!
```

**Result**: UI never freezes, user can continue navigating/viewing stats.

---

### 7. **OpenCV - Core Image Processing & Frame Capture**

**Location**: `backend/src/multi_camera_manager.py` (CameraStream class)

**Critical Role in Pipeline**:

OpenCV is the foundation of the entire vision pipeline. Every frame that gets processed must first be captured and prepared by OpenCV.

**Frame Capture**:
```python
def _capture_loop(self):
    while self.is_running:
        # OpenCV reads raw frames from camera hardware
        ret, frame = self.cap.read()
        
        if ret:
            # frame is a NumPy array (480, 640, 3) in BGR format
            self.last_frame = frame.copy()
            self.frame_queue.put(frame, block=False)
```

**Frame Preprocessing** (Before Face Detection):
```python
def start(self):
    self.cap = cv2.VideoCapture(camera_id)
    
    # Set resolution and properties for optimal performance
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Width
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height
    self.cap.set(cv2.CAP_PROP_FPS, 30)            # Frames per second
    
    # Platform-specific camera backend
    if platform == "Windows":
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    elif platform == "Linux":
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
```

**Drawing Detection Overlays**:
```python
# After MTCNN detection and face recognition, OpenCV draws results
def draw_detections(frame, detections, matches):
    for face_box, name, confidence in matches:
        x1, y1, x2, y2 = face_box
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put text label with person name and confidence
        label = f"{name} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame
```

**Key Responsibilities**:
1. **Frame Acquisition**: Continuously reads video frames from hardware cameras
2. **Format Conversion**: Converts between BGR (OpenCV), RGB (display), and model input formats
3. **Resizing**: Scales frames to optimal size for processing (e.g., 640×480)
4. **Preprocessing**: Normalizes pixel values for deep learning models
5. **Visualization**: Draws bounding boxes, text labels, and confidence scores
6. **Performance Optimization**: Discards frames if processing is slow (queue management)

**Integration Points**:
- **Input**: Raw camera frames in BGR format (NumPy arrays)
- **Output**: Processed frames with detection overlays for UI display
- **To MTCNN**: Frames passed for face detection
- **From Recognition**: Detection results used to draw overlays back on frames

**Performance Notes**:
- OpenCV operations are optimized in C++ (faster than pure Python)
- Frame queue prevents lag by dropping old frames if processing is slow
- Resolution set to 640×480 balances accuracy and speed
- Platform-aware backends ensure compatibility across OS

---

### 8. **Platform-Aware Camera Backend**

**Location**: `backend/src/multi_camera_manager.py` → `CameraStream.start()`

**Dynamic Backend Selection** (Windows/Linux/macOS):
```python
def start(self):
    system = platform.system()
    
    if system == "Windows":
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    elif system == "Linux":
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    elif system == "Darwin":  # macOS
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
```

**Why**: Different OSes use different camera drivers. This ensures compatibility.

---

## Project Structure

```
FaceRecognitionSystem/
│
├── 📄 README.md                          # This file
├── 🐍 run_frontend.py                    # Application entry point
├── � requirements.txt                   # Python dependencies
│
├── 📁 frontend/                          # PySide6 GUI Application
│   ├── 🐍 __init__.py                    # Package marker
│   ├── 🐍 main_window.py                 # Main application window & page routing
│   ├── 🐍 config.py                      # Application configuration (colors, sizes, thresholds)
│   ├── 🐍 utils.py                       # Shared utilities & helpers
│   │
│   ├── 📁 pages/                         # Application screens (pages)
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 dashboard_page.py          # Statistics dashboard + person management
│   │   ├── 🐍 camera_page.py             # Real-time multi-camera monitoring
│   │   ├── 🐍 registration_page.py       # Person registration workflow
│   │   └── 🐍 settings_page.py           # System configuration
│   │
│   ├── 📁 widgets/                       # Reusable UI components
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 sidebar.py                 # Navigation sidebar with active state
│   │   ├── 🐍 topbar.py                  # Top bar with datetime & system info
│   │   ├── 🐍 cards.py                   # Statistics cards & info panels
│   │   └── 🐍 overlays.py                # Alerts, loading spinners, dialogs
│   │
│   ├── 📁 styles/                        # Stylesheets
│   │   └── 🎨 dark_theme.qss             # Qt style sheet (dark theme)
│   │
│   └── 📁 rough/                         # Experimental/draft code (not used)
│
├── 📁 backend/                           # Python backend logic
│   │
│   ├── ⚙️ camera_config.ini              # Camera configuration (device IDs, names)
│   │
│   ├── 📁 src/                           # Core business logic
│   │   ├── 🐍 dataset_manager.py         # Dataset validation & auto-repair
│   │   ├── 🐍 multi_camera_manager.py    # Multi-camera control & face recognition
│   │   ├── 🐍 person_registration.py     # Person registration workflow
│   │   └── 🐍 stats_manager.py           # Statistics computation (read-only)
│   │
│   └── 📁 dataset/                       # Face recognition database
│       ├── 📄 info.csv                   # Master registry (ID, Name, Image Path)
│       ├── 📄 face_info.csv              # Face metadata (quality, landmarks)
│       │
│       ├── 📁 images/                    # Original 3-angle captures
│       │   ├── p1_front.jpeg             # Person 1 front
│       │   ├── p1_left.jpeg              # Person 1 left
│       │   ├── p1_right.jpeg             # Person 1 right
│       │   └── ... (3 images per person)
│       │
│       ├── 📁 faces/                     # Processed cropped faces (MTCNN output)
│       │   ├── P001_front_face.jpg       # Normalized face image
│       │   └── ...
│       │
│       └── 📁 embeddings/                # Face embeddings & metadata
│           ├── 📦 all_embeddings.npz     # Binary embeddings (512-dim vectors per face)
│           └── 📄 embeddings.csv         # Embedding metadata (ID, Name, Timestamp)
│
└── 🧪 test1.py, test2.py                # Development test scripts
```

### File Descriptions

**Frontend Core**:
- `main_window.py`: Initializes QMainWindow, builds layout (topbar + sidebar + content), manages page switching
- `config.py`: Centralized configuration (APP_VERSION, colors, thresholds, sizes)
- `utils.py`: Shared functions (image conversion, data formatting, etc.)

**Frontend Pages**:
- `dashboard_page.py`: Statistics display (read-only), person removal (background thread)
- `camera_page.py`: Multi-camera grid, stream selection, detection overlay
- `registration_page.py`: Guided person enrollment (3-angle capture + commit)
- `settings_page.py`: Camera ID configuration, config file management

**Frontend Widgets**:
- `sidebar.py`: Navigation menu with active button highlighting
- `topbar.py`: System time display, status indicators
- `cards.py`: Statistics card components with icons and values
- `overlays.py`: Alert boxes, loading spinners, confirmation dialogs

**Backend Core**:
- `dataset_manager.py`: Validates CSV/image consistency on startup, auto-repairs orphaned data
- `multi_camera_manager.py`: Controls multiple camera threads, orchestrates face detection & recognition
- `person_registration.py`: Handles image saving, CSV updates, face detection, embedding generation
- `stats_manager.py`: Reads dataset files, computes statistics for dashboard (never modifies)

**Data Files**:
- `info.csv`: Master person registry (1 row per image angle)
- `face_info.csv`: Additional metadata about detected faces
- `all_embeddings.npz`: Binary NumPy file with 512-dimensional FaceNet embeddings
- `embeddings.csv`: Human-readable mapping of embeddings to person IDs

---

## Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Ubuntu/Linux** (or compatible OS)
- **Camera hardware** (built-in or USB webcam)
- **GPU** (CUDA-enabled NVIDIA card recommended for faster face detection; CPU fallback available)

### Step 1: Clone/Download Project
```bash
cd /path/to/FaceRecognitionSystem
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PySide6 (GUI framework)
- PyTorch (deep learning)
- Facenet-pytorch (MTCNN + InceptionResnet-V1)
- OpenCV (camera + image processing)
- NumPy, Pandas (data handling)
- Scikit-learn (similarity metrics)
- And all other required packages

### Step 4: Run the Application

```bash
python3 run_frontend.py
```

### Step 5: Configure Cameras (via Settings Page)

After launching the app:

1. Navigate to **Settings** page
2. Enter your active camera IDs (comma-separated, e.g., `0, 1, 2`) → Click **Update Camera List**
3. (Optional) Assign friendly display names to each camera in the table → Click **Update/Overwrite Camera Names**

This populates `backend/camera_config.ini` with:
```ini
[Display Names]
0 = Gate A - Entrance
1 = Gate B - Exit
2 = Lobby - Main Hall
```

Or directly edit `backend/camera_config.ini` (manually) if you prefer (not recommended).

---

## Workflow: Camera Setup
1. Navigate to **Settings** page
2. Enter camera IDs (e.g., "0,1,2")
3. Click **[Update]**

### Step 5: First-Time Setup

1. **Dashboard**: Verify system statistics load correctly
2. **Camera Monitor**: Click [Load] → [Start] to begin monitoring
3. **Registration**: Enroll first person (3-angle capture)
4. **Recognition**: Return to Camera Monitor—newly registered face should be detected

---

## Database Structure

### CSV Format Reference

**info.csv** (Master Registry):
```csv
Sr No.,Name,ID,Image Path
1,John Doe,P001,dataset/images/p1_front.jpeg
2,John Doe,P001,dataset/images/p1_left.jpeg
3,John Doe,P001,dataset/images/p1_right.jpeg
4,Sarah Smith,P002,dataset/images/p2_front.jpeg
...
```

**embeddings.csv** (Embedding Metadata):
```csv
ID,Name,Embedding Key,Timestamp
P001,John Doe,P001_front,2024-06-04 10:30:45
P001,John Doe,P001_left,2024-06-04 10:30:46
P001,John Doe,P001_right,2024-06-04 10:30:47
...
```

**face_info.csv** (Face Quality Metrics):
```csv
ID,Landmarks,Quality Score,Angle,Timestamp
P001,[[x,y],...],0.95,front,2024-06-04 10:30:45
...
```

---

## Configuration Reference

### ApplicationConfig (frontend/config.py)

```python
# Recognition
DEFAULT_CONFIDENCE_THRESHOLD = 85      # % threshold for face match

# Camera
DEFAULT_FPS = 30
DEFAULT_RESOLUTION = "1080p"

# UI
SIDEBAR_WIDTH = 250
TOPBAR_HEIGHT = 60

# Colors
COLOR_PRIMARY = "#00bfff"       # Cyan
COLOR_BACKGROUND = "#0a0e27"   # Dark blue
COLOR_SUCCESS = "#4caf50"       # Green
```

### camera_config.ini

```ini
[Display Names]
0 = Gate A - Entrance
1 = Gate B - Exit
2 = Lobby - Main Hall
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No camera found" on Load | Camera not detected by system | Check `ls /dev/video*` on Linux; verify USB connection |
| "Failed to initialize MTCNN" | GPU/CUDA issue or PyTorch not installed | Run `pip install torch torchvision` or use CPU fallback |
| "CSV file not found" | Dataset folder missing | Extract `dummy_dataset.zip` or create dataset structure |
| Registration hangs | Face detection timeout | Ensure face is clearly visible; try different lighting |
| UI freezes during remove | Main thread blocking | Wait—removal runs in background thread, UI will respond |
| Settings not persisting | Config file permissions | Check write permissions on `backend/camera_config.ini` |

---

## Performance Tips

1. **GPU Acceleration**: Install CUDA + cuDNN for 10x faster face detection
2. **Multi-Camera**: Limit to 4 simultaneous cameras for optimal performance
3. **Recognition Threshold**: Increase to 90% for stricter matching (fewer false positives)
4. **Resolution**: Reduce camera resolution (480p vs 1080p) for smoother streaming

---

## Future Enhancements

- [ ] Liveness detection (prevent spoofing with photos)
- [ ] Attendance logging with timestamps
- [ ] Export reports (CSV, PDF)
- [ ] Alert system for unauthorized persons
- [ ] Database backup & restoration
- [ ] Multi-user authentication
- [ ] REST API for integration with other systems

---

**Last Updated**: June 4, 2024
**Version**: 1.0.0
**Author**: Vivek Avhad
**License**: MIT (or your chosen license)
   python3 run_frontend.py
   ```

## Notes
- The app expects the backend dataset folder and embedding data to exist.
- Packaging for Ubuntu is possible, but native dependencies like PySide6, OpenCV, and PyTorch must be bundled correctly.
- This README is intentionally concise and focused on project purpose, structure, and main UI flows.