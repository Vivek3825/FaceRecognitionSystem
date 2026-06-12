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

- **Firebase Authentication** with secure login, sign-up, and password reset
- **Real-time multi-camera monitoring** with live face detection overlay
- **Person registration** with 3-angle face capture (front, left, right)
- **Face recognition** using deep learning embeddings (FaceNet with InceptionResnet-V1)
- **Dataset management** with automatic validation and consistency checking
- **Comprehensive statistics** and database health monitoring

The system uses:
- **Frontend**: PySide6 (Qt for Python) with a modern dark theme
- **Authentication**: Firebase (Pyrebase4) — email/password login, sign-up, password reset
- **Camera Management & Image Processing**: OpenCV (cv2) for frame capture, resizing, preprocessing, and drawing overlays
- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Recognition**: FaceNet InceptionResnet-V1 with embeddings
- **Backend**: Python with NumPy, Pandas, PyTorch
- **Database**: CSV files with NumPy embeddings (NPZ format)

---

## Core Features

### 1. **Firebase Authentication System**
- Secure email/password login via Firebase Authentication
- New user sign-up with password validation
- Forgot password → email reset link flow
- Persistent login loop: app restarts the login screen cleanly on logout

### 2. **Real-Time Camera Monitoring**
- Multi-camera support with simultaneous stream processing
- Configurable grid layouts (2×2, 1×4, 2×3)
- Individual camera feed enlargement with detailed info
- Live detection overlay showing recognized faces and confidence scores

### 3. **Person Registration System**
- Capture 3 face angles per person (front, left, right) for robust recognition
- Automatic face detection and validation before storing
- Incremental embedding generation for new registrations
- Consistent CSV/image file management

### 4. **Face Recognition Engine**
- Cosine similarity-based face matching against known embeddings
- Confidence threshold filtering (default: 85%)
- Real-time embedding generation for captured faces
- Platform-aware camera backend selection (Windows/Linux/macOS)

### 5. **Database & Dataset Management**
- Automatic dataset consistency validation on startup
- Self-healing mechanism for orphaned CSV rows or missing images
- Comprehensive statistics: person count, embedding count, image counts, database size
- Safe person deletion with cascading removal of all related data

### 6. **System Statistics Dashboard**
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
Login Loop
    ↓
frontend/pages/login_page.py (LoginPage — Firebase Authentication)
    ├── Sign In   → accept()   → MainWindow
    ├── Sign Up   → creates account, prompts login
    └── Forgot PW → sends reset email via Firebase
    ↓
frontend/main_window.py (MainWindow)
    ├── TopBarWidget (Time & system info)
    ├── SidebarWidget (Navigation + Logout button)
    └── QStackedWidget (Page switching)
        ├── DashboardPage (Statistics)
        ├── CameraMonitorPage (Live monitoring)
        │       ↓  [User clicks Load]
        │   ModelLoaderThread (QThread — background CPU core)
        │       ↓  finished_loading signal
        │   MultiCameraManager ready → Start button unlocked
        ├── RegistrationPage (Person registration)
        └── SettingsPage (Configuration)
    ↓
[Logout clicked] → window.logout_var = True → LoginPage re-shown
[Window X closed] → sys.exit(0)
```

### Backend Processing

1. **ModelLoaderThread** (QThread) initializes `MultiCameraManager` in the background when the user clicks Load — MTCNN, InceptionResnet-V1 models, and the face database all load on a separate CPU core, keeping the UI fully responsive during the 30–40 second wait
2. **Dataset Manager** validates all CSV files and image folders on startup
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
│   ├── login_page.py       # Firebase login/signup/reset dialog (NEW)
│   ├── dashboard_page.py   # Statistics & person management (READ-ONLY)
│   ├── camera_page.py      # Multi-camera monitoring, ModelLoaderThread, real-time detection
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
│   ├── login_setup.py              # FirebaseAuth class — wraps Pyrebase4 (NEW)
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

### 0. **Login Page** - Firebase Authentication *(New)*

**Purpose**: Secure gateway to the application. Shown on every fresh start and every logout.

**Layout**:
```
┌─────────────────────────────────────────────────────┐
│              (Dark background #0a0e27)            │
│                                                     │
│         ┌───────────────────────────────┐           │
│         │       System Access           │           │
│         │                               │           │
│         │  Email:    [________________] │           │
│         │  Password: [________________] │           │
│         │                               │           │
│         │  [Status / Error message]     │           │
│         │                               │           │
│         │  [ Login ]     [ Sign Up ]    │           │
│         │                               │           │
│         │       Forgot Password?        │           │
│         └───────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

**Buttons & Actions**:

| Button | Action | Outcome |
|--------|--------|---------|
| **Login** | `FirebaseAuth.sign_in(email, pw)` | Opens MainWindow on success; shows error on failure |
| **Sign Up** | `FirebaseAuth.sign_up(email, pw)` | Creates account; prompts user to log in |
| **Forgot Password?** | `FirebaseAuth.reset_password(email)` | Sends Firebase reset email; confirms on screen |

**Status Messages** (inline, color-coded):
- 🔵 "Authenticating..." / "Creating account..." — progress feedback
- 🟢 "Account created! You can now log in." — success
- 🔴 "Invalid email or password" / "Error creating account." — failure

**Behind the Scenes**:
- `LoginPage` is a `QDialog` — `login.exec()` blocks the main loop until accepted or rejected
- Firebase initialization (`pyrebase.initialize_app`) is deferred by 100ms via `QTimer.singleShot` so the login window renders before any network call
- On successful login: `self.accept()` is called, which signals `main()` to open `MainWindow`
- On window close (X button): `QDialog.Rejected` breaks the login loop and exits the app

---

### 1. **Dashboard Page** - System Overview & Management

**Purpose**: Central hub for monitoring system health and managing person records.

**Layout**:
```
┌────────────────────────────────────────────────────────┐
│  System Statistics & Database Info              [+]    │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 👥           │  │ 🧠           │  │ 🔗           │  │
│  │ Total        │  │ Total        │  │ Unique       │  │
│  │ Persons      │  │ Embeddings   │  │ Embeddings   │  │ 
│  │ [Count]      │  │ [Count]      │  │ [Count]      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 📷           │  │ 👤           │  │ 💾           │  │
│  │ Original     │  │ Face         │  │ Database     │  │
│  │ Images       │  │ Images       │  │ Size         │  │
│  │ [Count]      │  │ [Count]      │  │ [Size]       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │ 
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Registered Persons Table                         │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Sr.No | Name         | ID    | Images | Actions  │  │
│  │ 1     | John Doe     | P001  | 3      | Remove   │  │
│  │ 2     | Sarah Smith  | P002  | 3      | Remove   │  │
│  │ ...                                              │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
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
│ │ CAM 1       │ │  └───────────────────────────┘    │
│ │ (thumbnail) │ │                                   │
│ │             │ │  Camera Details:                  │
│ └─────────────┘ │  Name: Gate A - Entrance          │
│ ┌─────────────┐ │  FPS: 30                          │
│ │ CAM 2       │ │  Status: Active                   │
│ │ (thumbnail) │ │                                   │
│ │             │ │                                   │
│ └─────────────┘ │                                   │
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
- **Load**: Spawns `ModelLoaderThread` on a separate CPU core — shows "⏳ Loading AI Models... Please wait. This may take 30–40 seconds." while MTCNN + InceptionResnet-V1 + face database load in the background. UI stays fully interactive. On completion, emits `finished_loading` signal → shows "✅ AI Models Loaded Successfully! You can now click Start." and unlocks the Start button.
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
- **[Logout]** (bottom): Sets `logout_var = True`, closes `MainWindow`, and returns to the `LoginPage` — AI models stay loaded in memory

**Active State**: Highlighted button shows current page with left cyan border + color change

### Login Buttons

| Button | Action | Purpose |
|--------|--------|---------|
| **Login** | Firebase sign-in | Authenticates with email + password; opens app on success |
| **Sign Up** | Firebase create user | Registers new account; does not auto-login |
| **Forgot Password?** | Firebase reset email | Sends password reset link to entered email |

### Dashboard Buttons

| Button | Action | Purpose |
|--------|--------|---------|
| **Refresh Stats** | Manual refresh | Forces immediate stats recompute from CSV |
| **Remove** (per person) | Delete person + data | Cascading deletion, runs in background thread |

### Camera Monitor Buttons

| Button | Function | Details |
|--------|----------|---------|
| **Load** | Start `ModelLoaderThread` | Loads MTCNN + FaceNet in background (30–40s); UI stays responsive; shows status messages |
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

### 1. **FirebaseAuth - Authentication Backend** *(New)*

**Location**: `backend/src/login_setup.py`

**What it does**: Wraps Pyrebase4 to provide clean methods for the login page.

```python
class FirebaseAuth:
    def __init__(self):
        self.firebase = pyrebase.initialize_app(firebaseConfig)
        self.auth = self.firebase.auth()

    def sign_in(self, email, password) -> user | None
    def sign_up(self, email, password) -> user | None
    def reset_password(self, email) -> bool
```

**Firebase project**: Connected to `surveillance-system-98193` on Firebase.

**Error handling**: All methods catch exceptions and return `None`/`False` on failure — the `LoginPage` translates this into a user-facing message.

---

### 2. **Login Loop & Logout Flow** *(New)*

**Location**: `frontend/main_window.py` → `main()`

**How it works**: `main()` runs a `while` loop — each iteration shows the login dialog. On successful login, `MainWindow` is created and `app.exec()` blocks until the window closes. The loop then checks `logout_var` to decide whether to show the login screen again or exit.

```python
def main():
    app = QApplication(sys.argv)

    logout_var = True
    while logout_var:
        login = LoginPage()
        if login.exec() == QDialog.Accepted:
            logout_var = False
            window = MainWindow()
            window.show()
            app.exec()           # blocks here until window closes
            if window.logout_var:
                logout_var = True  # logout clicked → re-show login
            else:
                break              # X button → exit
        else:
            break                  # X on login screen → exit

    sys.exit(0)
```

**Logout mechanism** (`main_window.py` → `handle_logout`):
```python
def handle_logout(self):
    self.logout_var = True   # signal to main() loop
    self.close()             # causes app.exec() to return
```

**Result**:
- Logout → re-login: shows login screen cleanly with no restart required
- Window X button: clean `sys.exit(0)`

---

### 3. **ModelLoaderThread - Non-Blocking AI Initialization**

**Location**: `frontend/pages/camera_page.py`

**Purpose**: MTCNN + InceptionResnet-V1 take 30–40 seconds to load. Running this on the main thread would freeze the UI. `ModelLoaderThread` runs the entire initialization on a **separate CPU core** via `QThread`, completely bypassing the UI thread and GIL contention.

```python
class ModelLoaderThread(QThread):
    """Loads heavy PyTorch models on a separate CPU core to prevent UI freezing"""
    finished_loading = Signal(object)

    def run(self):
        # Runs on a background thread — UI stays fully responsive
        from backend.src.multi_camera_manager import MultiCameraManager
        manager = MultiCameraManager()          # loads MTCNN + FaceNet + face DB
        self.finished_loading.emit(manager)     # signals UI thread when done
```

**Load button flow**:
```python
def _load_button(self):
    self._show_temporary_warning("⏳ Loading AI Models... Please wait. This may take 30-40 seconds.")
    self.loader_thread = ModelLoaderThread()
    self.loader_thread.finished_loading.connect(self._on_models_loaded)
    self.loader_thread.start()

def _on_models_loaded(self, manager):
    """Runs automatically the exact millisecond PyTorch is ready"""
    self.manager = manager
    self._show_temporary_warning("✅ AI Models Loaded Successfully! You can now click Start.")
```

**Why this works**: `QThread.run()` executes on a different OS thread. PyTorch model loading is largely C++ / native code that releases the GIL — it runs in true parallel with the Qt event loop. The `finished_loading` signal is queued across the thread boundary and safely delivers the manager object to the main thread only after loading is complete.

---

### 4. **DatasetManager - Automatic Consistency Validation**

**Location**: `backend/src/dataset_manager.py`

**Problem Solved**: CSV rows can get orphaned or images can be missing due to failed operations, crashes, or manual edits.

**Solution - Validation Logic**:
```python
def validate_information(self):
    # Check 1: CSV file consistency
    if len(face_info_df) != len(info_df) or len(info_df) != len(embeddings_df):
        raise ValidationError(101)

    # Check 2: Folder consistency
    if len(image_files) != len(info_df):
        raise ValidationError(104)
```

**Auto-Repair Mechanism**:
```python
def fix_issues(self, error_codes):
    if 101 in errors:  # CSV mismatch
        self.face_info_df = self.face_info_df[self.face_info_df['ID'].isin(valid_ids)]

    if 104 in errors:  # Missing images
        for img_file in images/:
            if extract_id(img_file) not in valid_ids:
                os.remove(img_file)
```

**Result**: App starts with guaranteed data consistency — no crashes from file mismatches.

---

### 5. **MultiCameraManager - Parallel Face Recognition**

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
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Drop oldest frame to prevent lag
                    self.frame_queue.get(block=False)
```

**Face Recognition Pipeline**:
```python
for camera_name, camera in self.cameras.items():
    frame = camera.get_frame()
    faces, _ = self.mtcnn.detect(frame)

    if faces is not None:
        for face_box in faces:
            embedding = self.model(preprocess(face_img))
            embedding = embedding / np.linalg.norm(embedding)

            similarities = cosine_similarity([embedding], self.known_embeddings)
            best_match_idx = np.argmax(similarities)
            confidence = similarities[0][best_match_idx] * 100

            if confidence >= THRESHOLD:
                name = self.known_names[best_match_idx]
```

---

### 6. **PersonRegistrationSystem - Incremental Registration**

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

**ID Generation Strategy**:
```python
def get_next_person_id(self):
    max_id = max(int(row['ID'][1:]) for row in csv_data)
    return f"P{(max_id + 1):03d}"  # P001, P002, P003...
```

**Embedding Storage**:
```python
# all_embeddings.npz (binary format)
{ "P001_front": [0.123, 0.456, ...],   # 512-dim vector
  "P001_left":  [0.124, 0.457, ...],
  "P001_right": [0.125, 0.458, ...] }
```

---

### 7. **StatsManager - Read-Only Statistics Computation**

**Location**: `backend/src/stats_manager.py`

**Design**: Never modifies data — only reads and computes metrics for dashboard display.

```python
def get_all_statistics(self):
    return {
        'person_count': len(info_df['ID'].unique()),
        'embeddings_count': len(embeddings_df),
        'unique_embeddings': len(embeddings_df['ID'].unique()),
        'total_images': len(info_df),
        'face_images': len([f for f in faces_dir if is_file(f)]),
        'database_info': { 'total_size': format_bytes(size_sum), ... }
    }
```

---

### 8. **MainWindow - Page Navigation with QStackedWidget**

**Location**: `frontend/main_window.py`

**Pattern Used**: QStackedWidget (efficient page switching without destruction/recreation)

```python
class MainWindow(QMainWindow):
    def __init__(self, camera_manager):       # ← receives pre-loaded manager
        self.pages = {
            "dashboard": DashboardPage(),
            "camera": CameraMonitorPage(camera_manager=camera_manager),  # ← injected
            "registration": RegistrationPage(),
            "settings": SettingsPage(),
        }
        self.stacked_widget = QStackedWidget()
        for key, page in self.pages.items():
            self.stacked_widget.addWidget(page)

    def show_page(self, page_name):
        page_index = list(self.pages.keys()).index(page_name)
        self.stacked_widget.setCurrentIndex(page_index)
```

---

### 9. **Threading for Responsive UI**

**Worker Thread Pattern** (Dashboard → Remove Person):
```python
class RemovePersonWorker(QThread):
    worker_finished = Signal(bool, str)

    def run(self):
        RemovePerson(personID, dataset_path)
        self.worker_finished.emit(True, "Success message")
```

**Result**: UI never freezes — user can continue navigating/viewing stats during deletions.

---

### 10. **OpenCV - Core Image Processing & Frame Capture**

**Location**: `backend/src/multi_camera_manager.py` (CameraStream class)

**Frame Capture**:
```python
def _capture_loop(self):
    while self.is_running:
        ret, frame = self.cap.read()   # NumPy array (480, 640, 3) BGR
        if ret:
            self.last_frame = frame.copy()
            self.frame_queue.put(frame, block=False)
```

**Drawing Detection Overlays**:
```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(frame, f"{name} ({confidence:.1f}%)", (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

**Platform-Aware Backend Selection**:
```python
if system == "Windows":  cap = cv2.VideoCapture(id, cv2.CAP_DSHOW)
elif system == "Linux":  cap = cv2.VideoCapture(id, cv2.CAP_V4L2)
elif system == "Darwin": cap = cv2.VideoCapture(id, cv2.CAP_AVFOUNDATION)
```

---

## Project Structure

```
FaceRecognitionSystem/
│
├── 📄 README.md                          # This file
├── 🐍 run_frontend.py                    # Application entry point
├── 📋 requirements.txt                   # Python dependencies
│
├── 📁 frontend/                          # PySide6 GUI Application
│   ├── 🐍 __init__.py
│   ├── 🐍 main_window.py                 # Main window, login loop, logout handling
│   ├── 🐍 config.py                      # Application configuration (colors, sizes, thresholds)
│   ├── 🐍 utils.py                       # Shared utilities & helpers
│   │
│   ├── 📁 pages/                         # Application screens
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 login_page.py              # Firebase login/signup/reset dialog [NEW]
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
│   └── 📁 styles/
│       └── 🎨 dark_theme.qss             # Qt stylesheet (dark theme)
│
├── 📁 backend/
│   ├── ⚙️ camera_config.ini              # Camera configuration (device IDs, names)
│   │
│   ├── 📁 src/                           # Core business logic
│   │   ├── 🐍 login_setup.py             # FirebaseAuth — Pyrebase4 wrapper [NEW]
│   │   ├── 🐍 dataset_manager.py         # Dataset validation & auto-repair
│   │   ├── 🐍 multi_camera_manager.py    # Multi-camera control & face recognition
│   │   ├── 🐍 person_registration.py     # Person registration workflow
│   │   └── 🐍 stats_manager.py           # Statistics computation (read-only)
│   │
│   └── 📁 dataset/
│       ├── 📄 info.csv                   # Master registry (ID, Name, Image Path)
│       ├── 📄 face_info.csv              # Face metadata (quality, landmarks)
│       ├── 📁 images/                    # Original 3-angle captures
│       ├── 📁 faces/                     # Processed cropped faces (MTCNN output)
│       └── 📁 embeddings/
│           ├── 📦 all_embeddings.npz     # Binary embeddings (512-dim vectors)
│           └── 📄 embeddings.csv         # Embedding metadata (ID, Name, Timestamp)
│
└── 🐍 run_frontend.py                    # Entry point
```

### File Descriptions

**Frontend Core**:
- `main_window.py`: Initializes QMainWindow, builds layout, manages page switching and the login/logout loop
- `config.py`: Centralized configuration (APP_VERSION, colors, thresholds, sizes)
- `utils.py`: Shared functions (image conversion, data formatting, etc.)

**Frontend Pages**:
- `login_page.py`: Firebase authentication dialog — login, sign-up, forgot password *(New)*
- `dashboard_page.py`: Statistics display (read-only), person removal (background thread)
- `camera_page.py`: Multi-camera grid, stream selection, detection overlay
- `registration_page.py`: Guided person enrollment (3-angle capture + commit)
- `settings_page.py`: Camera ID configuration, config file management

**Frontend Widgets**:
- `sidebar.py`: Navigation menu with active button highlighting and logout button
- `topbar.py`: System time display, status indicators
- `cards.py`: Statistics card components with icons and values
- `overlays.py`: Alert boxes, loading spinners, confirmation dialogs

**Backend Core**:
- `login_setup.py`: `FirebaseAuth` class — wraps Pyrebase4 for sign-in, sign-up, password reset *(New)*
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
- **GPU** (CUDA-enabled NVIDIA card recommended; CPU fallback available)
- **Firebase project** with Email/Password authentication enabled

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
- Pyrebase4 (Firebase authentication)
- NumPy, Pandas (data handling)
- Scikit-learn (similarity metrics)

### Step 4: Firebase setup

Create your own Firebase project and add credentials to `backend/src/login_setup.py`:

**Step 1** — Go to [Firebase Console](https://console.firebase.google.com/) → **Add project**

**Step 2** — In your project: **Project Settings → General → Your apps → Add app (Web `</>`)**  
Copy the `firebaseConfig` object shown and paste it into `login_setup.py`:

```python
# REPLACE
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
```
```python
# WITH
self.firebaseConfig = {
    "apiKey": "Your API Key",
    "authDomain": "Your Auth Domain",
    "projectId": "Your Project ID",
    "databaseURL": "Your Database URL",
    "storageBucket": "Your Storage Bucket",
    "messagingSenderId": "Your Messaging Sender ID",
    "appId": "Your App ID",
    "measurementId": "Your Measurement ID"
}
```

**Step 3** — Enable **Authentication → Sign-in method → Email/Password**

**Step 4** — Create a test user via **Authentication → Users → Add user**, or use **Sign Up** in the app

> ⚠️ Never commit real credentials to a public repository.  
> Add `login_setup.py` to `.gitignore`, or use environment variables:
> ```python
> import os
> "apiKey": os.environ.get("FIREBASE_API_KEY")
> ```

### Step 5: Run the Application
```bash
python3 run_frontend.py
```

The app will:
1. Show the login screen immediately
2. Open the main dashboard after successful login
3. AI models load in the background when you click **Load** on the Camera Monitor page (30–40 seconds, UI stays responsive)

### Step 6: Configure Cameras (via Settings Page)

After launching the app:

1. Navigate to **Settings** page
2. Enter your active camera IDs (comma-separated, e.g., `0, 1, 2`) → Click **Update Camera List**
3. (Optional) Assign friendly display names → Click **Update/Overwrite Camera Names**

This populates `backend/camera_config.ini`:
```ini
[Display Names]
0 = Gate A - Entrance
1 = Gate B - Exit
2 = Lobby - Main Hall
```

### Step 7: First-Time Setup

1. **Dashboard**: Verify system statistics load correctly
2. **Camera Monitor**: Click [Load] → wait for "✅ AI Models Loaded" (30–40s) → click [Start]
3. **Registration**: Enroll first person (3-angle capture)
4. **Recognition**: Return to Camera Monitor — newly registered face should be detected

---

## Database Structure

### CSV Format Reference

**info.csv** (Master Registry):
```csv
Sr No.,Name,ID,Image Path
1,John Doe,P001,dataset/images/p1_front.jpeg
2,John Doe,P001,dataset/images/p1_left.jpeg
3,John Doe,P001,dataset/images/p1_right.jpeg
```

**embeddings.csv** (Embedding Metadata):
```csv
ID,Name,Embedding Key,Timestamp
P001,John Doe,P001_front,2024-06-04 10:30:45
P001,John Doe,P001_left,2024-06-04 10:30:46
P001,John Doe,P001_right,2024-06-04 10:30:47
```

**face_info.csv** (Face Quality Metrics):
```csv
ID,Landmarks,Quality Score,Angle,Timestamp
P001,[[x,y],...],0.95,front,2024-06-04 10:30:45
```

---

## Configuration Reference

### ApplicationConfig (`frontend/config.py`)
```python
DEFAULT_CONFIDENCE_THRESHOLD = 85   # % threshold for face match
DEFAULT_FPS = 30
SIDEBAR_WIDTH = 250
TOPBAR_HEIGHT = 60
COLOR_PRIMARY = "#00bfff"
COLOR_BACKGROUND = "#0a0e27"
COLOR_SUCCESS = "#4caf50"
```

### `backend/camera_config.ini`
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
| "Invalid email or password" on login | Wrong credentials or unregistered email | Use Sign Up first, or check Firebase Console → Authentication |
| Login screen hangs briefly | Firebase SDK initializing network connection | Normal on first launch; subsequent logins are faster |
| Load takes 30–40 seconds | MTCNN + FaceNet loading in background thread | Expected — UI stays responsive; wait for the ✅ success message |
| UI freezes during Load | ModelLoaderThread not used | Should not happen; check that `_load_button` starts the thread |
| "No camera found" on Start | Camera not detected by system | Check `ls /dev/video*` on Linux; verify USB connection |
| "Failed to initialize MTCNN" | PyTorch not installed or CUDA error | Run `pip install torch torchvision` or use CPU fallback |
| "CSV file not found" | Dataset folder missing | Extract `dummy_dataset.zip` or create dataset structure |
| Registration hangs | Face detection timeout | Ensure face is clearly visible; try different lighting |
| UI freezes during remove | Main thread blocking | Wait — removal runs in background thread |
| Settings not persisting | Config file permissions | Check write permissions on `backend/camera_config.ini` |
| Password reset email not arriving | Incorrect email or spam filter | Check spam folder; verify email is registered in Firebase |

---

## Performance Tips

1. **GPU Acceleration**: Install CUDA + cuDNN for 10x faster face detection and model loading
2. **Background Loading**: `ModelLoaderThread` keeps the UI fully responsive during the 30–40s model load — navigate other pages while it loads
3. **Multi-Camera**: Limit to 4 simultaneous cameras for optimal performance
4. **Recognition Threshold**: Increase to 90% for stricter matching (fewer false positives)
5. **Resolution**: Reduce camera resolution (480p vs 1080p) for smoother streaming

---

## Future Enhancements

- [ ] Liveness detection (prevent spoofing with photos)
- [ ] Attendance logging with timestamps
- [ ] Export reports (CSV, PDF)
- [ ] Alert system for unauthorized persons
- [ ] Database backup & restoration
- [x] ~~Multi-user authentication~~ *(Done — Firebase Email/Password auth)*
- [ ] REST API for integration with other systems
- [ ] Role-based access control (admin vs viewer)

---

**Last Updated**: June 12, 2025
**Version**: 1.1.0
**Author**: Vivek Avhad
**License**: MIT
