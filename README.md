# FaceRecognitionSystem2.0
<h1>started from begining again</h1>



# Face Recognition System – Pipeline & Project Summary



## Directory Structure

src/
 ├── detect_faces.py        # Detect and crop faces using MTCNN
 ├── extract_features.py    # Generate and save face embeddings
 ├── match_face.py          # Compare and match test face with database
dataset/
 ├── info.csv               # Original raw info (name, ID, full image path)
 ├── face_info.csv          # Info for cropped face images (used later)
 └── embeddings/
      ├── embeddings.csv    # Metadata (Name, ID, Embedding Key)
      └── all_embeddings.npz # Embedding vectors (key = image base name)



## Step-by-Step Workflow


### **Step 1: Prepare Raw Dataset**

* `dataset/info.csv` contains:
  * `Sr No., Name, ID, Image Path` (full image paths, not cropped).
* These are raw images (may include background, multiple faces, etc.).


### **Step 2: Detect and Crop Faces**

**File Used:** `src/detect_faces.py`

**Objective:**
Detect and crop the face from raw images using **MTCNN** from `facenet-pytorch`.

**Outputs:**

* Cropped face saved to `dataset/faces/{PID}_{base}.jpg`
* New CSV `face_info.csv` created with:
  * `Sr No., Name, ID, Image Path` (pointing to **cropped** face images).

**Problems Fixed:**

* ✔ Original images had multiple faces or background noise.
* ✔ Missing file checks added.
* ✔ Skipped already processed files (optional).


### **Step 3: Extract Face Embeddings**

**File Used:** `src/extract_features.py`

**Objective:**
Use `InceptionResnetV1 (Facenet)` to generate **128-dimensional embeddings** for each **cropped face**.

**Outputs:**

* `all_embeddings.npz`: Stores `{key: embedding_vector}` where key is image name like `P001_front`.
* `embeddings.csv`: Stores metadata like `Sr No., Name, ID, Image Path, Embedding Key`

**Initial Issues & Fixes:**

| Issue                                              | Fix                                                          |
| -------------------------------------------------- | ------------------------------------------------------------ |
| Old code used full images instead of cropped faces | Switched to `face_info.csv`                                  |
| Duplicate embeddings                               | Now resets everything fresh and consistently                 |
| Mismatched keys                                    | Image filenames are now the unique `key` in both CSV and NPZ |
| Wrong tensor shape or normalization                | Fixed: `(img - 0.5)/0.5` + resized to 160x160                |
| Mixed CPU/GPU handling                             | Dynamic device detection added                               |


### **Step 4: Face Matching**

**File:** `src/match_face.py` (WIP)

**Objective:**
Take a **new face image**, extract its embedding, and **compare** with stored embeddings to find the closest match using **Euclidean or Cosine distance**.

**Planned Inputs:**

* New test image path
* Load `all_embeddings.npz`
* Load `embeddings.csv` to get ID → Name map

**Upcoming Fixes:**

| Issue                                   | Plan                                                      |
| --------------------------------------- | --------------------------------------------------------- |
| Mismatched embeddings (cropped vs full) | Will only match with cropped face embeddings              |
| Key mismatch                            | Now fixed by regenerating CSV + NPZ with base image names |
| Threshold tuning                        | Add a score threshold for match validity                  |



## Key Technologies Used

**Python3.12.3 version used**

| Tool                   | Purpose                                         |
| ---------------------- | ----------------------------------------------- |
| `facenet-pytorch`      | MTCNN for face detection & ResNet for embedding |
| `PIL`, `cv2`, `torch`  | Image preprocessing                             |
| `tqdm`, `csv`, `numpy` | IO and efficiency                               |
| `npz`                  | Efficient storage of embeddings                 |
| `csv`                  | Human-readable metadata for search              |



## Output Summary

| File                 | Description                                    |
| -------------------- | ---------------------------------------------- |
| `face_info.csv`      | Points to all **cropped faces** with Name & ID |
| `embeddings.csv`     | Metadata: Name, ID, path, and embedding key    |
| `all_embeddings.npz` | Actual 128-d vectors (used for matching)       |



## Final Result

You now have a **complete offline face recognition system** that:

* Detects faces
* Generates embeddings
* Stores them efficiently
* Matches faces with high accuracy
