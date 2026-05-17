#!/usr/bin/env python3
"""
Person Registration System - Clean & Simple
Uses existing detect_faces.py and extract_features.py utilities
"""

import os
import csv
import base64
import shutil
import subprocess
import sys
from datetime import datetime
import numpy as np
from PIL import Image
import io

class PersonRegistrationSystem:
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            # Use the actual backend/dataset path
            dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset")
        
        self.dataset_path = os.path.abspath(dataset_path)
        self.images_path = os.path.join(self.dataset_path, "images")
        self.faces_path = os.path.join(self.dataset_path, "faces")
        self.embeddings_path = os.path.join(self.dataset_path, "embeddings")
        
        # CSV files
        self.info_csv = os.path.join(self.dataset_path, "info.csv")
        self.face_info_csv = os.path.join(self.dataset_path, "face_info.csv")
        self.embeddings_csv = os.path.join(self.embeddings_path, "embeddings.csv")
        self.embeddings_npz = os.path.join(self.embeddings_path, "all_embeddings.npz")
        
        # Ensure directories exist
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.faces_path, exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)
    
    def get_next_person_id(self):
        """Get the next available person ID"""
        try:
            if not os.path.exists(self.info_csv):
                return "P001"
            
            with open(self.info_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                max_id = 0
                for row in reader:
                    person_id = row['ID']
                    if person_id.startswith('P'):
                        try:
                            num = int(person_id[1:])
                            max_id = max(max_id, num)
                        except ValueError:
                            continue
                
            return f"P{(max_id + 1):03d}"
        except Exception as e:
            print(f"Error getting next person ID: {e}")
            return "P001"
    
    def get_next_person_number(self):
        """Get the next available person number for image naming"""
        try:
            if not os.path.exists(self.info_csv):
                return 1
            
            with open(self.info_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                max_num = 0
                for row in reader:
                    image_path = row['Image Path']
                    if 'p' in os.path.basename(image_path):
                        try:
                            base_name = os.path.basename(image_path)
                            num_part = base_name.split('_')[0]
                            if num_part.startswith('p'):
                                num_str = num_part[1:]
                                num = int(num_str)
                                max_num = max(max_num, num)
                        except (ValueError, IndexError):
                            continue
                
            return max_num + 1
        except Exception as e:
            print(f"Error getting next person number: {e}")
            return 1
    
    def get_next_sr_number(self):
        """Get the next available serial number for CSV"""
        try:
            if not os.path.exists(self.info_csv):
                return 1
            
            with open(self.info_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                max_sr = 0
                for row in reader:
                    try:
                        sr_num = int(row['Sr No.'])
                        max_sr = max(max_sr, sr_num)
                    except (ValueError, KeyError):
                        continue
                
            return max_sr + 1
        except Exception as e:
            print(f"Error getting next serial number: {e}")
            return 1
    
    def save_images(self, person_name, person_id, images_data):
        """Save the three captured images"""
        try:
            person_num = self.get_next_person_number()
            saved_paths = {}
            
            angles = ['front', 'left', 'right']
            
            for angle in angles:
                if angle not in images_data:
                    return {"success": False, "error": f"Missing {angle} image"}
                
                filename = f"p{person_num}_{angle}.jpeg"
                filepath = os.path.join(self.images_path, filename)
                
                # Save image data
                image_data = images_data[angle]
                if isinstance(image_data, str):
                    # Handle base64 data
                    if image_data.startswith('data:image/'):
                        image_data = image_data.split(',')[1]
                    
                    # Decode and save
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    image.save(filepath, 'JPEG', quality=85)
                
                # Use relative path for CSV storage (like existing data)
                saved_paths[angle] = f"dataset/images/{filename}"
            
            return {
                "success": True,
                "paths": saved_paths,
                "person_number": person_num
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error saving images: {str(e)}"}
    
    def update_info_csv(self, person_name, person_id, image_paths):
        """Update info.csv with new person data"""
        try:
            next_sr = self.get_next_sr_number()
            
            new_rows = []
            for angle in ['front', 'left', 'right']:
                new_rows.append([next_sr, person_name, person_id, image_paths[angle]])
                next_sr += 1
            
            file_exists = os.path.exists(self.info_csv)
            
            with open(self.info_csv, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Sr No.', 'Name', 'ID', 'Image Path'])
                writer.writerows(new_rows)
            
            return {"success": True, "rows_added": len(new_rows)}
            
        except Exception as e:
            return {"success": False, "error": f"Error updating info.csv: {str(e)}"}
    
    def run_face_detection_incremental(self, person_name, person_id, image_paths):
        """Run face detection only for new person (incremental)"""
        try:
            print(f"🔍 Running face detection for {person_id}...")
            
            from facenet_pytorch import MTCNN
            import cv2
            from PIL import Image
            
            # Initialize MTCNN
            mtcnn = MTCNN(keep_all=False, device='cpu', post_process=True, image_size=160)
            
            new_face_rows = []
            
            for angle in ['front', 'left', 'right']:
                img_path = image_paths[angle]
                
                # Convert relative path to absolute
                if not os.path.isabs(img_path):
                    abs_img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), img_path)
                else:
                    abs_img_path = img_path
                
                if not os.path.exists(abs_img_path):
                    return {"success": False, "error": f"Image not found: {abs_img_path}"}
                
                # Process image
                pil_img = Image.open(abs_img_path).convert('RGB')
                face_tensor = mtcnn(pil_img)
                
                if face_tensor is None:
                    return {"success": False, "error": f"No face detected in {angle} image"}
                
                # Convert tensor to numpy and save
                face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
                face_np = ((face_np + 1) * 127.5).astype(np.uint8)
                face_np = cv2.resize(face_np, (160, 160))
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                
                # Save face
                base_name = os.path.splitext(os.path.basename(abs_img_path))[0]
                save_name = f"{person_id}_{base_name}.jpg"
                save_path = os.path.join(self.faces_path, save_name)
                cv2.imwrite(save_path, face_bgr)
                
                print(f"✅ Saved face: {save_name}")
                
                # Get next SR number
                if len(new_face_rows) == 0:
                    # First face for this person, get next SR from existing file
                    sr_num = 1
                    if os.path.exists(self.face_info_csv):
                        with open(self.face_info_csv, 'r', encoding='utf-8') as f:
                            sr_num = len(f.readlines())  # This includes header, so next is correct
                else:
                    sr_num = new_face_rows[-1][0] + 1
                
                # Person name is passed as parameter
                
                # Add to face_info.csv entries
                new_face_rows.append([sr_num, person_name, person_id, save_path])
            
            # Append to face_info.csv
            file_exists = os.path.exists(self.face_info_csv)
            with open(self.face_info_csv, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Sr No.', 'Name', 'ID', 'Image Path'])
                writer.writerows(new_face_rows)
            
            return {"success": True, "faces_created": len(new_face_rows)}
            
        except Exception as e:
            return {"success": False, "error": f"Face detection error: {str(e)}"}
    
    def run_feature_extraction_incremental(self, person_id):
        """Run feature extraction only for new person (incremental)"""
        try:
            print(f"🧠 Running feature extraction for {person_id}...")
            
            from facenet_pytorch import InceptionResnetV1
            import torch
            from PIL import Image
            
            # Load FaceNet model
            model = InceptionResnetV1(pretrained='vggface2').eval()
            
            # Find new face files for this person
            face_files = []
            for face_file in os.listdir(self.faces_path):
                if face_file.startswith(person_id) and face_file.endswith('.jpg'):
                    face_files.append(face_file)
            
            if not face_files:
                return {"success": False, "error": f"No face files found for {person_id}"}
            
            new_embedding_rows = []
            new_embeddings = {}
            
            # Load existing embeddings
            if os.path.exists(self.embeddings_npz):
                existing_data = np.load(self.embeddings_npz)
                new_embeddings.update({key: existing_data[key] for key in existing_data.keys()})
            
            sr_num = 1
            if os.path.exists(self.embeddings_csv):
                with open(self.embeddings_csv, 'r', encoding='utf-8') as f:
                    sr_num = len(f.readlines())  # Include header
            
            for face_file in sorted(face_files):
                face_path = os.path.join(self.faces_path, face_file)
                
                # Load and process image
                img = Image.open(face_path).convert("RGB")
                img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                
                # Resize if needed
                if img_tensor.shape[1:] != (160, 160):
                    img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(160, 160))[0]
                
                # Normalize
                img_tensor = (img_tensor - 0.5) / 0.5
                img_tensor = img_tensor.unsqueeze(0)
                
                # Extract embedding
                with torch.no_grad():
                    embedding = model(img_tensor).squeeze().numpy()
                
                # Create key
                key = os.path.splitext(face_file)[0]  # P018_p18_front
                new_embeddings[key] = embedding
                
                # Add to CSV row (get name from face_info.csv)
                person_name = "Unknown"
                if os.path.exists(self.face_info_csv):
                    with open(self.face_info_csv, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row['ID'] == person_id:
                                person_name = row['Name']
                                break
                
                new_embedding_rows.append([sr_num, person_name, person_id, face_path, key])
                sr_num += 1
                
                print(f"✅ Extracted embedding: {key}")
            
            # Append to embeddings.csv
            file_exists = os.path.exists(self.embeddings_csv)
            with open(self.embeddings_csv, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Sr No.', 'Name', 'ID', 'Image Path', 'Embedding Key'])
                writer.writerows(new_embedding_rows)
            
            # Save updated NPZ
            np.savez(self.embeddings_npz, **new_embeddings)
            
            return {"success": True, "embeddings_created": len(new_embedding_rows)}
            
        except Exception as e:
            return {"success": False, "error": f"Feature extraction error: {str(e)}"}
    
    def verify_data_consistency(self):
        """Verify that all data files are consistent"""
        print("🔍 Verifying data consistency...")
        
        report = {"consistent": True, "issues": [], "counts": {}}
        
        try:
            # Count entries in each file
            counts = {}
            
            # Count CSV entries (subtract header)
            for name, path in [
                ("info_csv", self.info_csv),
                ("face_info_csv", self.face_info_csv),
                ("embeddings_csv", self.embeddings_csv)
            ]:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        counts[name] = len(f.readlines()) - 1
                else:
                    counts[name] = 0
            
            # Count files
            for name, path in [
                ("images", self.images_path),
                ("faces", self.faces_path)
            ]:
                if os.path.exists(path):
                    counts[name] = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                else:
                    counts[name] = 0
            
            # Count NPZ embeddings
            if os.path.exists(self.embeddings_npz):
                data = np.load(self.embeddings_npz)
                counts["npz_embeddings"] = len(data.keys())
            else:
                counts["npz_embeddings"] = 0
            
            report["counts"] = counts
            
            # Check consistency (use info_csv as baseline)
            expected = counts["info_csv"]
            
            for key, value in counts.items():
                if key != "info_csv" and value != expected:
                    report["consistent"] = False
                    report["issues"].append(f"{key} has {value} entries, expected {expected}")
            
            if report["consistent"]:
                print("✅ All data files are consistent!")
            else:
                print("⚠️  Data inconsistencies found:")
                for issue in report["issues"]:
                    print(f"   - {issue}")
            
            return report
            
        except Exception as e:
            report["consistent"] = False
            report["error"] = str(e)
            return report
    
    def backup_files(self):
        """Create backups of all important files"""
        backups = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_to_backup = [
            self.info_csv,
            self.face_info_csv, 
            self.embeddings_csv,
            self.embeddings_npz
        ]
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                # Read file content
                if file_path.endswith('.npz'):
                    data = np.load(file_path)
                    backups[file_path] = {key: data[key] for key in data.keys()}
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        backups[file_path] = f.read()
            else:
                backups[file_path] = None
        
        return backups
    
    def restore_from_backup(self, backups, saved_files=None):
        """Restore files from backup"""
        try:
            print("🔄 Restoring from backup...")
            
            # Remove any files that were created
            if saved_files:
                if 'image_paths' in saved_files:
                    for path in saved_files['image_paths'].values():
                        abs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
                        if os.path.exists(abs_path):
                            os.remove(abs_path)
                            print(f"🗑️  Removed: {abs_path}")
            
            # Restore CSV files
            for file_path, content in backups.items():
                if file_path.endswith('.npz'):
                    if content is not None:
                        np.savez(file_path, **content)
                        print(f"🔄 Restored: {file_path}")
                else:
                    if content is not None:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"🔄 Restored: {file_path}")
            
            print("✅ Backup restoration completed")
            
        except Exception as e:
            print(f"⚠️  Warning: Backup restoration failed: {e}")
    
    def register_person(self, person_name, images_data):
        """
        Complete registration workflow using existing utilities
        """
        backups = None
        saved_files = {}
        
        try:
            print(f"🚀 Starting registration for {person_name}...")
            
            # Step 1: Create backups
            print("💾 Creating backups...")
            backups = self.backup_files()
            
            # Step 2: Get person ID
            person_id = self.get_next_person_id()
            print(f"🆔 Generated person ID: {person_id}")
            
            # Step 3: Save images and update info.csv
            print("📸 Saving images...")
            save_result = self.save_images(person_name, person_id, images_data)
            if not save_result["success"]:
                return save_result
            
            saved_files['image_paths'] = save_result["paths"]
            
            print("📝 Updating info.csv...")
            info_result = self.update_info_csv(person_name, person_id, save_result["paths"])
            if not info_result["success"]:
                self.restore_from_backup(backups, saved_files)
                return info_result
            
            # Step 4: Run face detection for new person only (fast!)
            face_result = self.run_face_detection_incremental(person_name, person_id, save_result["paths"])
            if not face_result["success"]:
                print("❌ Face detection failed")
                self.restore_from_backup(backups, saved_files)
                return face_result
            
            # Step 5: Run feature extraction for new person only (fast!)
            feature_result = self.run_feature_extraction_incremental(person_id)
            if not feature_result["success"]:
                print("❌ Feature extraction failed")
                self.restore_from_backup(backups, saved_files)
                return feature_result
            
            # Step 6: Verify consistency
            print("🔍 Verifying consistency...")
            consistency = self.verify_data_consistency()
            if not consistency["consistent"]:
                print("❌ Consistency check failed")
                self.restore_from_backup(backups, saved_files)
                return {
                    "success": False,
                    "error": "Data consistency check failed",
                    "issues": consistency["issues"]
                }
            
            print(f"🎉 Registration completed successfully!")
            
            return {
                "success": True,
                "person_id": person_id,
                "person_name": person_name,
                "message": f"Successfully registered {person_name} with ID {person_id}",
                "consistency_report": consistency
            }
            
        except Exception as e:
            print(f"💥 Unexpected error: {str(e)}")
            if backups:
                self.restore_from_backup(backups, saved_files)
            
            return {
                "success": False,
                "error": f"Registration failed: {str(e)}"
            }


# For testing
if __name__ == "__main__":
    registration_system = PersonRegistrationSystem()
    print("Clean Person Registration System initialized!")
    print(f"Next person ID: {registration_system.get_next_person_id()}")