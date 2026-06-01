import numpy as np
import pandas as pd
import os
import shutil
import csv
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    #force=True
)
logger = logging.getLogger(__name__)

ERRORS = {
    99: "Unable to locate issue. Maybe duplication has occurred.",
    100: "Everything is consistent.",
    101: "Some face_ids in face_info.csv and info.csv are missing or extra.",
    102: "Some face_ids in face_info.csv and embeddings.csv are missing or extra.",
    103: "Number of embeddings does not match number of face_ids in face_info.csv.",
    104: "face images are missing in the faces directory.",
    105: "face images are missing in the images directory.",
}

class DatasetManager:
    def __init__(self, dataset_path=None):
        # Resolve dataset path relative to this module if not provided
        if dataset_path is None:
            dataset_path = Path(__file__).resolve().parents[1] / 'dataset'
        self.dataset_path = Path(dataset_path)
        self.faces_dir = self.dataset_path / 'faces'
        self.images_dir = self.dataset_path / 'images'
        self.embeddings_dir = self.dataset_path / 'embeddings'
        
        # CSV file paths
        self.face_info_csv = os.path.join(dataset_path, 'face_info.csv')
        self.info_csv = os.path.join(dataset_path, 'info.csv')
        self.embeddings_csv = os.path.join(self.embeddings_dir, 'embeddings.csv')
        self.all_embeddings_npz = os.path.join(self.embeddings_dir, 'all_embeddings.npz')

        # Check if all required files exist
        required_files = [self.face_info_csv, self.info_csv, self.embeddings_csv, self.all_embeddings_npz]
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Required file not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Dataframes to hold CSV data
        self.embeddings = {}
        self.face_info_df = pd.read_csv(self.face_info_csv)
        self.info_df = pd.read_csv(self.info_csv)
        self.embeddings_df = pd.read_csv(self.embeddings_csv)

        self.embeddings_to_dict()
        self.warnings = self.validate_information()
        
        if 100 not in self.warnings:
            self.fix_issues(self.warnings)

        #return 'DatasetManager initialized successfully with consistent dataset state.'

    def embeddings_to_dict(self):
        """Convert embeddings from CSV/NPZ files to dictionary format."""
        try:
            if os.path.exists(self.all_embeddings_npz):
                npz_data = np.load(self.all_embeddings_npz)
                self.embeddings = {key: npz_data[key] for key in npz_data.keys()}
                self.embedding_count = len(self.embeddings)
                logger.info(f"Loaded {self.embedding_count} embeddings from NPZ file")
            else:
                self.embeddings = {}
                self.embedding_count = 0
                logger.warning("No embeddings NPZ file found")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            self.embeddings = {}
            self.embedding_count = 0

    def validate_information(self):
        errors = []
        
        # 1. Check CSV lengths and subsets
        if len(self.face_info_df) != len(self.info_df) or len(self.face_info_df) != len(self.embeddings_df):
            logger.error("CSV files have different number of rows.")
            errors.extend(self.check_issues()) # Add all CSV errors
        else:
            logger.info("CSV files have the same number of rows.")

        # 2. ALWAYS check folders, regardless of CSV status
        folder_status = self.count_folder_contents()
        if folder_status != 100:
            errors.extend(folder_status)
        else:
            logger.info("Folder contents are consistent with CSV files.")

        return errors if errors else [100]
    
    def check_issues(self):
        csv_errors = []
        face_info_set = set(self.face_info_df['ID'])
        info_set = set(self.info_df['ID'])
        face_ids_embeddings_set = set(self.embeddings_df['ID'])

        if not face_info_set.issubset(info_set) or not info_set.issubset(face_info_set):
            logger.error("Some face_ids in face_info.csv and info.csv are missing or extra.")
            csv_errors.append(101)
        
        if not info_set.issubset(face_ids_embeddings_set) or not face_ids_embeddings_set.issubset(info_set):
            logger.error("Some face_ids in face_info.csv and embeddings.csv are missing or extra.")
            csv_errors.append(102)
        
        if not self.embedding_count == len(info_set):
            logger.error("Number of embeddings does not match number of ids in info.csv.")
            csv_errors.append(103)
        
        return csv_errors

    def count_folder_contents(self):
        file_errors = []
        face_images = os.listdir(self.faces_dir)
        images = os.listdir(self.images_dir)

        if len(face_images) != len(self.info_df):
            logger.error("Number of face images does not match number of ids in info.csv.")
            file_errors.append(104)

        if len(images) != len(self.info_df):
            logger.error("Number of images does not match number of ids in info.csv.")
            file_errors.append(105)

        return file_errors if file_errors else 100

    
    def fix_issues(self, error_codes):
            # Force IDs to strings to prevent int/str mismatch bugs
            valid_ids = self.info_df['ID'].astype(str).values 

            if 101 in error_codes:
                # Safely cast ID column to string before comparing
                self.face_info_df['ID'] = self.face_info_df['ID'].astype(str)
                
                # Keep only the rows where the ID is in the valid_ids list
                original_len = len(self.face_info_df)
                self.face_info_df = self.face_info_df[self.face_info_df['ID'].isin(valid_ids)]
                
                self.face_info_df.to_csv(self.face_info_csv, index=False)
                rows_removed = original_len - len(self.face_info_df)
                logger.info(f"Fixed face_info.csv. Removed {rows_removed} orphaned rows.")

            if 102 in error_codes:
                # Safely cast ID column to string before comparing
                self.embeddings_df['ID'] = self.embeddings_df['ID'].astype(str)
                
                # Keep only the rows where the ID is in the valid_ids list
                original_len = len(self.embeddings_df)
                self.embeddings_df = self.embeddings_df[self.embeddings_df['ID'].isin(valid_ids)]
                
                self.embeddings_df.to_csv(self.embeddings_csv, index=False)
                rows_removed = original_len - len(self.embeddings_df)
                logger.info(f"Fixed embeddings.csv. Removed {rows_removed} orphaned rows.")

            if 103 in error_codes:
                self.fix_embedding_issues()
                logger.info("Fixed embedding issues.")

            if 104 in error_codes or 105 in error_codes:
                self.fix_folder_issues()
                logger.info("Fixed folder issues.")

    def fix_folder_issues(self):
        face_images_folder = os.listdir(self.faces_dir)
        images_folder = os.listdir(self.images_dir)
        IDs_in_info = set(self.info_df['ID'].values)

        try:
            for files in face_images_folder:
                person_id = files.split('_')[0] # Assuming ID format is something like "P001_front"
                if person_id not in IDs_in_info:
                    os.remove(os.path.join(self.faces_dir, files))
                    logger.info(f"Removed extra face image: {files}")

            for files in images_folder:
                person_id = files.split('_')[0] # Assuming ID format is something like "P001_front"
                if person_id not in IDs_in_info:
                    os.remove(os.path.join(self.images_dir, files))
                    logger.info(f"Removed extra image: {files}")

        except Exception as e:
            logger.error(f"Error fixing folder issues: {str(e)}")


    def fix_embedding_issues(self):
        IDs_in_info = set(self.info_df['ID'].values)
        for key in list(self.embeddings.keys()):
            new_key = key.split('_')[0]
            if new_key not in IDs_in_info:
                del self.embeddings[key]
                logger.info(f"Removed embedding for {key} from dictionary.")
        self.embedding_count = len(self.embeddings)

        if self.embedding_count != len(self.info_df):
            logger.warning("After fixing embedding issues, the number of embeddings still does not match the number of ids in info.csv.")
        else:
            logger.info(f"Converting fixed embeddings dictionary back to NPZ file.")
            logger.info(f"Total embeddings after fix: {self.embedding_count}")
            self.dict_to_npz()
    
    def dict_to_npz(self):
        """Convert embeddings dictionary back to NPZ file."""
        np.savez(self.all_embeddings_npz, **self.embeddings)
        logger.info(f"Converted embeddings dict to NPZ file.")

class RemovePerson(DatasetManager):
    def __init__(self, personID=None, dataset_path=None):
        if dataset_path is None:
            dataset_path = Path(__file__).resolve().parents[1] / 'dataset'
        else:
            dataset_path = Path(dataset_path)

        info_path = Path(dataset_path) / 'info.csv'
        self.info = pd.read_csv(info_path)
        self.personID = personID
        self.dataset_path = Path(dataset_path)

        if self.personID not in self.info['ID'].values:
            logger.error(f"{self.personID} not found in the dataset.")
            raise ValueError(f"{self.personID} not found in the dataset.")

        self.info = self.info[self.info['ID'] != self.personID]
        self.info.to_csv(info_path, index=False)

        # Re-run dataset manager initialization to validate and fix dataset state
        super().__init__(dataset_path)


if __name__ == "__main__":
    #DatasetManager()
    RemovePerson(personID="P002")
