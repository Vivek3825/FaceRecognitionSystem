"""
Statistics Manager - Provides comprehensive system statistics
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import configparser

logger = logging.getLogger(__name__)

class StatsManager:
    """Manages and provides system statistics (Read-Only)"""
    
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            # Defaults to assuming this file is in backend/src/
            dataset_path = Path(__file__).resolve().parents[1] / 'dataset'
        self.dataset_path = Path(dataset_path)
        
        self.faces_dir = self.dataset_path / 'faces'
        self.images_dir = self.dataset_path / 'images'
        self.embeddings_dir = self.dataset_path / 'embeddings'
        
        # CSV file paths
        self.face_info_csv = self.dataset_path / 'face_info.csv'
        self.info_csv = self.dataset_path / 'info.csv'
        self.embeddings_csv = self.embeddings_dir / 'embeddings.csv'
        self.all_embeddings_npz = self.embeddings_dir / 'all_embeddings.npz'
    
    def get_person_count(self):
        """Get total count of unique persons"""
        try:
            if self.info_csv.exists():
                df = pd.read_csv(self.info_csv)
                return len(df['ID'].astype(str).unique())
            return 0
        except Exception as e:
            logger.error(f"Error getting person count: {str(e)}")
            return 0
    
    def get_all_persons(self):
        """Get all persons with their details as a DataFrame"""
        try:
            if self.info_csv.exists():
                return pd.read_csv(self.info_csv)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting all persons: {str(e)}")
            return pd.DataFrame()
    
    def get_embeddings_count(self):
        """Get total count of embeddings rows"""
        try:
            if self.embeddings_csv.exists():
                df = pd.read_csv(self.embeddings_csv)
                return len(df)
            return 0
        except Exception as e:
            logger.error(f"Error getting embeddings count: {str(e)}")
            return 0
    
    def get_unique_embeddings_count(self):
        """Get count of unique embeddings (by person ID)"""
        try:
            if self.embeddings_csv.exists():
                df = pd.read_csv(self.embeddings_csv)
                return len(df['ID'].astype(str).unique())
            return 0
        except Exception as e:
            logger.error(f"Error getting unique embeddings count: {str(e)}")
            return 0
    
    def get_total_images_count(self):
        """Get total count of original images"""
        try:
            if self.info_csv.exists():
                df = pd.read_csv(self.info_csv)
                return len(df)
            return 0
        except Exception as e:
            logger.error(f"Error getting total images count: {str(e)}")
            return 0
    
    def get_face_images_count(self):
        """Get total count of processed face images"""
        try:
            if self.faces_dir.exists():
                return len([f for f in os.listdir(self.faces_dir) if os.path.isfile(os.path.join(self.faces_dir, f))])
            return 0
        except Exception as e:
            logger.error(f"Error getting face images count: {str(e)}")
            return 0
    
    def get_database_info(self):
        """Get comprehensive database information including file sizes"""
        info = {
            'dataset_path': str(self.dataset_path),
            'folders': {},
            'files': {},
            'total_size': "0 B"
        }
        
        try:
            if self.dataset_path.exists():
                info['dataset_exists'] = True
                info['last_modified'] = datetime.fromtimestamp(
                    os.path.getmtime(self.dataset_path)
                ).strftime("%Y-%m-%d %H:%M:%S")
            
            # Helper to calculate folder size
            def process_folder(folder_path, name):
                if folder_path.exists():
                    files = [f for f in os.listdir(folder_path) if (folder_path / f).is_file()]
                    size = sum(os.path.getsize(folder_path / f) for f in files)
                    info['folders'][name] = {
                        'count': len(files),
                        'size': self._format_size(size),
                        'size_bytes': size
                    }

            process_folder(self.faces_dir, 'faces')
            process_folder(self.images_dir, 'images')
            process_folder(self.embeddings_dir, 'embeddings')
            
            # NPZ file details
            if self.all_embeddings_npz.exists():
                npz_size = os.path.getsize(self.all_embeddings_npz)
                npz_data = np.load(self.all_embeddings_npz)
                info['files']['all_embeddings.npz'] = {
                    'size': self._format_size(npz_size),
                    'size_bytes': npz_size,
                    'arrays': len(npz_data.files)
                }
            
            # CSV files
            for csv_file, name in [
                (self.info_csv, 'info.csv'),
                (self.face_info_csv, 'face_info.csv'),
                (self.embeddings_csv, 'embeddings.csv')
            ]:
                if csv_file.exists():
                    csv_size = os.path.getsize(csv_file)
                    try:
                        df = pd.read_csv(csv_file)
                        rows = len(df)
                    except pd.errors.EmptyDataError:
                        rows = 0
                        
                    info['files'][name] = {
                        'size': self._format_size(csv_size),
                        'size_bytes': csv_size,
                        'rows': rows
                    }
            
            # Calculate total size securely
            total_bytes = sum(f.get('size_bytes', 0) for f in info['files'].values()) + \
                          sum(f.get('size_bytes', 0) for f in info['folders'].values())
            info['total_size'] = self._format_size(total_bytes)
            
        except Exception as e:
            logger.error(f"Error getting database info: {str(e)}")
        
        return info
    
    def _format_size(self, bytes_size):
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} TB"
    
    def get_available_cameras(self):
        """Get available cameras from config file"""
        cameras = {}
        try:
            config_file = Path(__file__).resolve().parents[1] / "camera_config.ini"
            
            if config_file.exists():
                config = configparser.ConfigParser()
                config.read(str(config_file))
                
                if 'Display Names' in config:
                    for camera_id, camera_name in config['Display Names'].items():
                        cameras[int(camera_id)] = {
                            'name': camera_name,
                            'id': int(camera_id),
                            'status': 'Available'
                        }
            
            if not cameras:
                cameras[0] = {'name': 'Default Camera', 'id': 0, 'status': 'Available'}
                
        except Exception as e:
            logger.error(f"Error getting available cameras: {str(e)}")
            cameras[0] = {'name': 'Default Camera (Fallback)', 'id': 0, 'status': 'Error'}
        
        return cameras
    
    def get_data_quality_report(self):
        """Get comprehensive data quality and consistency report"""
        report = {
            'status': 'GOOD',
            'issues': [],
            'warnings': [],
            'consistency_checks': {}
        }
        
        try:
            # Load data safely
            info_df = pd.read_csv(self.info_csv) if self.info_csv.exists() else pd.DataFrame()
            face_info_df = pd.read_csv(self.face_info_csv) if self.face_info_csv.exists() else pd.DataFrame()
            embeddings_df = pd.read_csv(self.embeddings_csv) if self.embeddings_csv.exists() else pd.DataFrame()
            
            if info_df.empty or face_info_df.empty or embeddings_df.empty:
                report['status'] = 'WARNING'
                report['warnings'].append("One or more CSV files are completely empty or missing.")
                return report
            
            info_count = len(info_df)
            report['consistency_checks']['info_csv_rows'] = info_count
            report['consistency_checks']['face_info_csv_rows'] = len(face_info_df)
            report['consistency_checks']['embeddings_csv_rows'] = len(embeddings_df)
            
            # IDs cast to string to prevent mismatch errors
            info_ids = set(info_df['ID'].astype(str))
            face_info_ids = set(face_info_df['ID'].astype(str))
            embeddings_ids = set(embeddings_df['ID'].astype(str))
            
            if info_ids != face_info_ids or info_ids != embeddings_ids:
                report['status'] = 'WARNING'
                report['warnings'].append("ID mismatch detected across CSV files. DatasetManager should be run to fix this.")
            
            # Check file counts
            face_files = [f for f in os.listdir(self.faces_dir) if f.endswith('.jpg') or f.endswith('.png')] if self.faces_dir.exists() else []
            report['consistency_checks']['face_files_count'] = len(face_files)
            
            if len(face_files) != info_count:
                report['status'] = 'WARNING'
                report['warnings'].append(f"Folder image count ({len(face_files)}) does not match DB person count ({info_count}).")
            
            if not report['warnings'] and not report['issues']:
                report['message'] = 'All data consistency checks passed!'
            
        except Exception as e:
            report['status'] = 'ERROR'
            report['issues'].append(f"Error during quality check: {str(e)}")
        
        return report
    
    def get_all_statistics(self):
        """Get all statistics in one call to populate the Dashboard"""
        # Note: We call .to_dict('records') on the DataFrame here so the UI gets clean dictionaries
        return {
            'person_count': self.get_person_count(),
            'embeddings_count': self.get_embeddings_count(),
            'unique_embeddings': self.get_unique_embeddings_count(),
            'total_images': self.get_total_images_count(),
            'face_images': self.get_face_images_count(),
            'database_info': self.get_database_info(),
            'all_persons': self.get_all_persons().to_dict('records'),
            'available_cameras': self.get_available_cameras(),
            'data_quality': self.get_data_quality_report()
        }