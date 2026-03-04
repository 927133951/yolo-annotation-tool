import os
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from config import ProjectConfig, ConfigManager


class ProjectManager:
    def __init__(self, base_dir: str = None):
        self.config_manager = ConfigManager(base_dir)
        self.current_project: Optional[ProjectConfig] = None
        
    def create_project_structure(self, project_name: str, classes: List[str]) -> ProjectConfig:
        project_dir = self.config_manager.projects_dir / project_name
        
        if project_dir.exists():
            raise ValueError(f"项目 '{project_name}' 已存在")
        
        folders = [
            "images/original",
            "images/train",
            "images/val",
            "labels/original",
            "labels/train",
            "labels/val",
            "models",
            "models/onnx",
            "logs",
            "reports",
            "exports"
        ]
        
        for folder in folders:
            folder_path = project_dir / folder
            folder_path.mkdir(parents=True, exist_ok=True)
        
        config = self.config_manager.create_project_config(
            project_name=project_name,
            classes=classes,
            model_type='yolov8',
            model_size='n'
        )
        
        self.config_manager.save_project_config(config)
        
        classes_file = project_dir / "classes.txt"
        with open(classes_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(classes))
        
        self.current_project = config
        return config
    
    def load_project(self, project_name: str) -> ProjectConfig:
        project_dir = self.config_manager.projects_dir / project_name
        self.current_project = self.config_manager.load_project_config(project_dir)
        return self.current_project
    
    def get_project_structure(self, project_name: str = None) -> dict:
        if project_name:
            project_dir = self.config_manager.projects_dir / project_name
        elif self.current_project:
            project_dir = self.current_project.base_dir
        else:
            raise ValueError("未指定项目")
        
        structure = {
            'root': str(project_dir),
            'images': {
                'original': str(project_dir / "images" / "original"),
                'train': str(project_dir / "images" / "train"),
                'val': str(project_dir / "images" / "val")
            },
            'labels': {
                'original': str(project_dir / "labels" / "original"),
                'train': str(project_dir / "labels" / "train"),
                'val': str(project_dir / "labels" / "val")
            },
            'models': {
                'pt': str(project_dir / "models"),
                'onnx': str(project_dir / "models" / "onnx")
            },
            'logs': str(project_dir / "logs"),
            'reports': str(project_dir / "reports"),
            'exports': str(project_dir / "exports")
        }
        return structure
    
    def add_images(self, image_paths: List[str], project_name: str = None) -> int:
        if project_name:
            project_dir = self.config_manager.projects_dir / project_name
        elif self.current_project:
            project_dir = self.current_project.base_dir
        else:
            raise ValueError("未指定项目")
        
        original_images_dir = project_dir / "images" / "original"
        added_count = 0
        
        for img_path in image_paths:
            img_path = Path(img_path)
            if img_path.exists() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                dest_path = original_images_dir / img_path.name
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)
                    added_count += 1
        
        return added_count
    
    def get_unlabeled_images(self, project_name: str = None) -> List[Path]:
        if project_name:
            project_dir = self.config_manager.projects_dir / project_name
        elif self.current_project:
            project_dir = self.current_project.base_dir
        else:
            raise ValueError("未指定项目")
        
        original_images_dir = project_dir / "images" / "original"
        original_labels_dir = project_dir / "labels" / "original"
        
        unlabeled = []
        for img_file in original_images_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = original_labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    unlabeled.append(img_file)
        
        return sorted(unlabeled)
    
    def get_labeled_images(self, project_name: str = None) -> List[Path]:
        if project_name:
            project_dir = self.config_manager.projects_dir / project_name
        elif self.current_project:
            project_dir = self.current_project.base_dir
        else:
            raise ValueError("未指定项目")
        
        original_images_dir = project_dir / "images" / "original"
        original_labels_dir = project_dir / "labels" / "original"
        
        labeled = []
        for img_file in original_images_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = original_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    labeled.append(img_file)
        
        return sorted(labeled)
    
    def get_annotation_stats(self, project_name: str = None) -> dict:
        if project_name:
            project_dir = self.config_manager.projects_dir / project_name
        elif self.current_project:
            project_dir = self.current_project.base_dir
        else:
            raise ValueError("未指定项目")
        
        original_images_dir = project_dir / "images" / "original"
        original_labels_dir = project_dir / "labels" / "original"
        
        total_images = 0
        labeled_images = 0
        total_annotations = 0
        
        for img_file in original_images_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                total_images += 1
                label_file = original_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    labeled_images += 1
                    with open(label_file, 'r') as f:
                        total_annotations += len(f.readlines())
        
        return {
            'total_images': total_images,
            'labeled_images': labeled_images,
            'unlabeled_images': total_images - labeled_images,
            'total_annotations': total_annotations,
            'label_progress': f"{labeled_images}/{total_images}" if total_images > 0 else "0/0"
        }
    
    def update_model_config(self, model_type: str, model_size: str, model_version: str = None) -> None:
        if not self.current_project:
            raise ValueError("未加载项目")
        
        self.current_project.model_config = self.config_manager.create_model_config(
            model_type, model_size, model_version
        )
        self.config_manager.save_project_config(self.current_project)
    
    def list_all_projects(self) -> List[dict]:
        return self.config_manager.list_projects()
