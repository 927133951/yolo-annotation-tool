import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class ModelConfig:
    name: str
    version: str
    size: str
    full_name: str
    

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    patience: int = 50
    save_period: int = 10
    

@dataclass
class ProjectConfig:
    base_dir: Path
    project_name: str
    classes: List[str]
    model_config: ModelConfig
    training_config: TrainingConfig
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    

class ConfigManager:
    SUPPORTED_MODELS = {
        'yolov5': {
            'sizes': ['n', 's', 'm', 'l', 'x'],
            'versions': ['5.0', '5.1', '7.0'],
            'description': 'YOLOv5 - 经典稳定版本'
        },
        'yolov8': {
            'sizes': ['n', 's', 'm', 'l', 'x'],
            'versions': ['8.0', '8.1', '8.2'],
            'description': 'YOLOv8 - 推荐使用'
        },
        'yolov9': {
            'sizes': ['c', 'e'],
            'versions': ['9.0'],
            'description': 'YOLOv9 - 高精度版本'
        },
        'yolov10': {
            'sizes': ['n', 's', 'm', 'b', 'l', 'x'],
            'versions': ['10.0'],
            'description': 'YOLOv10 - 最新版本'
        },
        'yolo11': {
            'sizes': ['n', 's', 'm', 'l', 'x'],
            'versions': ['11.0'],
            'description': 'YOLO11 - 最新一代'
        }
    }
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = Path.cwd()
        self.base_dir = Path(base_dir)
        self.projects_dir = self.base_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
    def get_available_models(self) -> Dict:
        return self.SUPPORTED_MODELS
    
    def create_model_config(self, model_type: str, model_size: str, model_version: str = None) -> ModelConfig:
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        sizes = self.SUPPORTED_MODELS[model_type]['sizes']
        if model_size not in sizes:
            raise ValueError(f"模型 {model_type} 不支持尺寸 {model_size}，可用尺寸: {sizes}")
        
        if model_version is None:
            model_version = self.SUPPORTED_MODELS[model_type]['versions'][-1]
        
        full_name = f"{model_type}{model_size}"
        
        return ModelConfig(
            name=model_type,
            version=model_version,
            size=model_size,
            full_name=full_name
        )
    
    def generate_model_name(self, project_name: str, model_config: ModelConfig) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{project_name}_{model_config.full_name}_{date_str}"
    
    def create_project_config(
        self,
        project_name: str,
        classes: List[str],
        model_type: str,
        model_size: str,
        model_version: str = None,
        training_config: TrainingConfig = None
    ) -> ProjectConfig:
        model_config = self.create_model_config(model_type, model_size, model_version)
        
        if training_config is None:
            training_config = TrainingConfig()
        
        return ProjectConfig(
            base_dir=self.projects_dir / project_name,
            project_name=project_name,
            classes=classes,
            model_config=model_config,
            training_config=training_config
        )
    
    def save_project_config(self, config: ProjectConfig) -> Path:
        config_path = config.base_dir / "config.json"
        config_dict = {
            'project_name': config.project_name,
            'classes': config.classes,
            'model_config': {
                'name': config.model_config.name,
                'version': config.model_config.version,
                'size': config.model_config.size,
                'full_name': config.model_config.full_name
            },
            'training_config': {
                'epochs': config.training_config.epochs,
                'batch_size': config.training_config.batch_size,
                'image_size': config.training_config.image_size,
                'learning_rate': config.training_config.learning_rate,
                'patience': config.training_config.patience,
                'save_period': config.training_config.save_period
            },
            'created_at': config.created_at
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        return config_path
    
    def load_project_config(self, project_path: Path) -> ProjectConfig:
        config_path = Path(project_path) / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        model_config = ModelConfig(
            name=config_dict['model_config']['name'],
            version=config_dict['model_config']['version'],
            size=config_dict['model_config']['size'],
            full_name=config_dict['model_config']['full_name']
        )
        
        training_config = TrainingConfig(
            epochs=config_dict['training_config']['epochs'],
            batch_size=config_dict['training_config']['batch_size'],
            image_size=config_dict['training_config']['image_size'],
            learning_rate=config_dict['training_config']['learning_rate'],
            patience=config_dict['training_config']['patience'],
            save_period=config_dict['training_config']['save_period']
        )
        
        return ProjectConfig(
            base_dir=Path(project_path),
            project_name=config_dict['project_name'],
            classes=config_dict['classes'],
            model_config=model_config,
            training_config=training_config,
            created_at=config_dict['created_at']
        )
    
    def list_projects(self) -> List[Dict]:
        projects = []
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir():
                config_file = project_dir / "config.json"
                if config_file.exists():
                    try:
                        config = self.load_project_config(project_dir)
                        projects.append({
                            'name': config.project_name,
                            'path': str(project_dir),
                            'created_at': config.created_at,
                            'model': config.model_config.full_name,
                            'classes': config.classes
                        })
                    except Exception as e:
                        print(f"加载项目 {project_dir.name} 配置失败: {e}")
        return projects
