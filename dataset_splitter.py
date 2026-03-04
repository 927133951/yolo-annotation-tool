import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm
import yaml


class DatasetSplitter:
    def __init__(self, project_config):
        self.project_config = project_config
        self.project_dir = project_config.base_dir
        
    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        seed: int = 42,
        copy_files: bool = True
    ) -> Tuple[int, int]:
        random.seed(seed)
        
        original_images_dir = self.project_dir / "images" / "original"
        original_labels_dir = self.project_dir / "labels" / "original"
        
        train_images_dir = self.project_dir / "images" / "train"
        train_labels_dir = self.project_dir / "labels" / "train"
        val_images_dir = self.project_dir / "images" / "val"
        val_labels_dir = self.project_dir / "labels" / "val"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        labeled_images = []
        for img_file in original_images_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = original_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    labeled_images.append((img_file, label_file))
        
        if not labeled_images:
            raise ValueError("没有找到已标注的图像")
        
        random.shuffle(labeled_images)
        
        train_count = int(len(labeled_images) * train_ratio)
        train_data = labeled_images[:train_count]
        val_data = labeled_images[train_count:]
        
        print(f"数据集划分: 训练集 {len(train_data)} 张, 验证集 {len(val_data)} 张")
        
        print("复制训练集...")
        for img_file, label_file in tqdm(train_data, desc="训练集"):
            if copy_files:
                shutil.copy2(img_file, train_images_dir / img_file.name)
                shutil.copy2(label_file, train_labels_dir / label_file.name)
            else:
                (train_images_dir / img_file.name).symlink_to(img_file)
                (train_labels_dir / label_file.name).symlink_to(label_file)
        
        print("复制验证集...")
        for img_file, label_file in tqdm(val_data, desc="验证集"):
            if copy_files:
                shutil.copy2(img_file, val_images_dir / img_file.name)
                shutil.copy2(label_file, val_labels_dir / label_file.name)
            else:
                (val_images_dir / img_file.name).symlink_to(img_file)
                (val_labels_dir / label_file.name).symlink_to(label_file)
        
        self.create_yaml_config()
        
        return len(train_data), len(val_data)
    
    def create_yaml_config(self) -> Path:
        yaml_content = {
            'path': str(self.project_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.project_config.classes),
            'names': self.project_config.classes
        }
        
        yaml_path = self.project_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        return yaml_path
    
    def get_dataset_stats(self) -> dict:
        train_images_dir = self.project_dir / "images" / "train"
        val_images_dir = self.project_dir / "images" / "val"
        
        train_count = len(list(train_images_dir.glob("*.*")))
        val_count = len(list(val_images_dir.glob("*.*")))
        
        return {
            'train_count': train_count,
            'val_count': val_count,
            'total_count': train_count + val_count,
            'classes': self.project_config.classes,
            'num_classes': len(self.project_config.classes)
        }
    
    def verify_dataset(self) -> Tuple[bool, List[str]]:
        errors = []
        
        train_images_dir = self.project_dir / "images" / "train"
        train_labels_dir = self.project_dir / "labels" / "train"
        val_images_dir = self.project_dir / "images" / "val"
        val_labels_dir = self.project_dir / "labels" / "val"
        
        for images_dir, labels_dir, split_name in [
            (train_images_dir, train_labels_dir, "train"),
            (val_images_dir, val_labels_dir, "val")
        ]:
            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if not label_file.exists():
                        errors.append(f"[{split_name}] 缺少标注文件: {img_file.name}")
                    else:
                        with open(label_file, 'r') as f:
                            for line_num, line in enumerate(f, 1):
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    errors.append(
                                        f"[{split_name}] {label_file.name} 第{line_num}行格式错误"
                                    )
                                    continue
                                
                                try:
                                    class_id = int(parts[0])
                                    if class_id >= len(self.project_config.classes):
                                        errors.append(
                                            f"[{split_name}] {label_file.name} 第{line_num}行类别ID超出范围"
                                        )
                                    
                                    for val in parts[1:]:
                                        v = float(val)
                                        if not 0 <= v <= 1:
                                            errors.append(
                                                f"[{split_name}] {label_file.name} 第{line_num}行坐标值超出范围"
                                            )
                                except ValueError:
                                    errors.append(
                                        f"[{split_name}] {label_file.name} 第{line_num}行数值格式错误"
                                    )
        
        return len(errors) == 0, errors
    
    def check_minimum_requirements(self, min_train: int = 10, min_val: int = 5) -> Tuple[bool, str]:
        stats = self.get_dataset_stats()
        
        if stats['train_count'] < min_train:
            return False, f"训练集图像数量不足: {stats['train_count']} < {min_train}"
        
        if stats['val_count'] < min_val:
            return False, f"验证集图像数量不足: {stats['val_count']} < {min_val}"
        
        if stats['num_classes'] < 1:
            return False, "没有定义类别"
        
        return True, "数据集满足训练要求"
