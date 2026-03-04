import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from device_manager import device_manager

# 强制导入 torch - GPU必须可用
import torch


class ModelTrainer:
    def __init__(self, project_config):
        self.project_config = project_config
        self.project_dir = project_config.base_dir
        self.model_config = project_config.model_config
        self.training_config = project_config.training_config
        
        self.model = None
        self.training_results = None
        self.best_model_path = None
        self.final_model_path = None
        
        # 检查GPU是否可用
        if not device_manager.is_gpu_available():
            raise RuntimeError(f"GPU不可用，无法初始化训练器: {device_manager.get_gpu_error_message()}")
    
    def _resolve_device(self) -> str:
        """获取GPU设备 - 强制GPU"""
        return device_manager.get_best_device()
        
    def prepare_training(self) -> Tuple[bool, str]:
        dataset_yaml = self.project_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            return False, "数据集配置文件不存在，请先划分数据集"
        
        models_dir = self.project_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        return True, "训练准备完成"
    
    def load_model(self):
        if not device_manager.is_gpu_available():
            return False, f"GPU不可用: {device_manager.get_gpu_error_message()}"
        
        model_name = self.model_config.full_name
        
        try:
            from ultralytics import YOLO
            
            pretrained_path = Path(f"{model_name}.pt")
            if pretrained_path.exists():
                self.model = YOLO(str(pretrained_path))
                return True, f"成功加载预训练模型: {model_name}.pt"
            
            try:
                self.model = YOLO(f"{model_name}.pt")
                return True, f"成功下载并加载模型: {model_name}.pt"
            except Exception as e1:
                print(f"[训练] 无法加载预训练权重 {model_name}.pt: {e1}")
                print(f"[训练] 尝试从配置文件创建模型...")
                
                config_names = [f"{model_name}.yaml", f"{model_name}.yml"]
                
                for config_name in config_names:
                    try:
                        self.model = YOLO(config_name)
                        return True, f"成功从配置创建模型: {config_name}"
                    except Exception:
                        continue
                
                base_name = self.model_config.name
                size = self.model_config.size
                
                if base_name == 'yolo11':
                    try:
                        self.model = YOLO(f"yolo11{size}.yaml")
                        return True, f"成功创建YOLO11模型: yolo11{size}"
                    except Exception as e2:
                        return False, f"创建模型失败: {str(e2)}"
                
                return False, f"无法加载模型 {model_name}，请检查模型名称是否正确"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"加载模型失败: {str(e)}"
    
    def train(self, progress_callback=None) -> Tuple[bool, str]:
        # 检查GPU是否可用
        if not device_manager.is_gpu_available():
            return False, f"GPU不可用: {device_manager.get_gpu_error_message()}"
            
        if self.model is None:
            success, msg = self.load_model()
            if not success:
                return False, msg
        
        dataset_yaml = self.project_dir / "dataset.yaml"
        model_name = self.generate_model_name()
        output_dir = self.project_dir / "models" / model_name
        
        try:
            # 强制使用GPU
            device = self._resolve_device()
            print(f"[训练] 使用设备: {device}")
            
            results = self.model.train(
                data=str(dataset_yaml),
                epochs=self.training_config.epochs,
                batch=self.training_config.batch_size,
                imgsz=self.training_config.image_size,
                lr0=self.training_config.learning_rate,
                patience=self.training_config.patience,
                save_period=self.training_config.save_period,
                project=str(self.project_dir / "models"),
                name=model_name,
                exist_ok=True,
                verbose=True,
                device=device,
                workers=4,
                optimizer='auto',
                val=True,
                plots=True,
                save=True,
            )
            
            self.training_results = results
            self.save_training_results(results, output_dir)
            
            best_pt = output_dir / "weights" / "best.pt"
            last_pt = output_dir / "weights" / "last.pt"
            
            if best_pt.exists():
                self.best_model_path = best_pt
                final_best = self.project_dir / "models" / f"{model_name}_best.pt"
                shutil.copy2(best_pt, final_best)
            
            if last_pt.exists():
                self.final_model_path = last_pt
                final_last = self.project_dir / "models" / f"{model_name}_last.pt"
                shutil.copy2(last_pt, final_last)
            
            return True, f"训练完成，模型保存至: {output_dir}"
            
        except Exception as e:
            return False, f"训练失败: {str(e)}"
    
    def generate_model_name(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        project_name = self.project_config.project_name
        model_name = self.model_config.full_name
        return f"{project_name}_{model_name}_{date_str}"
    
    def save_training_results(self, results, output_dir: Path):
        results_file = output_dir / "training_results.json"
        
        try:
            results_dict = {
                'model_name': self.generate_model_name(),
                'model_type': self.model_config.name,
                'model_size': self.model_config.size,
                'project_name': self.project_config.project_name,
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'training_config': {
                    'epochs': self.training_config.epochs,
                    'batch_size': self.training_config.batch_size,
                    'image_size': self.training_config.image_size,
                    'learning_rate': self.training_config.learning_rate,
                    'patience': self.training_config.patience,
                },
                'classes': self.project_config.classes,
                'num_classes': len(self.project_config.classes),
            }
            
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                results_dict['metrics'] = {
                    'precision': metrics.get('metrics/precision(B)', None),
                    'recall': metrics.get('metrics/recall(B)', None),
                    'mAP50': metrics.get('metrics/mAP50(B)', None),
                    'mAP50_95': metrics.get('metrics/mAP50-95(B)', None),
                    'fitness': metrics.get('fitness', None),
                }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            print(f"保存训练结果失败: {e}")
    
    def validate_model(self) -> Tuple[bool, Dict]:
        # 检查GPU是否可用
        if not device_manager.is_gpu_available():
            return False, {}
            
        if self.model is None or self.best_model_path is None:
            return False, {}
        
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.best_model_path))
            
            # 强制使用GPU
            device = self._resolve_device()
            print(f"[验证] 使用设备: {device}")
            
            dataset_yaml = self.project_dir / "dataset.yaml"
            results = model.val(data=str(dataset_yaml), device=device)
            
            metrics = {
                'precision': results.results_dict.get('metrics/precision(B)', 0),
                'recall': results.results_dict.get('metrics/recall(B)', 0),
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            }
            
            return True, metrics
            
        except Exception as e:
            print(f"模型验证失败: {e}")
            return False, {}
    
    def get_training_logs(self) -> Path:
        logs_dir = self.project_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    def export_model(self, format: str = 'onnx') -> Tuple[bool, str]:
        # 检查GPU是否可用
        if not device_manager.is_gpu_available():
            return False, f"GPU不可用: {device_manager.get_gpu_error_message()}"
            
        if self.best_model_path is None or not self.best_model_path.exists():
            return False, "最佳模型不存在"
        
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.best_model_path))
            
            # 强制使用GPU
            device = self._resolve_device()
            print(f"[导出] 使用设备: {device}")
            
            onnx_dir = self.project_dir / "models" / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            
            model_name = self.generate_model_name()
            export_path = model.export(format=format, name=model_name, device=device)
            
            return True, str(export_path)
            
        except Exception as e:
            return False, f"模型导出失败: {str(e)}"


class TrainingMonitor:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_file = log_dir / "training_log.txt"
        
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_epoch(self, epoch: int, metrics: Dict):
        self.log(f"Epoch {epoch}: {json.dumps(metrics, default=str)}")
    
    def log_error(self, error: str):
        self.log(f"ERROR: {error}")
    
    def log_info(self, info: str):
        self.log(f"INFO: {info}")
