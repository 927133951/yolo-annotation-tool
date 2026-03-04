import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from config import ConfigManager, TrainingConfig
from project_manager import ProjectManager
from dataset_splitter import DatasetSplitter
from trainer import ModelTrainer
from onnx_converter import ONNXConverter
from report_generator import ReportGenerator


class AutoPipeline:
    def __init__(self, base_dir: str = None):
        self.project_manager = ProjectManager(base_dir)
        self.current_project_config = None
        self.trainer = None
        
    def load_project(self, project_name: str) -> bool:
        try:
            self.current_project_config = self.project_manager.load_project(project_name)
            return True
        except Exception as e:
            print(f"加载项目失败: {e}")
            return False
    
    def check_annotation_status(self) -> Tuple[bool, int, int]:
        if not self.current_project_config:
            return False, 0, 0
        
        stats = self.project_manager.get_annotation_stats()
        return (
            stats['labeled_images'] >= 10,
            stats['labeled_images'],
            stats['total_images']
        )
    
    def auto_split_dataset(self, train_ratio: float = 0.8) -> Tuple[bool, str]:
        if not self.current_project_config:
            return False, "未加载项目"
        
        try:
            splitter = DatasetSplitter(self.current_project_config)
            train_count, val_count = splitter.split_dataset(train_ratio=train_ratio)
            
            valid, msg = splitter.check_minimum_requirements()
            if not valid:
                return False, msg
            
            return True, f"数据集划分完成: 训练集{train_count}张, 验证集{val_count}张"
            
        except Exception as e:
            return False, f"数据集划分失败: {str(e)}"
    
    def auto_train(
        self,
        model_type: str = None,
        model_size: str = None,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        learning_rate: float = 0.01,
        patience: int = 50
    ) -> Tuple[bool, str]:
        if not self.current_project_config:
            return False, "未加载项目"
        
        if model_type and model_size:
            self.project_manager.update_model_config(model_type, model_size)
            self.current_project_config = self.project_manager.current_project
        
        self.current_project_config.training_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            patience=patience
        )
        
        self.trainer = ModelTrainer(self.current_project_config)
        
        success, msg = self.trainer.prepare_training()
        if not success:
            return False, msg
        
        success, msg = self.trainer.load_model()
        if not success:
            return False, msg
        
        print(f"开始训练: {self.current_project_config.project_name}")
        print(f"模型: {self.current_project_config.model_config.full_name}")
        print(f"训练轮数: {epochs}, 批次大小: {batch_size}")
        
        success, msg = self.trainer.train()
        return success, msg
    
    def auto_convert_onnx(self) -> Tuple[bool, str]:
        if not self.trainer or not self.trainer.best_model_path:
            return False, "没有可转换的模型"
        
        converter = ONNXConverter(self.current_project_config)
        return converter.convert_to_onnx(str(self.trainer.best_model_path))
    
    def auto_generate_report(self) -> Tuple[bool, str]:
        if not self.trainer:
            return False, "没有训练结果"
        
        try:
            splitter = DatasetSplitter(self.current_project_config)
            dataset_stats = splitter.get_dataset_stats()
            
            training_results = {}
            if self.trainer.training_results:
                training_results = {
                    'metrics': {
                        'precision': getattr(self.trainer.training_results, 'results_dict', {}).get('metrics/precision(B)'),
                        'recall': getattr(self.trainer.training_results, 'results_dict', {}).get('metrics/recall(B)'),
                        'mAP50': getattr(self.trainer.training_results, 'results_dict', {}).get('metrics/mAP50(B)'),
                        'mAP50_95': getattr(self.trainer.training_results, 'results_dict', {}).get('metrics/mAP50-95(B)'),
                    }
                }
            
            validation_metrics = {}
            valid, metrics = self.trainer.validate_model()
            if valid:
                validation_metrics = metrics
            
            report_gen = ReportGenerator(self.current_project_config)
            report_path = report_gen.generate_full_report(
                training_results, validation_metrics, dataset_stats
            )
            
            return True, str(report_path)
            
        except Exception as e:
            return False, f"生成报告失败: {str(e)}"
    
    def run_full_pipeline(
        self,
        project_name: str,
        model_type: str = 'yolov8',
        model_size: str = 'n',
        epochs: int = 100,
        batch_size: int = 16
    ) -> Tuple[bool, str]:
        print(f"\n{'='*50}")
        print(f"YOLO自动化训练流程")
        print(f"{'='*50}\n")
        
        print(f"[1/5] 加载项目: {project_name}")
        if not self.load_project(project_name):
            return False, "项目加载失败"
        print("      ✓ 项目加载成功")
        
        print(f"\n[2/5] 检查标注状态...")
        ready, labeled, total = self.check_annotation_status()
        print(f"      已标注: {labeled}/{total} 张图像")
        if not ready:
            return False, f"标注图像不足 (需要至少10张, 当前{labeled}张)"
        print("      ✓ 标注检查通过")
        
        print(f"\n[3/5] 划分数据集...")
        success, msg = self.auto_split_dataset()
        if not success:
            return False, msg
        print(f"      ✓ {msg}")
        
        print(f"\n[4/5] 开始训练...")
        print(f"      模型: {model_type}{model_size}")
        print(f"      轮数: {epochs}, 批次: {batch_size}")
        success, msg = self.auto_train(
            model_type=model_type,
            model_size=model_size,
            epochs=epochs,
            batch_size=batch_size
        )
        if not success:
            return False, msg
        print(f"      ✓ 训练完成")
        
        print(f"\n[5/5] 转换ONNX并生成报告...")
        success, onnx_path = self.auto_convert_onnx()
        if success:
            print(f"      ✓ ONNX模型: {onnx_path}")
        else:
            print(f"      ✗ ONNX转换失败: {onnx_path}")
        
        success, report_path = self.auto_generate_report()
        if success:
            print(f"      ✓ 训练报告: {report_path}")
        else:
            print(f"      ✗ 报告生成失败: {report_path}")
        
        print(f"\n{'='*50}")
        print(f"自动化流程完成!")
        print(f"{'='*50}\n")
        
        return True, "自动化流程执行完成"
    
    def list_available_projects(self):
        return self.project_manager.list_all_projects()


def run_cli():
    print("\nYOLO标注与训练自动化工具 - 命令行模式\n")
    
    pipeline = AutoPipeline()
    
    projects = pipeline.list_available_projects()
    if not projects:
        print("没有找到项目，请先使用GUI创建项目并标注图像")
        return
    
    print("可用项目:")
    for i, project in enumerate(projects, 1):
        print(f"  {i}. {project['name']} ({project['created_at']})")
    
    choice = input("\n选择项目编号: ").strip()
    try:
        index = int(choice) - 1
        project_name = projects[index]['name']
    except (ValueError, IndexError):
        print("无效选择")
        return
    
    print("\n选择模型类型:")
    print("  1. YOLOv8n (推荐 - 最快)")
    print("  2. YOLOv8s")
    print("  3. YOLOv8m")
    print("  4. YOLOv8l")
    print("  5. YOLOv8x")
    
    model_choice = input("选择模型 (1-5, 默认1): ").strip() or "1"
    model_sizes = ['n', 's', 'm', 'l', 'x']
    model_size = model_sizes[int(model_choice) - 1] if model_choice.isdigit() else 'n'
    
    epochs = input("训练轮数 (默认100): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 100
    
    batch_size = input("批次大小 (默认16): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 16
    
    success, msg = pipeline.run_full_pipeline(
        project_name=project_name,
        model_type='yolov8',
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size
    )
    
    if not success:
        print(f"\n错误: {msg}")


if __name__ == "__main__":
    run_cli()
