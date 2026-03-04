import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    def __init__(self, project_config):
        self.project_config = project_config
        self.project_dir = project_config.base_dir
        self.reports_dir = self.project_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_full_report(
        self,
        training_results: Dict,
        validation_metrics: Dict,
        dataset_stats: Dict,
        onnx_info: Dict = None
    ) -> Path:
        report_content = self._create_report_content(
            training_results, validation_metrics, dataset_stats, onnx_info
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"training_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self._generate_charts(training_results, validation_metrics)
        
        return report_path
    
    def _create_report_content(
        self,
        training_results: Dict,
        validation_metrics: Dict,
        dataset_stats: Dict,
        onnx_info: Dict = None
    ) -> str:
        lines = []
        
        lines.append(f"# YOLO模型训练报告")
        lines.append(f"")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"")
        
        lines.append(f"## 1. 项目信息")
        lines.append(f"")
        lines.append(f"| 项目名称 | {self.project_config.project_name} |")
        lines.append(f"| --- | --- |")
        lines.append(f"| 创建时间 | {self.project_config.created_at} |")
        lines.append(f"| 模型类型 | {self.project_config.model_config.full_name} |")
        lines.append(f"| 类别数量 | {len(self.project_config.classes)} |")
        lines.append(f"| 类别列表 | {', '.join(self.project_config.classes)} |")
        lines.append(f"")
        
        lines.append(f"## 2. 数据集统计")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"| --- | --- |")
        lines.append(f"| 训练集图像数 | {dataset_stats.get('train_count', 0)} |")
        lines.append(f"| 验证集图像数 | {dataset_stats.get('val_count', 0)} |")
        lines.append(f"| 总图像数 | {dataset_stats.get('total_count', 0)} |")
        lines.append(f"| 类别数 | {dataset_stats.get('num_classes', 0)} |")
        lines.append(f"")
        
        lines.append(f"## 3. 训练配置")
        lines.append(f"")
        training_config = self.project_config.training_config
        lines.append(f"| 参数 | 值 |")
        lines.append(f"| --- | --- |")
        lines.append(f"| 训练轮数 (Epochs) | {training_config.epochs} |")
        lines.append(f"| 批次大小 (Batch Size) | {training_config.batch_size} |")
        lines.append(f"| 图像尺寸 (Image Size) | {training_config.image_size} |")
        lines.append(f"| 学习率 (Learning Rate) | {training_config.learning_rate} |")
        lines.append(f"| 早停耐心值 (Patience) | {training_config.patience} |")
        lines.append(f"")
        
        lines.append(f"## 4. 训练结果")
        lines.append(f"")
        
        if training_results:
            metrics = training_results.get('metrics', {})
            lines.append(f"| 指标 | 值 |")
            lines.append(f"| --- | --- |")
            lines.append(f"| 精确率 (Precision) | {self._format_metric(metrics.get('precision'))} |")
            lines.append(f"| 召回率 (Recall) | {self._format_metric(metrics.get('recall'))} |")
            lines.append(f"| mAP@0.5 | {self._format_metric(metrics.get('mAP50'))} |")
            lines.append(f"| mAP@0.5:0.95 | {self._format_metric(metrics.get('mAP50_95'))} |")
            lines.append(f"| 适应度 (Fitness) | {self._format_metric(metrics.get('fitness'))} |")
            lines.append(f"")
        
        if validation_metrics:
            lines.append(f"### 验证集评估")
            lines.append(f"")
            lines.append(f"| 指标 | 值 |")
            lines.append(f"| --- | --- |")
            lines.append(f"| 精确率 (Precision) | {self._format_metric(validation_metrics.get('precision'))} |")
            lines.append(f"| 召回率 (Recall) | {self._format_metric(validation_metrics.get('recall'))} |")
            lines.append(f"| mAP@0.5 | {self._format_metric(validation_metrics.get('mAP50'))} |")
            lines.append(f"| mAP@0.5:0.95 | {self._format_metric(validation_metrics.get('mAP50_95'))} |")
            lines.append(f"")
        
        lines.append(f"## 5. 模型文件")
        lines.append(f"")
        models_dir = self.project_dir / "models"
        
        pt_files = list(models_dir.glob("*.pt"))
        if pt_files:
            lines.append(f"### PyTorch模型")
            lines.append(f"")
            for pt_file in pt_files:
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                lines.append(f"- `{pt_file.name}` ({size_mb:.2f} MB)")
            lines.append(f"")
        
        onnx_dir = models_dir / "onnx"
        onnx_files = list(onnx_dir.glob("*.onnx")) if onnx_dir.exists() else []
        if onnx_files:
            lines.append(f"### ONNX模型")
            lines.append(f"")
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                lines.append(f"- `{onnx_file.name}` ({size_mb:.2f} MB)")
            lines.append(f"")
        
        if onnx_info:
            lines.append(f"### ONNX模型信息")
            lines.append(f"")
            lines.append(f"| 属性 | 值 |")
            lines.append(f"| --- | --- |")
            lines.append(f"| 文件大小 | {onnx_info.get('file_size_mb', 0)} MB |")
            lines.append(f"| Opset版本 | {onnx_info.get('opset_version', 'N/A')} |")
            lines.append(f"| 输入形状 | {onnx_info.get('inputs', [{}])[0].get('shape', 'N/A')} |")
            lines.append(f"")
        
        lines.append(f"## 6. 文件结构")
        lines.append(f"")
        lines.append(f"```")
        lines.append(f"{self.project_config.project_name}/")
        lines.append(f"├── images/")
        lines.append(f"│   ├── original/     # 原始图像")
        lines.append(f"│   ├── train/        # 训练集图像")
        lines.append(f"│   └── val/          # 验证集图像")
        lines.append(f"├── labels/")
        lines.append(f"│   ├── original/     # 原始标注")
        lines.append(f"│   ├── train/        # 训练集标注")
        lines.append(f"│   └── val/          # 验证集标注")
        lines.append(f"├── models/")
        lines.append(f"│   ├── *.pt          # PyTorch模型")
        lines.append(f"│   └── onnx/")
        lines.append(f"│       └── *.onnx    # ONNX模型")
        lines.append(f"├── logs/             # 训练日志")
        lines.append(f"├── reports/          # 训练报告")
        lines.append(f"├── config.json       # 项目配置")
        lines.append(f"├── classes.txt       # 类别列表")
        lines.append(f"└── dataset.yaml      # 数据集配置")
        lines.append(f"```")
        lines.append(f"")
        
        lines.append(f"## 7. 使用说明")
        lines.append(f"")
        lines.append(f"### PyTorch模型推理")
        lines.append(f"```python")
        lines.append(f"from ultralytics import YOLO")
        lines.append(f"")
        lines.append(f"model = YOLO('path/to/best.pt')")
        lines.append(f"results = model.predict('image.jpg')")
        lines.append(f"```")
        lines.append(f"")
        lines.append(f"### ONNX模型推理")
        lines.append(f"```python")
        lines.append(f"import onnxruntime as ort")
        lines.append(f"import numpy as np")
        lines.append(f"from PIL import Image")
        lines.append(f"")
        lines.append(f"session = ort.InferenceSession('path/to/model.onnx')")
        lines.append(f"img = Image.open('image.jpg').resize((640, 640))")
        lines.append(f"img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0")
        lines.append(f"img_array = np.expand_dims(img_array, 0)")
        lines.append(f"outputs = session.run(None, {{session.get_inputs()[0].name: img_array}})")
        lines.append(f"```")
        lines.append(f"")
        
        return '\n'.join(lines)
    
    def _format_metric(self, value) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.4f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _generate_charts(self, training_results: Dict, validation_metrics: Dict):
        charts_dir = self.reports_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        if training_results and 'metrics' in training_results:
            metrics = training_results['metrics']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metric_names = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
            metric_values = [
                metrics.get('precision', 0) or 0,
                metrics.get('recall', 0) or 0,
                metrics.get('mAP50', 0) or 0,
                metrics.get('mAP50_95', 0) or 0
            ]
            
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
            bars = ax.bar(metric_names, metric_values, color=colors)
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('值')
            ax.set_title('模型评估指标')
            
            for bar, value in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(charts_dir / 'metrics_chart.png', dpi=150)
            plt.close()
        
        self._generate_class_distribution_chart()
    
    def _generate_class_distribution_chart(self):
        labels_dir = self.project_dir / "labels" / "train"
        
        if not labels_dir.exists():
            return
        
        class_counts = {cls: 0 for cls in self.project_config.classes}
        
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(self.project_config.classes):
                            class_name = self.project_config.classes[class_id]
                            class_counts[class_name] += 1
        
        if any(class_counts.values()):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            bars = ax.bar(classes, counts, color='#2196F3')
            
            ax.set_xlabel('类别')
            ax.set_ylabel('标注框数量')
            ax.set_title('训练集类别分布')
            plt.xticks(rotation=45, ha='right')
            
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            charts_dir = self.reports_dir / "charts"
            plt.savefig(charts_dir / 'class_distribution.png', dpi=150)
            plt.close()
    
    def generate_summary_json(
        self,
        training_results: Dict,
        validation_metrics: Dict,
        dataset_stats: Dict,
        onnx_info: Dict = None
    ) -> Path:
        summary = {
            'project_name': self.project_config.project_name,
            'model_type': self.project_config.model_config.full_name,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_stats': dataset_stats,
            'training_config': {
                'epochs': self.project_config.training_config.epochs,
                'batch_size': self.project_config.training_config.batch_size,
                'image_size': self.project_config.training_config.image_size,
                'learning_rate': self.project_config.training_config.learning_rate,
            },
            'training_results': training_results,
            'validation_metrics': validation_metrics,
            'classes': self.project_config.classes,
            'onnx_info': onnx_info
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.reports_dir / f"training_summary_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        return json_path
