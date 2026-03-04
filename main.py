import sys
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# 导入设备管理器
from device_manager import device_manager

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QStackedWidget,
    QFrame, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QProgressBar, QTextEdit, QSplitter,
    QCheckBox, QLineEdit, QFormLayout, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from config import ConfigManager, TrainingConfig
from project_manager import ProjectManager
from dataset_splitter import DatasetSplitter
from trainer import ModelTrainer
from onnx_converter import ONNXConverter
from report_generator import ReportGenerator
from annotator import YOLOAnnotator


class TrainingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    metrics_update = pyqtSignal(dict)
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
    
    def run(self):
        try:
            self.progress.emit("准备训练环境...")
            success, msg = self.trainer.prepare_training()
            if not success:
                self.finished.emit(False, msg)
                return
            
            self.progress.emit("加载模型...")
            success, msg = self.trainer.load_model()
            if not success:
                self.finished.emit(False, msg)
                return
            
            self.progress.emit("开始训练...")
            success, msg = self.trainer.train()
            
            if success:
                self.progress.emit("训练完成，验证模型...")
                valid, metrics = self.trainer.validate_model()
                if valid:
                    self.metrics_update.emit(metrics)
            
            self.finished.emit(success, msg)
            
        except Exception as e:
            self.finished.emit(False, f"训练过程出错: {str(e)}")


class ONNXConversionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, dict)
    
    def __init__(self, converter, model_path):
        super().__init__()
        self.converter = converter
        self.model_path = model_path
    
    def run(self):
        try:
            self.progress.emit("正在转换为ONNX格式...")
            success, msg = self.converter.convert_to_onnx(self.model_path)
            
            onnx_info = {}
            if success:
                self.progress.emit("验证ONNX模型...")
                valid, onnx_info = self.converter.verify_onnx_model(msg)
                if not valid:
                    onnx_info = {'error': 'ONNX验证失败'}
            
            self.finished.emit(success, msg, onnx_info)
            
        except Exception as e:
            self.finished.emit(False, str(e), {})


class ModelSelectionDialog(QDialog):
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.selected_model = None
        self.selected_size = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("选择模型")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        models = self.config_manager.get_available_models()
        self.model_combo.addItems(models.keys())
        self.model_combo.currentTextChanged.connect(self.update_sizes)
        form_layout.addRow("模型类型:", self.model_combo)
        
        self.size_combo = QComboBox()
        self.update_sizes(self.model_combo.currentText())
        form_layout.addRow("模型尺寸:", self.size_combo)
        
        layout.addLayout(form_layout)
        
        info_label = QLabel(
            "模型尺寸说明:\n"
            "n/nano - 最小最快\n"
            "s/small - 小型\n"
            "m/medium - 中型\n"
            "l/large - 大型\n"
            "x/extra - 最大最准"
        )
        info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(info_label)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def update_sizes(self, model_type):
        self.size_combo.clear()
        models = self.config_manager.get_available_models()
        if model_type in models:
            self.size_combo.addItems(models[model_type]['sizes'])
    
    def get_selection(self):
        return self.model_combo.currentText(), self.size_combo.currentText()


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("创建新项目")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入项目名称")
        form_layout.addRow("项目名称:", self.name_edit)
        
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("用逗号分隔，如: person,car,dog")
        form_layout.addRow("类别列表:", self.classes_edit)
        
        layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_project_info(self):
        name = self.name_edit.text().strip()
        classes_text = self.classes_edit.text().strip()
        classes = [c.strip() for c in classes_text.split(',') if c.strip()]
        return name, classes


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_manager = ProjectManager()
        self.current_project_config = None
        self.trainer = None
        self.training_worker = None
        self.onnx_worker = None
        
        self.init_ui()
        self.refresh_project_list()
    
    def init_ui(self):
        self.setWindowTitle("YOLO标注与训练自动化工具")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        project_group = QGroupBox("项目管理")
        project_layout = QVBoxLayout(project_group)
        
        btn_layout = QHBoxLayout()
        self.btn_new_project = QPushButton("新建项目")
        self.btn_new_project.clicked.connect(self.create_new_project)
        btn_layout.addWidget(self.btn_new_project)
        
        self.btn_open_project = QPushButton("打开项目")
        self.btn_open_project.clicked.connect(self.open_selected_project)
        btn_layout.addWidget(self.btn_open_project)
        project_layout.addLayout(btn_layout)
        
        self.project_list = QListWidget()
        self.project_list.itemDoubleClicked.connect(self.open_selected_project)
        project_layout.addWidget(self.project_list)
        
        layout.addWidget(project_group)
        
        self.project_info_group = QGroupBox("当前项目")
        project_info_layout = QVBoxLayout(self.project_info_group)
        
        self.project_name_label = QLabel("未加载项目")
        self.project_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        project_info_layout.addWidget(self.project_name_label)
        
        self.project_stats_label = QLabel("")
        project_info_layout.addWidget(self.project_stats_label)
        
        self.model_info_label = QLabel("")
        project_info_layout.addWidget(self.model_info_label)
        
        layout.addWidget(self.project_info_group)
        
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        
        self.btn_annotate = QPushButton("开始标注")
        self.btn_annotate.clicked.connect(self.start_annotation)
        self.btn_annotate.setEnabled(False)
        action_layout.addWidget(self.btn_annotate)
        
        self.btn_select_model = QPushButton("选择模型")
        self.btn_select_model.clicked.connect(self.select_model)
        self.btn_select_model.setEnabled(False)
        action_layout.addWidget(self.btn_select_model)
        
        self.btn_train = QPushButton("开始训练")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setEnabled(False)
        self.btn_train.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        action_layout.addWidget(self.btn_train)
        
        layout.addWidget(action_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.stacked_widget = QStackedWidget()
        
        welcome_page = self.create_welcome_page()
        self.stacked_widget.addWidget(welcome_page)
        
        project_page = self.create_project_page()
        self.stacked_widget.addWidget(project_page)
        
        training_page = self.create_training_page()
        self.stacked_widget.addWidget(training_page)
        
        layout.addWidget(self.stacked_widget)
        
        return panel
    
    def create_welcome_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        
        welcome_label = QLabel("欢迎使用 YOLO 标注与训练自动化工具")
        welcome_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        welcome_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_label)
        
        desc_label = QLabel(
            "\n功能特点:\n"
            "• 直观的图像标注界面\n"
            "• 自动化数据集划分\n"
            "• 一键模型训练\n"
            "• 自动ONNX转换\n"
            "• 完整训练报告\n\n"
            "请创建或打开项目开始使用"
        )
        desc_label.setStyleSheet("font-size: 14px; color: #666;")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)
        
        return page
    
    def create_project_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        
        stats_group = QGroupBox("数据集统计")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        config_group = QGroupBox("训练配置")
        config_layout = QFormLayout(config_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        config_layout.addRow("训练轮数:", self.epochs_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        config_layout.addRow("批次大小:", self.batch_spin)
        
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        config_layout.addRow("图像尺寸:", self.imgsz_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        config_layout.addRow("学习率:", self.lr_spin)
        
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 200)
        self.patience_spin.setValue(50)
        config_layout.addRow("早停耐心值:", self.patience_spin)
        
        self.btn_save_config = QPushButton("保存配置")
        self.btn_save_config.clicked.connect(self.save_training_config)
        config_layout.addRow("", self.btn_save_config)
        
        layout.addWidget(config_group)
        
        layout.addStretch()
        
        return page
    
    def create_training_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        progress_layout.addWidget(self.training_log)
        
        layout.addWidget(progress_group)
        
        result_group = QGroupBox("训练结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        result_layout.addWidget(self.result_text)
        
        btn_layout = QHBoxLayout()
        
        self.btn_convert_onnx = QPushButton("转换为ONNX")
        self.btn_convert_onnx.clicked.connect(self.convert_to_onnx)
        self.btn_convert_onnx.setEnabled(False)
        btn_layout.addWidget(self.btn_convert_onnx)
        
        self.btn_open_report = QPushButton("查看报告")
        self.btn_open_report.clicked.connect(self.open_report)
        self.btn_open_report.setEnabled(False)
        btn_layout.addWidget(self.btn_open_report)
        
        result_layout.addLayout(btn_layout)
        
        layout.addWidget(result_group)
        
        return page
    
    def refresh_project_list(self):
        self.project_list.clear()
        projects = self.project_manager.list_all_projects()
        
        for project in projects:
            item = QListWidgetItem(
                f"{project['name']} - {project['model']} ({project['created_at']})"
            )
            item.setData(Qt.UserRole, project['name'])
            self.project_list.addItem(item)
    
    def create_new_project(self):
        dialog = NewProjectDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name, classes = dialog.get_project_info()
            
            if not name:
                QMessageBox.warning(self, "错误", "请输入项目名称")
                return
            
            if not classes:
                QMessageBox.warning(self, "错误", "请输入至少一个类别")
                return
            
            try:
                config = self.project_manager.create_project_structure(name, classes)
                self.current_project_config = config
                self.update_project_info()
                self.refresh_project_list()
                QMessageBox.information(self, "成功", f"项目 '{name}' 创建成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"创建项目失败: {str(e)}")
    
    def open_selected_project(self):
        current_item = self.project_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个项目")
            return
        
        project_name = current_item.data(Qt.UserRole)
        
        try:
            self.current_project_config = self.project_manager.load_project(project_name)
            self.update_project_info()
            self.stacked_widget.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开项目失败: {str(e)}")
    
    def update_project_info(self):
        if not self.current_project_config:
            return
        
        self.project_name_label.setText(self.current_project_config.project_name)
        
        stats = self.project_manager.get_annotation_stats()
        self.project_stats_label.setText(
            f"图像: {stats['labeled_images']}/{stats['total_images']} 已标注\n"
            f"标注框: {stats['total_annotations']}"
        )
        
        self.model_info_label.setText(
            f"模型: {self.current_project_config.model_config.full_name}"
        )
        
        self.update_stats_display()
        
        self.btn_annotate.setEnabled(True)
        self.btn_select_model.setEnabled(True)
        self.btn_train.setEnabled(stats['labeled_images'] > 0)
        
        training_config = self.current_project_config.training_config
        self.epochs_spin.setValue(training_config.epochs)
        self.batch_spin.setValue(training_config.batch_size)
        self.imgsz_spin.setValue(training_config.image_size)
        self.lr_spin.setValue(training_config.learning_rate)
        self.patience_spin.setValue(training_config.patience)
    
    def update_stats_display(self):
        if not self.current_project_config:
            return
        
        stats = self.project_manager.get_annotation_stats()
        
        stats_text = f"""
项目: {self.current_project_config.project_name}
类别: {', '.join(self.current_project_config.classes)}

数据集统计:
  总图像数: {stats['total_images']}
  已标注: {stats['labeled_images']}
  未标注: {stats['unlabeled_images']}
  标注框数: {stats['total_annotations']}
  标注进度: {stats['label_progress']}
"""
        self.stats_text.setText(stats_text)
    
    def save_training_config(self):
        if not self.current_project_config:
            return
        
        self.current_project_config.training_config = TrainingConfig(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            image_size=self.imgsz_spin.value(),
            learning_rate=self.lr_spin.value(),
            patience=self.patience_spin.value()
        )
        
        self.project_manager.config_manager.save_project_config(self.current_project_config)
        QMessageBox.information(self, "成功", "训练配置已保存")
    
    def start_annotation(self):
        if not self.current_project_config:
            return
        
        self.annotator_window = YOLOAnnotator(self.project_manager)
        self.annotator_window.show()
    
    def select_model(self):
        if not self.current_project_config:
            return
        
        dialog = ModelSelectionDialog(
            self.project_manager.config_manager, self
        )
        
        if dialog.exec_() == QDialog.Accepted:
            model_type, model_size = dialog.get_selection()
            self.project_manager.update_model_config(model_type, model_size)
            self.current_project_config = self.project_manager.current_project
            self.model_info_label.setText(
                f"模型: {self.current_project_config.model_config.full_name}"
            )
            QMessageBox.information(
                self, "成功",
                f"已选择模型: {model_type}{model_size}"
            )
    
    def start_training(self):
        if not self.current_project_config:
            return
        
        stats = self.project_manager.get_annotation_stats()
        if stats['labeled_images'] < 10:
            QMessageBox.warning(
                self, "警告",
                f"已标注图像数量不足 ({stats['labeled_images']} < 10)，请先标注更多图像"
            )
            return
        
        reply = QMessageBox.question(
            self, "确认训练",
            f"即将开始训练模型:\n"
            f"项目: {self.current_project_config.project_name}\n"
            f"模型: {self.current_project_config.model_config.full_name}\n"
            f"训练集: {stats['labeled_images']} 张图像\n"
            f"训练轮数: {self.epochs_spin.value()}\n\n"
            f"是否继续?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.save_training_config()
        
        try:
            splitter = DatasetSplitter(self.current_project_config)
            train_count, val_count = splitter.split_dataset()
            
            valid, msg = splitter.check_minimum_requirements()
            if not valid:
                QMessageBox.warning(self, "警告", msg)
                return
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据集划分失败: {str(e)}")
            return
        
        self.stacked_widget.setCurrentIndex(2)
        self.training_log.clear()
        self.result_text.clear()
        
        self.log_training("开始训练流程...")
        self.log_training(f"训练集: {train_count} 张图像")
        self.log_training(f"验证集: {val_count} 张图像")
        
        self.trainer = ModelTrainer(self.current_project_config)
        
        self.training_worker = TrainingWorker(self.trainer)
        self.training_worker.progress.connect(self.log_training)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.metrics_update.connect(self.on_metrics_update)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.btn_train.setEnabled(False)
        
        self.training_worker.start()
    
    def log_training(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] {message}")
    
    def on_training_finished(self, success: bool, message: str):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        
        if success:
            self.log_training("训练完成!")
            self.log_training(message)
            
            self.btn_convert_onnx.setEnabled(True)
            self.btn_open_report.setEnabled(True)
            
            self.generate_report()
        else:
            self.log_training(f"训练失败: {message}")
            QMessageBox.critical(self, "训练失败", message)
    
    def on_metrics_update(self, metrics: dict):
        result_text = "训练结果:\n\n"
        for key, value in metrics.items():
            if value is not None:
                result_text += f"{key}: {value:.4f}\n"
        
        self.result_text.setText(result_text)
    
    def convert_to_onnx(self):
        if not self.trainer or not self.trainer.best_model_path:
            QMessageBox.warning(self, "警告", "没有可转换的模型")
            return
        
        self.log_training("开始ONNX转换...")
        
        converter = ONNXConverter(self.current_project_config)
        
        self.onnx_worker = ONNXConversionWorker(
            converter, str(self.trainer.best_model_path)
        )
        self.onnx_worker.progress.connect(self.log_training)
        self.onnx_worker.finished.connect(self.on_onnx_finished)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.btn_convert_onnx.setEnabled(False)
        
        self.onnx_worker.start()
    
    def on_onnx_finished(self, success: bool, message: str, onnx_info: dict):
        self.progress_bar.setVisible(False)
        self.btn_convert_onnx.setEnabled(True)
        
        if success:
            self.log_training(f"ONNX转换完成: {message}")
            
            if onnx_info:
                self.log_training(f"ONNX模型大小: {onnx_info.get('file_size_mb', 0):.2f} MB")
                self.log_training(f"Opset版本: {onnx_info.get('opset_version', 'N/A')}")
            
            QMessageBox.information(
                self, "转换完成",
                f"ONNX模型已保存:\n{message}"
            )
        else:
            self.log_training(f"ONNX转换失败: {message}")
            QMessageBox.critical(self, "转换失败", message)
    
    def generate_report(self):
        if not self.trainer:
            return
        
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
            
            self.log_training(f"报告已生成: {report_path}")
            
        except Exception as e:
            self.log_training(f"生成报告失败: {str(e)}")
    
    def open_report(self):
        reports_dir = self.current_project_config.base_dir / "reports"
        
        if not reports_dir.exists():
            QMessageBox.warning(self, "警告", "报告目录不存在")
            return
        
        report_files = list(reports_dir.glob("*.md"))
        if not report_files:
            QMessageBox.warning(self, "警告", "没有找到报告文件")
            return
        
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        
        try:
            os.startfile(str(latest_report))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开报告: {str(e)}")


def main():
    # 初始化设备管理器并预热设备
    print("正在初始化设备管理器...")
    device_manager.warmup_device()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
