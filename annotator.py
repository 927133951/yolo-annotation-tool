import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QFileDialog,
    QScrollArea, QFrame, QSplitter, QMessageBox, QInputDialog,
    QComboBox, QSpinBox, QGroupBox, QStatusBar, QProgressBar
)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PIL import Image
import numpy as np


class AnnotationCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap: Optional[QPixmap] = None
        self.display_pixmap: Optional[QPixmap] = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_boxes: List[Tuple[QRect, int]] = []
        self.selected_box_index = -1
        
        self.classes: List[str] = []
        self.current_class_index = 0
        
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        
    def set_image(self, image_path: str):
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            return False
        self.current_boxes = []
        self.selected_box_index = -1
        self.update_display()
        return True
    
    def update_display(self):
        if not self.original_pixmap:
            return
        
        canvas_size = self.size()
        img_size = self.original_pixmap.size()
        
        scale_x = canvas_size.width() / img_size.width()
        scale_y = canvas_size.height() / img_size.height()
        self.scale_factor = min(scale_x, scale_y, 1.0)
        
        new_width = int(img_size.width() * self.scale_factor)
        new_height = int(img_size.height() * self.scale_factor)
        
        self.display_pixmap = self.original_pixmap.scaled(
            new_width, new_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.offset_x = (canvas_size.width() - new_width) // 2
        self.offset_y = (canvas_size.height() - new_height) // 2
        
        self.draw_boxes()
    
    def draw_boxes(self):
        if not self.display_pixmap:
            return
        
        temp_pixmap = QPixmap(self.display_pixmap)
        painter = QPainter(temp_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        colors = [
            QColor(255, 0, 0, 200),
            QColor(0, 255, 0, 200),
            QColor(0, 0, 255, 200),
            QColor(255, 255, 0, 200),
            QColor(255, 0, 255, 200),
            QColor(0, 255, 255, 200),
            QColor(128, 0, 255, 200),
            QColor(255, 128, 0, 200),
        ]
        
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        
        for i, (rect, class_idx) in enumerate(self.current_boxes):
            color = colors[class_idx % len(colors)]
            pen = QPen(color, 2)
            
            if i == self.selected_box_index:
                pen.setWidth(3)
                pen.setStyle(Qt.DashLine)
            
            painter.setPen(pen)
            painter.drawRect(rect)
            
            label = self.classes[class_idx] if class_idx < len(self.classes) else f"Class {class_idx}"
            text_rect = QRect(rect.x(), rect.y() - 20, rect.width(), 20)
            
            painter.fillRect(text_rect, color)
            painter.setPen(QPen(Qt.white))
            painter.drawText(text_rect, Qt.AlignCenter, label)
        
        if self.drawing:
            current_rect = QRect(self.start_point, self.end_point).normalized()
            pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(current_rect)
        
        painter.end()
        
        final_pixmap = QPixmap(self.size())
        final_pixmap.fill(Qt.black)
        
        painter = QPainter(final_pixmap)
        painter.drawPixmap(self.offset_x, self.offset_y, temp_pixmap)
        painter.end()
        
        self.setPixmap(final_pixmap)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.display_pixmap:
            pos = event.pos()
            canvas_rect = QRect(
                self.offset_x, self.offset_y,
                self.display_pixmap.width(), self.display_pixmap.height()
            )
            
            if canvas_rect.contains(pos):
                for i, (rect, _) in enumerate(self.current_boxes):
                    if rect.contains(pos.x() - self.offset_x, pos.y() - self.offset_y):
                        self.selected_box_index = i
                        self.draw_boxes()
                        return
                
                self.drawing = True
                self.start_point = QPoint(
                    pos.x() - self.offset_x,
                    pos.y() - self.offset_y
                )
                self.end_point = self.start_point
                self.selected_box_index = -1
        
        elif event.button() == Qt.RightButton:
            if self.selected_box_index >= 0:
                del self.current_boxes[self.selected_box_index]
                self.selected_box_index = -1
                self.draw_boxes()
    
    def mouseMoveEvent(self, event):
        if self.drawing and self.display_pixmap:
            pos = event.pos()
            self.end_point = QPoint(
                pos.x() - self.offset_x,
                pos.y() - self.offset_y
            )
            self.draw_boxes()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, self.end_point).normalized()
            
            if rect.width() > 10 and rect.height() > 10:
                self.current_boxes.append((rect, self.current_class_index))
            
            self.draw_boxes()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()
    
    def set_classes(self, classes: List[str]):
        self.classes = classes
    
    def set_current_class(self, class_index: int):
        self.current_class_index = class_index
    
    def get_yolo_annotations(self) -> List[str]:
        if not self.original_pixmap or not self.current_boxes:
            return []
        
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        
        yolo_lines = []
        for rect, class_idx in self.current_boxes:
            x_center = (rect.x() + rect.width() / 2) / img_width
            y_center = (rect.y() + rect.height() / 2) / img_height
            width = rect.width() / img_width
            height = rect.height() / img_height
            
            yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_lines
    
    def load_annotations(self, annotations: List[str]):
        if not self.original_pixmap:
            return
        
        self.current_boxes = []
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        
        for line in annotations:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_idx = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                x = int((x_center - width / 2) * img_width)
                y = int((y_center - height / 2) * img_height)
                w = int(width * img_width)
                h = int(height * img_height)
                
                rect = QRect(x, y, w, h)
                self.current_boxes.append((rect, class_idx))
        
        self.draw_boxes()
    
    def clear_annotations(self):
        self.current_boxes = []
        self.selected_box_index = -1
        self.draw_boxes()


class YOLOAnnotator(QMainWindow):
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.current_image_path: Optional[Path] = None
        self.unlabeled_images: List[Path] = []
        self.current_image_index = -1
        
        self.init_ui()
        self.load_project_info()
    
    def init_ui(self):
        self.setWindowTitle("YOLO标注工具")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, 3)
        
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        self.progress_bar = QProgressBar()
        self.statusBar.addPermanentWidget(self.progress_bar)
    
    def create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        project_group = QGroupBox("项目信息")
        project_layout = QVBoxLayout(project_group)
        self.project_name_label = QLabel("项目: 未加载")
        self.project_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        project_layout.addWidget(self.project_name_label)
        self.stats_label = QLabel("统计: 0/0 已标注")
        project_layout.addWidget(self.stats_label)
        layout.addWidget(project_group)
        
        image_list_group = QGroupBox("图像列表")
        image_list_layout = QVBoxLayout(image_list_group)
        
        btn_layout = QHBoxLayout()
        self.btn_import = QPushButton("导入图像")
        self.btn_import.clicked.connect(self.import_images)
        btn_layout.addWidget(self.btn_import)
        
        self.btn_refresh = QPushButton("刷新列表")
        self.btn_refresh.clicked.connect(self.refresh_image_list)
        btn_layout.addWidget(self.btn_refresh)
        image_list_layout.addLayout(btn_layout)
        
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        self.image_list.itemDoubleClicked.connect(self.on_image_double_clicked)
        image_list_layout.addWidget(self.image_list)
        
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("上一张")
        self.btn_prev.clicked.connect(self.prev_image)
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("下一张")
        self.btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(self.btn_next)
        image_list_layout.addLayout(nav_layout)
        
        layout.addWidget(image_list_group)
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        toolbar_layout = QHBoxLayout()
        
        self.btn_clear = QPushButton("清除标注")
        self.btn_clear.clicked.connect(self.clear_current_annotations)
        toolbar_layout.addWidget(self.btn_clear)
        
        self.btn_save = QPushButton("保存标注")
        self.btn_save.clicked.connect(self.save_current_annotations)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        toolbar_layout.addWidget(self.btn_save)
        
        self.btn_delete = QPushButton("删除图像")
        self.btn_delete.clicked.connect(self.delete_current_image)
        toolbar_layout.addWidget(self.btn_delete)
        
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        
        self.canvas = AnnotationCanvas()
        scroll_area.setWidget(self.canvas)
        layout.addWidget(scroll_area, 1)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        class_group = QGroupBox("类别选择")
        class_layout = QVBoxLayout(class_group)
        
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        class_layout.addWidget(self.class_combo)
        
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(150)
        class_layout.addWidget(self.class_list)
        
        btn_class_layout = QHBoxLayout()
        self.btn_add_class = QPushButton("添加类别")
        self.btn_add_class.clicked.connect(self.add_class)
        btn_class_layout.addWidget(self.btn_add_class)
        
        self.btn_remove_class = QPushButton("删除类别")
        self.btn_remove_class.clicked.connect(self.remove_class)
        btn_class_layout.addWidget(self.btn_remove_class)
        class_layout.addLayout(btn_class_layout)
        
        layout.addWidget(class_group)
        
        annotation_group = QGroupBox("当前标注")
        annotation_layout = QVBoxLayout(annotation_group)
        
        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self.on_annotation_clicked)
        annotation_layout.addWidget(self.annotation_list)
        
        self.btn_delete_annotation = QPushButton("删除选中标注")
        self.btn_delete_annotation.clicked.connect(self.delete_selected_annotation)
        annotation_layout.addWidget(self.btn_delete_annotation)
        
        layout.addWidget(annotation_group)
        
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        
        self.btn_auto_save = QPushButton("自动保存并继续")
        self.btn_auto_save.clicked.connect(self.auto_save_and_next)
        self.btn_auto_save.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        action_layout.addWidget(self.btn_auto_save)
        
        layout.addWidget(action_group)
        
        layout.addStretch()
        
        return panel
    
    def load_project_info(self):
        if self.project_manager.current_project:
            project = self.project_manager.current_project
            self.project_name_label.setText(f"项目: {project.project_name}")
            
            self.canvas.set_classes(project.classes)
            self.class_combo.clear()
            self.class_combo.addItems(project.classes)
            self.class_list.clear()
            self.class_list.addItems(project.classes)
            
            self.refresh_image_list()
    
    def refresh_image_list(self):
        self.image_list.clear()
        
        if not self.project_manager.current_project:
            return
        
        stats = self.project_manager.get_annotation_stats()
        self.stats_label.setText(
            f"统计: {stats['labeled_images']}/{stats['total_images']} 已标注 "
            f"(共{stats['total_annotations']}个标注框)"
        )
        
        self.unlabeled_images = self.project_manager.get_unlabeled_images()
        labeled_images = self.project_manager.get_labeled_images()
        
        self.image_list.addItem(f"--- 未标注 ({len(self.unlabeled_images)}) ---")
        for img_path in self.unlabeled_images:
            item = QListWidgetItem(img_path.name)
            item.setData(Qt.UserRole, str(img_path))
            item.setForeground(QColor(200, 0, 0))
            self.image_list.addItem(item)
        
        self.image_list.addItem(f"--- 已标注 ({len(labeled_images)}) ---")
        for img_path in labeled_images:
            item = QListWidgetItem(img_path.name)
            item.setData(Qt.UserRole, str(img_path))
            item.setForeground(QColor(0, 150, 0))
            self.image_list.addItem(item)
        
        total = stats['total_images']
        labeled = stats['labeled_images']
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(labeled)
    
    def import_images(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self, "选择图像文件", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        
        if file_paths:
            added = self.project_manager.add_images(file_paths)
            self.refresh_image_list()
            QMessageBox.information(self, "导入完成", f"成功导入 {added} 张图像")
    
    def on_image_selected(self, row):
        pass
    
    def on_image_double_clicked(self, item):
        image_path = item.data(Qt.UserRole)
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path: str):
        self.current_image_path = Path(image_path)
        
        if self.canvas.set_image(image_path):
            self.statusBar.showMessage(f"已加载: {self.current_image_path.name}")
            
            label_path = self.get_label_path()
            if label_path.exists():
                with open(label_path, 'r') as f:
                    annotations = f.readlines()
                self.canvas.load_annotations(annotations)
            
            self.update_annotation_list()
        else:
            QMessageBox.warning(self, "错误", f"无法加载图像: {image_path}")
    
    def get_label_path(self) -> Path:
        if not self.current_image_path:
            return None
        
        project = self.project_manager.current_project
        label_dir = project.base_dir / "labels" / "original"
        return label_dir / f"{self.current_image_path.stem}.txt"
    
    def save_current_annotations(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "没有当前图像")
            return
        
        annotations = self.canvas.get_yolo_annotations()
        label_path = self.get_label_path()
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        self.statusBar.showMessage(f"已保存: {label_path.name}")
        self.refresh_image_list()
    
    def clear_current_annotations(self):
        self.canvas.clear_annotations()
        self.update_annotation_list()
    
    def delete_current_image(self):
        if not self.current_image_path:
            return
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除图像 {self.current_image_path.name} 吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            label_path = self.get_label_path()
            
            if self.current_image_path.exists():
                os.remove(self.current_image_path)
            if label_path.exists():
                os.remove(label_path)
            
            self.canvas.clear_annotations()
            self.current_image_path = None
            self.refresh_image_list()
            self.statusBar.showMessage("图像已删除")
    
    def prev_image(self):
        current_row = self.image_list.currentRow()
        if current_row > 0:
            for i in range(current_row - 1, -1, -1):
                item = self.image_list.item(i)
                if item.data(Qt.UserRole):
                    self.image_list.setCurrentRow(i)
                    self.load_image(item.data(Qt.UserRole))
                    break
    
    def next_image(self):
        current_row = self.image_list.currentRow()
        for i in range(current_row + 1, self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole):
                self.image_list.setCurrentRow(i)
                self.load_image(item.data(Qt.UserRole))
                break
    
    def auto_save_and_next(self):
        self.save_current_annotations()
        self.next_image()
    
    def on_class_changed(self, index):
        self.canvas.set_current_class(index)
    
    def add_class(self):
        text, ok = QInputDialog.getText(self, "添加类别", "输入类别名称:")
        if ok and text:
            if self.project_manager.current_project:
                self.project_manager.current_project.classes.append(text)
                self.class_combo.addItem(text)
                self.class_list.addItem(text)
                self.canvas.set_classes(self.project_manager.current_project.classes)
    
    def remove_class(self):
        current_row = self.class_list.currentRow()
        if current_row >= 0:
            if self.project_manager.current_project:
                del self.project_manager.current_project.classes[current_row]
                self.class_combo.removeItem(current_row)
                self.class_list.takeItem(current_row)
                self.canvas.set_classes(self.project_manager.current_project.classes)
    
    def update_annotation_list(self):
        self.annotation_list.clear()
        
        for i, (rect, class_idx) in enumerate(self.canvas.current_boxes):
            class_name = self.canvas.classes[class_idx] if class_idx < len(self.canvas.classes) else f"Class {class_idx}"
            item_text = f"{i+1}. {class_name} ({rect.width()}x{rect.height()})"
            self.annotation_list.addItem(item_text)
    
    def on_annotation_clicked(self, item):
        index = self.annotation_list.row(item)
        self.canvas.selected_box_index = index
        self.canvas.draw_boxes()
    
    def delete_selected_annotation(self):
        if self.canvas.selected_box_index >= 0:
            del self.canvas.current_boxes[self.canvas.selected_box_index]
            self.canvas.selected_box_index = -1
            self.canvas.draw_boxes()
            self.update_annotation_list()


def run_annotator(project_manager):
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = YOLOAnnotator(project_manager)
    window.show()
    
    return app.exec_()
