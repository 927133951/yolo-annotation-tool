import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import json
import threading
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QStackedWidget,
    QFrame, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QProgressBar, QTextEdit, QSplitter,
    QCheckBox, QLineEdit, QFormLayout, QDialog, QDialogButtonBox,
    QTabWidget, QScrollArea, QGridLayout, QSizePolicy, QToolBar,
    QAction, QStatusBar, QMenu, QToolTip, QHeaderView, QTableWidget,
    QTableWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QLinearGradient, QImage

from config import ConfigManager, TrainingConfig, ProjectConfig
from project_manager import ProjectManager
from dataset_splitter import DatasetSplitter
from trainer import ModelTrainer
from onnx_converter import ONNXConverter
from report_generator import ReportGenerator
from inference import YOLOInference, InferenceConfig, InferenceResult, DetectionResult
from device_manager import device_manager


class Styles:
    PRIMARY = "#2196F3"
    PRIMARY_DARK = "#1976D2"
    PRIMARY_LIGHT = "#BBDEFB"
    ACCENT = "#FF5722"
    SUCCESS = "#4CAF50"
    WARNING = "#FF9800"
    ERROR = "#F44336"
    TEXT_PRIMARY = "#212121"
    TEXT_SECONDARY = "#757575"
    BACKGROUND = "#FAFAFA"
    CARD = "#FFFFFF"
    BORDER = "#E0E0E0"
    
    BUTTON_PRIMARY = """
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 4px;
            font-weight: bold;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #1565C0;
        }
        QPushButton:disabled {
            background-color: #BDBDBD;
        }
    """
    
    BUTTON_SUCCESS = """
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 4px;
            font-weight: bold;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #43A047;
        }
        QPushButton:pressed {
            background-color: #388E3C;
        }
    """
    
    BUTTON_WARNING = """
        QPushButton {
            background-color: #FF9800;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 4px;
            font-weight: bold;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #F57C00;
        }
    """
    
    BUTTON_DANGER = """
        QPushButton {
            background-color: #F44336;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 4px;
            font-weight: bold;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #E53935;
        }
    """
    
    BUTTON_OUTLINE = """
        QPushButton {
            background-color: transparent;
            color: #2196F3;
            border: 2px solid #2196F3;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #E3F2FD;
        }
    """
    
    GROUP_BOX = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            margin-top: 12px;
            padding-top: 10px;
            background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #212121;
        }
    """
    
    LIST_ITEM = """
        QListWidget {
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            background-color: white;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #EEEEEE;
        }
        QListWidget::item:selected {
            background-color: #E3F2FD;
            color: #1976D2;
        }
        QListWidget::item:hover {
            background-color: #F5F5F5;
        }
    """
    
    TAB_WIDGET = """
        QTabWidget::pane {
            border: 1px solid #E0E0E0;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #E0E0E0;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #2196F3;
        }
        QTabBar::tab:hover:!selected {
            background-color: #F5F5F5;
        }
    """
    
    INPUT = """
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 8px;
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            background-color: white;
            color: #212121;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 2px solid #2196F3;
        }
        QComboBox:hover {
            background-color: #F5F5F5;
            border: 1px solid #BDBDBD;
        }
        QComboBox::drop-down:hover {
            background-color: #F5F5F5;
        }
        QComboBox QAbstractItemView {
            background-color: white;
            border: 1px solid #E0E0E0;
            border-radius: 4px;
        }
        QComboBox QAbstractItemView::item {
            padding: 8px;
            color: #212121;
        }
        QComboBox QAbstractItemView::item:hover {
            background-color: #E3F2FD;
            color: #1976D2;
        }
        QComboBox QAbstractItemView::item:selected {
            background-color: #2196F3;
            color: white;
        }
    """
    
    PROGRESS = """
        QProgressBar {
            border: none;
            border-radius: 4px;
            background-color: #E0E0E0;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 4px;
        }
    """


class ClassItemWidget(QFrame):
    class_selected = pyqtSignal(int)
    class_deleted = pyqtSignal(int)
    class_renamed = pyqtSignal(int, str)
    editing_started = pyqtSignal()
    editing_finished = pyqtSignal()
    
    COLORS = [
        '#FF5252', '#4CAF50', '#2196F3', '#FFC107', '#9C27B0',
        '#00BCD4', '#FF5722', '#8BC34A', '#3F51B5', '#795548',
        '#E91E63', '#009688', '#FF9800', '#673AB7', '#607D8B'
    ]
    
    def __init__(self, class_name: str, class_index: int, parent=None):
        super().__init__(parent)
        self.class_name = class_name
        self._original_name = class_name
        self.class_index = class_index
        self.is_selected = False
        self.is_editing = False
        
        self.setFixedHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameShape(QFrame.StyledPanel)
        self.init_ui()
        self.update_style()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(10)
        
        self.selection_indicator = QLabel()
        self.selection_indicator.setFixedSize(6, 24)
        self.selection_indicator.setStyleSheet("background-color: transparent; border-radius: 3px;")
        layout.addWidget(self.selection_indicator)
        
        self.color_indicator = QLabel()
        self.color_indicator.setFixedSize(18, 18)
        color = self.COLORS[self.class_index % len(self.COLORS)]
        self.color_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 9px;"
        )
        layout.addWidget(self.color_indicator)
        
        self.name_label = QLabel(self.class_name)
        self.name_label.setStyleSheet("font-size: 14px; color: #212121; background: transparent;")
        layout.addWidget(self.name_label, 1)
        
        self.name_edit = QLineEdit()
        self.name_edit.setText(self.class_name)
        self.name_edit.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #4CAF50;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
                color: #212121;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
                background-color: #F1F8E9;
            }
        """)
        self.name_edit.setVisible(False)
        self.name_edit.returnPressed.connect(self.finish_editing)
        self.name_edit.installEventFilter(self)
        layout.addWidget(self.name_edit, 1)
        
        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(26, 26)
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #BDBDBD;
                border: none;
                font-size: 20px;
                font-weight: bold;
                border-radius: 13px;
            }
            QPushButton:hover {
                background-color: #FFEBEE;
                color: #F44336;
            }
        """)
        self.delete_btn.clicked.connect(self.on_delete_clicked)
        self.delete_btn.setVisible(False)
        layout.addWidget(self.delete_btn)
    
    def on_delete_clicked(self):
        if self.is_editing:
            self.cancel_editing()
        else:
            self.class_deleted.emit(self.class_index)
    
    def eventFilter(self, obj, event):
        if obj == self.name_edit:
            if event.type() == event.KeyPress:
                if event.key() == Qt.Key_Escape:
                    self.cancel_editing()
                    return True
                elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                    self.finish_editing()
                    return True
            elif event.type() == event.FocusOut:
                self.finish_editing()
                return False
        return super().eventFilter(obj, event)
    
    def set_selected(self, selected: bool):
        if self.is_editing and not selected:
            self.finish_editing()
        self.is_selected = selected
        self.update_style()
    
    def update_style(self):
        if self.is_editing:
            self.setStyleSheet("""
                QFrame {
                    background-color: #F1F8E9;
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                }
            """)
            self.selection_indicator.setStyleSheet("background-color: #4CAF50; border-radius: 3px;")
        elif self.is_selected:
            self.setStyleSheet("""
                QFrame {
                    background-color: #E3F2FD;
                    border: none;
                    border-radius: 8px;
                }
            """)
            self.selection_indicator.setStyleSheet("background-color: #2196F3; border-radius: 3px;")
            self.name_label.setStyleSheet("font-size: 14px; color: #1565C0; font-weight: bold; background: transparent;")
            self.delete_btn.setVisible(True)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: transparent;
                    border: none;
                    border-radius: 8px;
                }
                QFrame:hover {
                    background-color: #F5F5F5;
                }
            """)
            self.selection_indicator.setStyleSheet("background-color: transparent; border-radius: 3px;")
            self.name_label.setStyleSheet("font-size: 14px; color: #424242; background: transparent;")
    
    def enterEvent(self, event):
        if not self.is_editing:
            self.delete_btn.setVisible(True)
            if not self.is_selected:
                self.setStyleSheet("""
                    QFrame {
                        background-color: #F5F5F5;
                        border: none;
                        border-radius: 8px;
                    }
                """)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        if not self.is_selected and not self.is_editing:
            self.delete_btn.setVisible(False)
            self.setStyleSheet("""
                QFrame {
                    background-color: transparent;
                    border: none;
                    border-radius: 8px;
                }
            """)
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_editing:
                if not self.name_edit.geometry().contains(event.pos()):
                    self.finish_editing()
                return
            
            if not self.delete_btn.geometry().contains(event.pos()):
                if not self.is_selected:
                    self.class_selected.emit(self.class_index)
    
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.delete_btn.geometry().contains(event.pos()):
                self.start_editing()
    
    def start_editing(self):
        if self.is_editing:
            return
        
        self.is_editing = True
        self._original_name = self.class_name
        self.name_edit.setText(self.class_name)
        self.name_edit.setVisible(True)
        self.name_label.setVisible(False)
        self.update_style()
        self.name_edit.setFocus()
        self.name_edit.selectAll()
        self.editing_started.emit()
    
    def finish_editing(self):
        if not self.is_editing:
            return
        
        new_name = self.name_edit.text().strip()
        
        if new_name and new_name != self._original_name:
            self.class_name = new_name
            self.name_label.setText(new_name)
            self.class_renamed.emit(self.class_index, new_name)
        elif not new_name:
            self.name_label.setText(self._original_name)
            self.class_name = self._original_name
        
        self.is_editing = False
        self.name_edit.setVisible(False)
        self.name_label.setVisible(True)
        self.update_style()
        self.editing_finished.emit()
    
    def cancel_editing(self):
        if not self.is_editing:
            return
        
        self.class_name = self._original_name
        self.name_edit.setText(self._original_name)
        self.is_editing = False
        self.name_edit.setVisible(False)
        self.name_label.setVisible(True)
        self.update_style()
        self.editing_finished.emit()
    
    def update_index(self, new_index: int):
        self.class_index = new_index
        color = self.COLORS[new_index % len(self.COLORS)]
        self.color_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 9px;"
        )


class ClassSelectorWidget(QWidget):
    class_changed = pyqtSignal(int)
    classes_updated = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.classes: List[str] = []
        self.class_items: List[ClassItemWidget] = []
        self.current_index = -1
        self.project_manager = None
        
        self.setAcceptDrops(True)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        header_layout = QHBoxLayout()
        
        header = QLabel("标注分类")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #212121; padding: 8px;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        self.add_btn = QPushButton("➕")
        self.add_btn.setFixedSize(28, 28)
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #E3F2FD;
                color: #2196F3;
                border: none;
                border-radius: 14px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #BBDEFB;
            }
        """)
        self.add_btn.setToolTip("添加新分类 (快捷键: Insert)")
        self.add_btn.clicked.connect(self.add_new_class)
        header_layout.addWidget(self.add_btn)
        
        layout.addLayout(header_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                width: 6px;
                background-color: #F5F5F5;
            }
            QScrollBar::handle:vertical {
                background-color: #BDBDBD;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #9E9E9E;
            }
        """)
        
        self.items_widget = QWidget()
        self.items_layout = QVBoxLayout(self.items_widget)
        self.items_layout.setContentsMargins(4, 4, 4, 4)
        self.items_layout.setSpacing(4)
        self.items_layout.addStretch()
        
        self.scroll_area.setWidget(self.items_widget)
        layout.addWidget(self.scroll_area)
        
        self.hint_label = QLabel("点击选择 • 双击编辑 • 拖拽排序")
        self.hint_label.setStyleSheet("color: #9E9E9E; font-size: 11px; padding: 8px;")
        layout.addWidget(self.hint_label)
    
    def set_project_manager(self, project_manager):
        self.project_manager = project_manager
    
    def set_classes(self, classes: List[str]):
        self.classes = list(classes)
        self.rebuild_items()
        
        if self.classes and self.current_index < 0:
            self.set_current_index(0)
    
    def rebuild_items(self):
        for item in self.class_items:
            item.deleteLater()
        self.class_items.clear()
        
        for i, class_name in enumerate(self.classes):
            self.add_class_item(class_name, i)
        
        self.items_layout.addStretch()
    
    def add_class_item(self, class_name: str, index: int):
        item = ClassItemWidget(class_name, index)
        item.class_selected.connect(self.on_item_selected)
        item.class_deleted.connect(self.on_item_deleted)
        item.class_renamed.connect(self.on_item_renamed)
        item.editing_started.connect(self.on_editing_started)
        
        if index == self.current_index:
            item.set_selected(True)
        
        self.class_items.append(item)
        self.items_layout.insertWidget(self.items_layout.count() - 1, item)
    
    def on_editing_started(self):
        pass
    
    def on_item_selected(self, index: int):
        self.set_current_index(index)
    
    def on_item_deleted(self, index: int):
        if len(self.classes) <= 1:
            QMessageBox.warning(self, "提示", "至少需要保留一个分类")
            return
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除分类 '{self.classes[index]}' 吗？\n已有的标注数据可能需要重新标注。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.classes[index]
            self.save_classes()
            self.rebuild_items()
            
            if self.current_index >= len(self.classes):
                self.current_index = len(self.classes) - 1
            if self.current_index >= 0:
                self.set_current_index(self.current_index)
            
            self.classes_updated.emit(self.classes)
    
    def on_item_renamed(self, index: int, new_name: str):
        if new_name in self.classes and self.classes[index] != new_name:
            QMessageBox.warning(self, "提示", f"分类 '{new_name}' 已存在")
            return
        
        self.classes[index] = new_name
        self.save_classes()
        self.classes_updated.emit(self.classes)
    
    def add_new_class(self):
        new_name = "新分类"
        counter = 1
        while new_name in self.classes:
            new_name = f"新分类{counter}"
            counter += 1
        
        self.classes.append(new_name)
        self.save_classes()
        
        self.items_layout.takeAt(self.items_layout.count() - 1)
        self.add_class_item(new_name, len(self.classes) - 1)
        self.items_layout.addStretch()
        
        new_item = self.class_items[-1]
        new_item.start_editing()
        
        self.set_current_index(len(self.classes) - 1)
        self.classes_updated.emit(self.classes)
    
    def set_current_index(self, index: int):
        if 0 <= index < len(self.classes):
            old_index = self.current_index
            self.current_index = index
            
            for i, item in enumerate(self.class_items):
                is_selected = (i == index)
                item.set_selected(is_selected)
            
            self.class_changed.emit(index)
            
            if 0 <= index < len(self.class_items):
                selected_item = self.class_items[index]
                self.scroll_area.ensureWidgetVisible(selected_item)
    
    def get_current_index(self) -> int:
        return self.current_index
    
    def get_current_class(self) -> str:
        if 0 <= self.current_index < len(self.classes):
            return self.classes[self.current_index]
        return ""
    
    def save_classes(self):
        if self.project_manager and self.project_manager.current_project:
            self.project_manager.current_project.classes = self.classes
            self.project_manager.config_manager.save_project_config(
                self.project_manager.current_project
            )
            
            classes_file = self.project_manager.current_project.base_dir / "classes.txt"
            with open(classes_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.classes))
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Insert:
            self.add_new_class()
        elif event.key() == Qt.Key_Delete and self.current_index >= 0:
            self.on_item_deleted(self.current_index)
        elif event.key() == Qt.Key_F2 and self.current_index >= 0:
            if self.current_index < len(self.class_items):
                self.class_items[self.current_index].start_editing()
        elif event.key() == Qt.Key_Up:
            new_index = max(0, self.current_index - 1)
            self.set_current_index(new_index)
        elif event.key() == Qt.Key_Down:
            new_index = min(len(self.classes) - 1, self.current_index + 1)
            self.set_current_index(new_index)
        else:
            super().keyPressEvent(event)


class AnnotationCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap: Optional[QPixmap] = None
        self.display_pixmap: Optional[QPixmap] = None
        self.base_scale_factor = 1.0
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_boxes: List[Tuple[QRect, int]] = []
        self.selected_box_index = -1
        
        self.history_stack: List[List[Tuple[QRect, int]]] = []
        self.max_history = 50
        
        self.classes: List[str] = []
        self.current_class_index = 0
        
        self.panning = False
        self.pan_start_pos = QPoint()
        
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setFocusPolicy(Qt.StrongFocus)
        
    def set_image(self, image_path: str):
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            return False
        self.current_boxes = []
        self.selected_box_index = -1
        self.history_stack = []
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.update_display()
        return True
    
    def update_display(self):
        if not self.original_pixmap:
            return
        
        canvas_size = self.size()
        img_size = self.original_pixmap.size()
        
        scale_x = canvas_size.width() / img_size.width()
        scale_y = canvas_size.height() / img_size.height()
        self.base_scale_factor = min(scale_x, scale_y, 1.0)
        
        effective_scale = self.base_scale_factor * self.zoom_factor
        
        new_width = int(img_size.width() * effective_scale)
        new_height = int(img_size.height() * effective_scale)
        
        self.display_pixmap = self.original_pixmap.scaled(
            new_width, new_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        base_offset_x = (canvas_size.width() - int(img_size.width() * self.base_scale_factor)) // 2
        base_offset_y = (canvas_size.height() - int(img_size.height() * self.base_scale_factor)) // 2
        
        self.offset_x = base_offset_x + self.pan_offset_x
        self.offset_y = base_offset_y + self.pan_offset_y
        
        self.draw_boxes()
    
    def draw_boxes(self):
        if not self.display_pixmap:
            return
        
        temp_pixmap = QPixmap(self.display_pixmap)
        painter = QPainter(temp_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        colors = [
            QColor(255, 82, 82),
            QColor(76, 175, 80),
            QColor(33, 150, 243),
            QColor(255, 193, 7),
            QColor(156, 39, 176),
            QColor(0, 188, 212),
            QColor(255, 87, 34),
            QColor(139, 195, 74),
        ]
        
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        
        effective_scale = self.base_scale_factor * self.zoom_factor
        
        for i, (rect, class_idx) in enumerate(self.current_boxes):
            color = colors[class_idx % len(colors)]
            
            scaled_rect = QRect(
                int(rect.x() * effective_scale),
                int(rect.y() * effective_scale),
                int(rect.width() * effective_scale),
                int(rect.height() * effective_scale)
            )
            
            pen = QPen(color, 2)
            
            if i == self.selected_box_index:
                pen.setWidth(3)
                pen.setStyle(Qt.DashLine)
            
            painter.setPen(pen)
            painter.drawRect(scaled_rect)
            
            label = self.classes[class_idx] if class_idx < len(self.classes) else f"Class {class_idx}"
            text_height = 16
            text_rect = QRect(scaled_rect.x(), scaled_rect.y() - text_height - 2, 
                             scaled_rect.width(), text_height)
            
            painter.fillRect(text_rect, color)
            painter.setPen(QPen(Qt.white))
            
            metrics = painter.fontMetrics()
            elided_label = metrics.elidedText(label, Qt.ElideRight, scaled_rect.width() - 4)
            painter.drawText(text_rect, Qt.AlignCenter, elided_label)
        
        if self.drawing:
            current_rect = QRect(self.start_point, self.end_point).normalized()
            pen = QPen(QColor(255, 255, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(current_rect)
        
        painter.end()
        
        final_pixmap = QPixmap(self.size())
        final_pixmap.fill(QColor(26, 26, 26))
        
        painter = QPainter(final_pixmap)
        painter.drawPixmap(self.offset_x, self.offset_y, temp_pixmap)
        painter.end()
        
        self.setPixmap(final_pixmap)
    
    def save_to_history(self):
        import copy
        self.history_stack.append(copy.deepcopy(self.current_boxes))
        if len(self.history_stack) > self.max_history:
            self.history_stack.pop(0)
    
    def undo(self):
        if self.history_stack:
            import copy
            self.current_boxes = self.history_stack.pop()
            self.selected_box_index = -1
            self.draw_boxes()
            return True
        return False
    
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo()
        elif event.key() == Qt.Key_Delete and self.selected_box_index >= 0:
            self.save_to_history()
            del self.current_boxes[self.selected_box_index]
            self.selected_box_index = -1
            self.draw_boxes()
        super().keyPressEvent(event)
    
    def wheelEvent(self, event):
        if not self.original_pixmap:
            return
        
        delta = event.angleDelta().y()
        zoom_step = 0.1
        
        if delta > 0:
            new_zoom = min(5.0, self.zoom_factor + zoom_step)
        else:
            new_zoom = max(0.1, self.zoom_factor - zoom_step)
        
        self.zoom_factor = new_zoom
        self.update_display()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        
        if event.button() == Qt.LeftButton and self.display_pixmap:
            pos = event.pos()
            canvas_rect = QRect(
                self.offset_x, self.offset_y,
                self.display_pixmap.width(), self.display_pixmap.height()
            )
            
            if canvas_rect.contains(pos):
                effective_scale = self.base_scale_factor * self.zoom_factor
                
                for i, (rect, _) in enumerate(self.current_boxes):
                    scaled_rect = QRect(
                        int(rect.x() * effective_scale),
                        int(rect.y() * effective_scale),
                        int(rect.width() * effective_scale),
                        int(rect.height() * effective_scale)
                    )
                    if scaled_rect.contains(pos.x() - self.offset_x, pos.y() - self.offset_y):
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
                self.save_to_history()
                del self.current_boxes[self.selected_box_index]
                self.selected_box_index = -1
                self.draw_boxes()
    
    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.pan_start_pos
            self.pan_offset_x += delta.x()
            self.pan_offset_y += delta.y()
            self.pan_start_pos = event.pos()
            self.update_display()
            return
        
        if self.drawing and self.display_pixmap:
            pos = event.pos()
            self.end_point = QPoint(
                pos.x() - self.offset_x,
                pos.y() - self.offset_y
            )
            self.draw_boxes()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, self.end_point).normalized()
            
            if rect.width() > 10 and rect.height() > 10:
                self.save_to_history()
                
                effective_scale = self.base_scale_factor * self.zoom_factor
                
                original_rect = QRect(
                    int(rect.x() / effective_scale),
                    int(rect.y() / effective_scale),
                    int(rect.width() / effective_scale),
                    int(rect.height() / effective_scale)
                )
                
                self.current_boxes.append((original_rect, self.current_class_index))
            
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
        if self.current_boxes:
            self.save_to_history()
        self.current_boxes = []
        self.selected_box_index = -1
        self.draw_boxes()
    
    def zoom_in(self):
        self.zoom_factor = min(5.0, self.zoom_factor + 0.2)
        self.update_display()
    
    def zoom_out(self):
        self.zoom_factor = max(0.1, self.zoom_factor - 0.2)
        self.update_display()
    
    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.update_display()


class TrainingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    metrics_update = pyqtSignal(dict)
    epoch_update = pyqtSignal(int, int)
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self._is_running = True
    
    def run(self):
        try:
            self.progress.emit("正在准备训练环境...")
            success, msg = self.trainer.prepare_training()
            if not success:
                self.finished.emit(False, msg)
                return
            
            self.progress.emit("正在加载模型...")
            success, msg = self.trainer.load_model()
            if not success:
                self.finished.emit(False, msg)
                return
            
            self.progress.emit("开始训练模型...")
            success, msg = self.trainer.train()
            
            if success:
                self.progress.emit("正在验证模型...")
                valid, metrics = self.trainer.validate_model()
                if valid:
                    self.metrics_update.emit(metrics)
            
            self.finished.emit(success, msg)
            
        except Exception as e:
            self.finished.emit(False, f"训练过程出错: {str(e)}")
    
    def stop(self):
        self._is_running = False


class ONNXConversionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, dict)
    
    def __init__(self, converter, model_path):
        super().__init__()
        self.converter = converter
        self.model_path = model_path
    
    def _clear_gpu_memory(self):
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except Exception:
            pass
    
    def run(self):
        try:
            self._clear_gpu_memory()
            self.progress.emit("正在转换为ONNX格式...")
            success, msg = self.converter.convert_to_onnx(self.model_path)
            
            onnx_info = {}
            if success:
                try:
                    self.progress.emit("正在验证ONNX模型...")
                    self._clear_gpu_memory()
                    valid, onnx_info = self.converter.verify_onnx_model(msg)
                    if not valid:
                        onnx_info = {'error': 'ONNX验证失败', 'file': msg}
                except Exception as ve:
                    print(f"验证过程出错: {ve}")
                    onnx_info = {'note': '跳过验证', 'file': msg}
            
            self.finished.emit(success, msg, onnx_info)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"ONNX转换Worker错误详情:\n{error_detail}")
            self.finished.emit(False, f"转换过程出错: {str(e)}", {})
        finally:
            self._clear_gpu_memory()


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建新项目")
        self.setMinimumWidth(450)
        self.setStyleSheet(f"QDialog {{ background-color: {Styles.BACKGROUND}; }}")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        title = QLabel("创建新的YOLO项目")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212121;")
        layout.addWidget(title)
        
        form_group = QGroupBox("项目信息")
        form_group.setStyleSheet(Styles.GROUP_BOX)
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(10)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入项目名称（英文）")
        self.name_edit.setStyleSheet(Styles.INPUT)
        form_layout.addRow("项目名称:", self.name_edit)
        
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("用逗号分隔，如: person,car,dog")
        self.classes_edit.setStyleSheet(Styles.INPUT)
        form_layout.addRow("类别列表:", self.classes_edit)
        
        layout.addWidget(form_group)
        
        hint_label = QLabel("提示: 项目名称将用于创建文件夹，请使用英文和数字")
        hint_label.setStyleSheet("color: #757575; font-size: 11px;")
        layout.addWidget(hint_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(Styles.BUTTON_OUTLINE)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        create_btn = QPushButton("创建项目")
        create_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        create_btn.clicked.connect(self.accept)
        btn_layout.addWidget(create_btn)
        
        layout.addLayout(btn_layout)
    
    def get_project_info(self):
        name = self.name_edit.text().strip()
        classes_text = self.classes_edit.text().strip()
        classes = [c.strip() for c in classes_text.split(',') if c.strip()]
        return name, classes


class ModelSelectionDialog(QDialog):
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("选择模型")
        self.setMinimumWidth(400)
        self.setStyleSheet(f"QDialog {{ background-color: {Styles.BACKGROUND}; }}")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        title = QLabel("选择训练模型")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212121;")
        layout.addWidget(title)
        
        form_group = QGroupBox("模型配置")
        form_group.setStyleSheet(Styles.GROUP_BOX)
        form_layout = QFormLayout(form_group)
        
        self.model_combo = QComboBox()
        models = self.config_manager.get_available_models()
        self.model_combo.addItems(models.keys())
        self.model_combo.currentTextChanged.connect(self.update_sizes)
        self.model_combo.setStyleSheet(Styles.INPUT)
        form_layout.addRow("模型类型:", self.model_combo)
        
        self.size_combo = QComboBox()
        self.update_sizes(self.model_combo.currentText())
        self.size_combo.setStyleSheet(Styles.INPUT)
        form_layout.addRow("模型尺寸:", self.size_combo)
        
        layout.addWidget(form_group)
        
        info_group = QGroupBox("模型尺寸说明")
        info_group.setStyleSheet(Styles.GROUP_BOX)
        info_layout = QVBoxLayout(info_group)
        
        size_info = QLabel(
            "n/nano - 最小最快，适合边缘设备\n"
            "s/small - 小型模型，速度与精度平衡\n"
            "m/medium - 中型模型，推荐使用\n"
            "l/large - 大型模型，精度更高\n"
            "x/extra - 最大最准，需要更多资源"
        )
        size_info.setStyleSheet("color: #616161; font-size: 12px;")
        info_layout.addWidget(size_info)
        layout.addWidget(info_group)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(Styles.BUTTON_OUTLINE)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        confirm_btn = QPushButton("确认选择")
        confirm_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(confirm_btn)
        
        layout.addLayout(btn_layout)
    
    def update_sizes(self, model_type):
        self.size_combo.clear()
        models = self.config_manager.get_available_models()
        if model_type in models:
            self.size_combo.addItems(models[model_type]['sizes'])
    
    def get_selection(self):
        return self.model_combo.currentText(), self.size_combo.currentText()


class ClassManagerDialog(QDialog):
    def __init__(self, project_manager, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.classes = []
        self.class_colors = {}
        
        if project_manager.current_project:
            self.classes = list(project_manager.current_project.classes)
            self._generate_colors()
        
        self.setWindowTitle("分类管理")
        self.setMinimumSize(500, 450)
        self.setStyleSheet(f"QDialog {{ background-color: {Styles.BACKGROUND}; }}")
        self.init_ui()
    
    def _generate_colors(self):
        colors = [
            '#FF5252', '#4CAF50', '#2196F3', '#FFC107', '#9C27B0',
            '#00BCD4', '#FF5722', '#8BC34A', '#3F51B5', '#795548',
            '#E91E63', '#009688', '#FF9800', '#673AB7', '#607D8B'
        ]
        for i, cls in enumerate(self.classes):
            self.class_colors[cls] = colors[i % len(colors)]
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        title = QLabel("标注分类管理")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212121;")
        layout.addWidget(title)
        
        info_label = QLabel("管理项目的标注分类，新增、编辑或删除分类")
        info_label.setStyleSheet("color: #616161; font-size: 12px;")
        layout.addWidget(info_label)
        
        list_group = QGroupBox("分类列表")
        list_group.setStyleSheet(Styles.GROUP_BOX)
        list_layout = QVBoxLayout(list_group)
        
        self.class_list = QListWidget()
        self.class_list.setStyleSheet(Styles.LIST_ITEM)
        self.class_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.class_list.setToolTip(
            "【分类列表】\n"
            "显示当前项目的所有标注分类\n\n"
            "操作方式:\n"
            "• 选中分类后可编辑或删除\n"
            "• 双击分类可快速编辑"
        )
        self.class_list.itemDoubleClicked.connect(self.edit_selected_class)
        list_layout.addWidget(self.class_list)
        
        btn_row = QHBoxLayout()
        
        self.btn_add = QPushButton("➕ 新增分类")
        self.btn_add.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_add.clicked.connect(self.add_class)
        self.btn_add.setToolTip(
            "【新增分类】\n"
            "添加新的标注分类\n\n"
            "示例:\n"
            "• 队友\n"
            "• 敌人\n"
            "• 怪物\n"
            "• NPC"
        )
        btn_row.addWidget(self.btn_add)
        
        self.btn_edit = QPushButton("✏️ 编辑")
        self.btn_edit.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_edit.clicked.connect(self.edit_selected_class)
        btn_row.addWidget(self.btn_edit)
        
        self.btn_delete = QPushButton("🗑️ 删除")
        self.btn_delete.setStyleSheet(Styles.BUTTON_DANGER)
        self.btn_delete.clicked.connect(self.delete_selected_class)
        self.btn_delete.setToolTip(
            "【删除分类】\n"
            "删除选中的分类\n\n"
            "注意:\n"
            "• 删除分类会影响已有标注\n"
            "• 请谨慎操作"
        )
        btn_row.addWidget(self.btn_delete)
        
        list_layout.addLayout(btn_row)
        layout.addWidget(list_group)
        
        add_group = QGroupBox("快速添加")
        add_group.setStyleSheet(Styles.GROUP_BOX)
        add_layout = QVBoxLayout(add_group)
        
        input_row = QHBoxLayout()
        
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("输入新分类名称...")
        self.new_class_edit.setStyleSheet(Styles.INPUT)
        self.new_class_edit.returnPressed.connect(self.add_class_from_input)
        self.new_class_edit.setToolTip(
            "【分类名称】\n"
            "输入新分类的名称\n\n"
            "命名建议:\n"
            "• 使用简洁明了的名称\n"
            "• 避免使用特殊字符\n"
            "• 建议使用中文或英文"
        )
        input_row.addWidget(self.new_class_edit)
        
        self.btn_quick_add = QPushButton("添加")
        self.btn_quick_add.setStyleSheet(Styles.BUTTON_SUCCESS)
        self.btn_quick_add.clicked.connect(self.add_class_from_input)
        input_row.addWidget(self.btn_quick_add)
        
        add_layout.addLayout(input_row)
        
        batch_row = QHBoxLayout()
        
        self.batch_edit = QLineEdit()
        self.batch_edit.setPlaceholderText("批量添加，用逗号分隔，如: 队友,敌人,怪物,NPC")
        self.batch_edit.setStyleSheet(Styles.INPUT)
        batch_row.addWidget(self.batch_edit)
        
        self.btn_batch_add = QPushButton("批量添加")
        self.btn_batch_add.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_batch_add.clicked.connect(self.batch_add_classes)
        self.btn_batch_add.setToolTip(
            "【批量添加】\n"
            "一次添加多个分类\n\n"
            "格式:\n"
            "• 用逗号分隔分类名称\n"
            "• 示例: 队友,敌人,怪物,NPC"
        )
        batch_row.addWidget(self.btn_batch_add)
        
        add_layout.addLayout(batch_row)
        layout.addWidget(add_group)
        
        stats_label = QLabel(f"当前共 {len(self.classes)} 个分类")
        stats_label.setStyleSheet("color: #757575; font-size: 11px;")
        self.stats_label = stats_label
        layout.addWidget(stats_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(Styles.BUTTON_OUTLINE)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("保存更改")
        save_btn.setStyleSheet(Styles.BUTTON_SUCCESS)
        save_btn.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
        
        self.refresh_class_list()
    
    def refresh_class_list(self):
        self.class_list.clear()
        
        for i, cls in enumerate(self.classes):
            color = self.class_colors.get(cls, '#2196F3')
            item = QListWidgetItem(f"  {i+1}. {cls}")
            item.setForeground(QColor(color))
            item.setData(Qt.UserRole, cls)
            self.class_list.addItem(item)
        
        self.stats_label.setText(f"当前共 {len(self.classes)} 个分类")
    
    def add_class(self):
        dialog = ClassEditDialog("", self)
        if dialog.exec_() == QDialog.Accepted:
            name = dialog.get_name().strip()
            if name:
                if name in self.classes:
                    QMessageBox.warning(self, "提示", f"分类 '{name}' 已存在")
                    return
                self.classes.append(name)
                self._generate_colors()
                self.refresh_class_list()
    
    def add_class_from_input(self):
        name = self.new_class_edit.text().strip()
        if not name:
            return
        
        if name in self.classes:
            QMessageBox.warning(self, "提示", f"分类 '{name}' 已存在")
            return
        
        self.classes.append(name)
        self._generate_colors()
        self.refresh_class_list()
        self.new_class_edit.clear()
    
    def batch_add_classes(self):
        text = self.batch_edit.text().strip()
        if not text:
            return
        
        names = [n.strip() for n in text.split(',') if n.strip()]
        added = 0
        
        for name in names:
            if name and name not in self.classes:
                self.classes.append(name)
                added += 1
        
        if added > 0:
            self._generate_colors()
            self.refresh_class_list()
            self.batch_edit.clear()
            QMessageBox.information(self, "成功", f"成功添加 {added} 个分类")
        else:
            QMessageBox.warning(self, "提示", "没有新分类被添加（可能全部已存在）")
    
    def edit_selected_class(self):
        current_item = self.class_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "提示", "请先选择要编辑的分类")
            return
        
        old_name = current_item.data(Qt.UserRole)
        dialog = ClassEditDialog(old_name, self)
        
        if dialog.exec_() == QDialog.Accepted:
            new_name = dialog.get_name().strip()
            if new_name and new_name != old_name:
                if new_name in self.classes:
                    QMessageBox.warning(self, "提示", f"分类 '{new_name}' 已存在")
                    return
                
                index = self.classes.index(old_name)
                self.classes[index] = new_name
                
                if old_name in self.class_colors:
                    self.class_colors[new_name] = self.class_colors.pop(old_name)
                
                self.refresh_class_list()
    
    def delete_selected_class(self):
        current_item = self.class_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "提示", "请先选择要删除的分类")
            return
        
        name = current_item.data(Qt.UserRole)
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除分类 '{name}' 吗？\n\n"
            f"注意: 已有的标注数据可能需要重新标注",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.classes.remove(name)
            if name in self.class_colors:
                del self.class_colors[name]
            self.refresh_class_list()
    
    def save_and_accept(self):
        if not self.classes:
            QMessageBox.warning(self, "提示", "至少需要一个分类")
            return
        
        if self.project_manager.current_project:
            self.project_manager.current_project.classes = self.classes
            self.project_manager.config_manager.save_project_config(
                self.project_manager.current_project
            )
            
            classes_file = self.project_manager.current_project.base_dir / "classes.txt"
            with open(classes_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.classes))
        
        self.accept()
    
    def get_classes(self):
        return self.classes


class ClassEditDialog(QDialog):
    def __init__(self, current_name="", parent=None):
        super().__init__(parent)
        self.current_name = current_name
        self.setWindowTitle("编辑分类" if current_name else "新增分类")
        self.setMinimumWidth(350)
        self.setStyleSheet(f"QDialog {{ background-color: {Styles.BACKGROUND}; }}")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        title = QLabel("编辑分类" if self.current_name else "新增分类")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #212121;")
        layout.addWidget(title)
        
        form_group = QGroupBox("分类信息")
        form_group.setStyleSheet(Styles.GROUP_BOX)
        form_layout = QFormLayout(form_group)
        
        self.name_edit = QLineEdit()
        self.name_edit.setText(self.current_name)
        self.name_edit.setPlaceholderText("输入分类名称")
        self.name_edit.setStyleSheet(Styles.INPUT)
        self.name_edit.setToolTip(
            "【分类名称】\n"
            "输入标注分类的名称\n\n"
            "命名建议:\n"
            "• 简洁明了\n"
            "• 易于理解\n"
            "• 避免特殊字符"
        )
        form_layout.addRow("名称:", self.name_edit)
        
        layout.addWidget(form_group)
        
        hint = QLabel("提示: 分类名称将用于标注和模型训练")
        hint.setStyleSheet("color: #757575; font-size: 11px;")
        layout.addWidget(hint)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(Styles.BUTTON_OUTLINE)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("确定")
        save_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        save_btn.clicked.connect(self.validate_and_accept)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
    
    def validate_and_accept(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入分类名称")
            return
        self.accept()
    
    def get_name(self):
        return self.name_edit.text().strip()


class ProjectPanel(QWidget):
    project_loaded = pyqtSignal()
    
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.init_ui()
        self.refresh_project_list()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        header = QLabel("项目管理")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #212121; padding: 10px;")
        layout.addWidget(header)
        
        btn_layout = QHBoxLayout()
        
        self.btn_new = QPushButton("  新建项目")
        self.btn_new.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_new.clicked.connect(self.create_project)
        btn_layout.addWidget(self.btn_new)
        
        self.btn_open = QPushButton("  打开项目")
        self.btn_open.setStyleSheet(Styles.BUTTON_SUCCESS)
        self.btn_open.clicked.connect(self.open_project)
        btn_layout.addWidget(self.btn_open)
        
        self.btn_refresh = QPushButton("  刷新列表")
        self.btn_refresh.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_refresh.clicked.connect(self.refresh_project_list)
        btn_layout.addWidget(self.btn_refresh)
        
        layout.addLayout(btn_layout)
        
        self.project_list = QListWidget()
        self.project_list.setStyleSheet(Styles.LIST_ITEM)
        self.project_list.itemDoubleClicked.connect(self.open_project)
        layout.addWidget(self.project_list)
        
        info_group = QGroupBox("当前项目信息")
        info_group.setStyleSheet(Styles.GROUP_BOX)
        info_layout = QVBoxLayout(info_group)
        
        self.project_name_label = QLabel("未加载项目")
        self.project_name_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")
        info_layout.addWidget(self.project_name_label)
        
        self.project_info_label = QLabel("")
        self.project_info_label.setStyleSheet("color: #616161;")
        info_layout.addWidget(self.project_info_label)
        
        self.project_stats_label = QLabel("")
        self.project_stats_label.setStyleSheet("color: #616161;")
        info_layout.addWidget(self.project_stats_label)
        
        layout.addWidget(info_group)
    
    def refresh_project_list(self):
        self.project_list.clear()
        projects = self.project_manager.list_all_projects()
        
        for project in projects:
            item = QListWidgetItem(
                f"📁 {project['name']}\n"
                f"   模型: {project['model']} | 创建: {project['created_at']}"
            )
            item.setData(Qt.UserRole, project['name'])
            item.setSizeHint(QSize(0, 50))
            self.project_list.addItem(item)
    
    def create_project(self):
        dialog = NewProjectDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name, classes = dialog.get_project_info()
            
            if not name:
                QMessageBox.warning(self, "输入错误", "请输入项目名称")
                return
            
            if not classes:
                QMessageBox.warning(self, "输入错误", "请输入至少一个类别")
                return
            
            try:
                config = self.project_manager.create_project_structure(name, classes)
                self.project_manager.current_project = config
                self.update_project_info()
                self.refresh_project_list()
                self.project_loaded.emit()
                QMessageBox.information(self, "创建成功", f"项目 '{name}' 已创建")
            except Exception as e:
                QMessageBox.critical(self, "创建失败", f"无法创建项目: {str(e)}")
    
    def open_project(self):
        current_item = self.project_list.currentItem()
        if not current_item:
            if self.project_list.count() > 0:
                self.project_list.setCurrentRow(0)
                current_item = self.project_list.currentItem()
            else:
                QMessageBox.warning(self, "提示", "请先创建或选择一个项目")
                return
        
        project_name = current_item.data(Qt.UserRole)
        
        try:
            self.project_manager.load_project(project_name)
            self.update_project_info()
            self.project_loaded.emit()
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"无法打开项目: {str(e)}")
    
    def update_project_info(self):
        if not self.project_manager.current_project:
            return
        
        project = self.project_manager.current_project
        self.project_name_label.setText(f"📁 {project.project_name}")
        
        self.project_info_label.setText(
            f"模型: {project.model_config.full_name}\n"
            f"类别: {', '.join(project.classes)}"
        )
        
        stats = self.project_manager.get_annotation_stats()
        self.project_stats_label.setText(
            f"图像: {stats['labeled_images']}/{stats['total_images']} 已标注\n"
            f"标注框: {stats['total_annotations']}"
        )


class AnnotationPanel(QWidget):
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.current_image_path: Optional[Path] = None
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        left_panel = self.create_left_panel()
        layout.addWidget(left_panel, 1)
        
        center_panel = self.create_center_panel()
        layout.addWidget(center_panel, 3)
        
        right_panel = self.create_right_panel()
        layout.addWidget(right_panel, 1)
    
    def create_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; border-right: 1px solid {Styles.BORDER}; }}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        header = QLabel("图像列表")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #212121;")
        layout.addWidget(header)
        
        btn_layout = QHBoxLayout()
        
        self.btn_import = QPushButton("导入")
        self.btn_import.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_import.clicked.connect(self.import_images)
        self.btn_import.setToolTip(
            "【导入图像】\n"
            "将图像文件导入到当前项目\n\n"
            "支持格式:\n"
            "• JPG / JPEG\n"
            "• PNG\n"
            "• BMP\n\n"
            "操作说明:\n"
            "• 支持多选批量导入\n"
            "• 图像会复制到项目目录\n"
            "• 重复文件会自动跳过"
        )
        btn_layout.addWidget(self.btn_import)
        
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_refresh.clicked.connect(self.refresh_image_list)
        self.btn_refresh.setToolTip(
            "【刷新列表】\n"
            "更新图像列表显示\n\n"
            "刷新内容:\n"
            "• 更新已标注/未标注状态\n"
            "• 更新标注进度\n"
            "• 同步文件变化"
        )
        btn_layout.addWidget(self.btn_refresh)
        
        layout.addLayout(btn_layout)
        
        self.image_list = QListWidget()
        self.image_list.setStyleSheet(Styles.LIST_ITEM)
        self.image_list.itemDoubleClicked.connect(self.on_image_double_clicked)
        self.image_list.setToolTip(
            "【图像列表】\n"
            "显示项目中所有图像\n\n"
            "状态标识:\n"
            "• 红色: 未标注图像\n"
            "• 绿色: 已标注图像\n\n"
            "操作方式:\n"
            "• 双击图像打开标注\n"
            "• 使用导航按钮切换"
        )
        layout.addWidget(self.image_list)
        
        nav_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setToolTip(
            "【上一张图像】\n"
            "切换到上一张图像\n\n"
            "快捷键: 可自定义"
        )
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setToolTip(
            "【下一张图像】\n"
            "切换到下一张图像\n\n"
            "快捷键: 可自定义"
        )
        nav_layout.addWidget(self.btn_next)
        
        layout.addLayout(nav_layout)
        
        self.progress_label = QLabel("进度: 0/0")
        self.progress_label.setStyleSheet("color: #616161; font-size: 12px;")
        self.progress_label.setToolTip(
            "【标注进度】\n"
            "显示当前标注完成情况\n\n"
            "格式: 已标注数/总数"
        )
        layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(Styles.PROGRESS)
        self.progress_bar.setToolTip(
            "【进度条】\n"
            "可视化显示标注完成比例"
        )
        layout.addWidget(self.progress_bar)
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: #1a1a1a; }}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = AnnotationCanvas()
        
        toolbar = QFrame()
        toolbar.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; border-bottom: 1px solid {Styles.BORDER}; }}")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        self.btn_clear = QPushButton("清除标注")
        self.btn_clear.setStyleSheet(Styles.BUTTON_WARNING)
        self.btn_clear.clicked.connect(self.clear_annotations)
        self.btn_clear.setToolTip(
            "【清除标注】\n"
            "清除当前图像的所有标注框\n\n"
            "注意:\n"
            "• 仅清除画布上的标注\n"
            "• 不会删除已保存的文件\n"
            "• 需要保存才会生效"
        )
        toolbar_layout.addWidget(self.btn_clear)
        
        self.btn_save = QPushButton("保存标注")
        self.btn_save.setStyleSheet(Styles.BUTTON_SUCCESS)
        self.btn_save.clicked.connect(self.save_annotations)
        self.btn_save.setToolTip(
            "【保存标注】\n"
            "保存当前图像的标注结果\n\n"
            "保存内容:\n"
            "• YOLO格式标注文件\n"
            "• 每行一个标注框\n"
            "• 格式: class x_center y_center width height\n\n"
            "快捷键: Ctrl+S"
        )
        toolbar_layout.addWidget(self.btn_save)
        
        self.btn_delete = QPushButton("删除图像")
        self.btn_delete.setStyleSheet(Styles.BUTTON_DANGER)
        self.btn_delete.clicked.connect(self.delete_image)
        self.btn_delete.setToolTip(
            "【删除图像】\n"
            "删除当前图像及其标注\n\n"
            "注意:\n"
            "• 同时删除图像文件和标注文件\n"
            "• 删除后无法恢复\n"
            "• 请谨慎操作"
        )
        toolbar_layout.addWidget(self.btn_delete)
        
        toolbar_layout.addWidget(self._create_separator())
        
        self.btn_zoom_in = QPushButton("🔍+")
        self.btn_zoom_in.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_zoom_in.setMinimumWidth(50)
        self.btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        self.btn_zoom_in.setToolTip(
            "【放大】\n"
            "放大图像显示\n\n"
            "快捷方式:\n"
            "• 鼠标滚轮向上"
        )
        toolbar_layout.addWidget(self.btn_zoom_in)
        
        self.btn_zoom_out = QPushButton("🔍-")
        self.btn_zoom_out.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_zoom_out.setMinimumWidth(50)
        self.btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        self.btn_zoom_out.setToolTip(
            "【缩小】\n"
            "缩小图像显示\n\n"
            "快捷方式:\n"
            "• 鼠标滚轮向下"
        )
        toolbar_layout.addWidget(self.btn_zoom_out)
        
        self.btn_reset_zoom = QPushButton("重置")
        self.btn_reset_zoom.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_reset_zoom.setMinimumWidth(60)
        self.btn_reset_zoom.clicked.connect(self.canvas.reset_zoom)
        self.btn_reset_zoom.setToolTip(
            "【重置视图】\n"
            "重置缩放和平移\n\n"
            "恢复到默认显示状态"
        )
        toolbar_layout.addWidget(self.btn_reset_zoom)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #616161; font-size: 11px; min-width: 40px;")
        toolbar_layout.addWidget(self.zoom_label)
        
        toolbar_layout.addStretch()
        
        self.image_info_label = QLabel("双击图像列表中的图片开始标注")
        self.image_info_label.setStyleSheet("color: #616161;")
        self.image_info_label.setToolTip(
            "【图像信息】\n"
            "显示当前图像的文件名和状态"
        )
        toolbar_layout.addWidget(self.image_info_label)
        
        layout.addWidget(toolbar)
        
        self.canvas.setToolTip(
            "【标注画布】\n"
            "图像标注工作区域\n\n"
            "操作方式:\n"
            "• 左键拖动: 绘制边界框\n"
            "• 左键点击框: 选中标注框\n"
            "• 右键点击: 删除选中的标注框\n"
            "• 鼠标滚轮: 缩放图像\n"
            "• 中键拖动: 平移图像\n\n"
            "快捷键:\n"
            "• Ctrl+Z: 撤销上一步操作\n"
            "• Delete: 删除选中标注框\n\n"
            "标注技巧:\n"
            "• 边界框应紧密包围目标\n"
            "• 避免过大或过小的框\n"
            "• 确保目标完整在框内"
        )
        layout.addWidget(self.canvas, 1)
        
        return panel
    
    def _create_separator(self) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("background-color: #E0E0E0;")
        separator.setFixedWidth(1)
        separator.setFixedHeight(20)
        return separator
    
    def create_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; border-left: 1px solid {Styles.BORDER}; }}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        header = QLabel("标注工具")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #212121;")
        layout.addWidget(header)
        
        class_group = QGroupBox("分类选择")
        class_group.setStyleSheet(Styles.GROUP_BOX)
        class_layout = QVBoxLayout(class_group)
        
        self.class_selector = ClassSelectorWidget()
        self.class_selector.set_project_manager(self.project_manager)
        self.class_selector.class_changed.connect(self.on_class_changed)
        self.class_selector.classes_updated.connect(self.on_classes_updated)
        self.class_selector.setMinimumHeight(150)
        class_layout.addWidget(self.class_selector)
        
        layout.addWidget(class_group)
        
        annotation_group = QGroupBox("当前标注")
        annotation_group.setStyleSheet(Styles.GROUP_BOX)
        annotation_layout = QVBoxLayout(annotation_group)
        
        self.annotation_list = QListWidget()
        self.annotation_list.setStyleSheet(Styles.LIST_ITEM)
        self.annotation_list.itemClicked.connect(self.on_annotation_clicked)
        self.annotation_list.setToolTip(
            "【标注列表】\n"
            "显示当前图像的所有标注框\n\n"
            "信息内容:\n"
            "• 序号: 标注框编号\n"
            "• 类别: 目标类别名称\n"
            "• 尺寸: 边界框宽x高\n\n"
            "操作方式:\n"
            "• 点击选中对应标注框\n"
            "• 选中后可删除"
        )
        annotation_layout.addWidget(self.annotation_list)
        
        self.btn_delete_annotation = QPushButton("删除选中")
        self.btn_delete_annotation.setStyleSheet(Styles.BUTTON_DANGER)
        self.btn_delete_annotation.clicked.connect(self.delete_annotation)
        self.btn_delete_annotation.setToolTip(
            "【删除选中标注】\n"
            "删除当前选中的标注框\n\n"
            "操作方式:\n"
            "• 先在列表中点击选中\n"
            "• 再点击此按钮删除\n\n"
            "快捷方式:\n"
            "• 右键点击标注框直接删除"
        )
        annotation_layout.addWidget(self.btn_delete_annotation)
        
        layout.addWidget(annotation_group)
        
        action_group = QGroupBox("快捷操作")
        action_group.setStyleSheet(Styles.GROUP_BOX)
        action_layout = QVBoxLayout(action_group)
        
        self.btn_auto_save = QPushButton("保存并继续 (Ctrl+S)")
        self.btn_auto_save.setStyleSheet(Styles.BUTTON_SUCCESS)
        self.btn_auto_save.clicked.connect(self.auto_save_and_next)
        self.btn_auto_save.setToolTip(
            "【保存并继续】\n"
            "保存当前标注并切换到下一张\n\n"
            "功能说明:\n"
            "• 自动保存当前图像标注\n"
            "• 自动加载下一张未标注图像\n"
            "• 提高标注效率\n\n"
            "快捷键: Ctrl+S"
        )
        action_layout.addWidget(self.btn_auto_save)
        
        hint_label = QLabel("快捷键:\n左键拖动 - 绘制框\n右键 - 删除选中框")
        hint_label.setStyleSheet("color: #757575; font-size: 11px;")
        hint_label.setToolTip(
            "【操作提示】\n"
            "标注操作的快捷方式\n\n"
            "• 左键拖动: 绘制新的边界框\n"
            "• 右键点击: 删除选中的标注框"
        )
        action_layout.addWidget(hint_label)
        
        layout.addWidget(action_group)
        
        layout.addStretch()
        
        return panel
    
    def load_project_info(self):
        if not self.project_manager.current_project:
            return
        
        project = self.project_manager.current_project
        self.canvas.set_classes(project.classes)
        
        self.class_selector.set_project_manager(self.project_manager)
        self.class_selector.set_classes(project.classes)
        
        self.refresh_image_list()
    
    def on_classes_updated(self, classes: list):
        self.canvas.set_classes(classes)
        if self.project_manager.current_project:
            self.project_manager.current_project.classes = classes
    
    def refresh_image_list(self):
        self.image_list.clear()
        
        if not self.project_manager.current_project:
            return
        
        stats = self.project_manager.get_annotation_stats()
        unlabeled = self.project_manager.get_unlabeled_images()
        labeled = self.project_manager.get_labeled_images()
        
        self.image_list.addItem(f"── 未标注 ({len(unlabeled)}) ──")
        for img_path in unlabeled:
            item = QListWidgetItem(f"📷 {img_path.name}")
            item.setData(Qt.UserRole, str(img_path))
            item.setForeground(QColor(Styles.ERROR))
            self.image_list.addItem(item)
        
        self.image_list.addItem(f"── 已标注 ({len(labeled)}) ──")
        for img_path in labeled:
            item = QListWidgetItem(f"✅ {img_path.name}")
            item.setData(Qt.UserRole, str(img_path))
            item.setForeground(QColor(Styles.SUCCESS))
            self.image_list.addItem(item)
        
        self.progress_label.setText(f"进度: {stats['labeled_images']}/{stats['total_images']}")
        self.progress_bar.setMaximum(stats['total_images'] if stats['total_images'] > 0 else 1)
        self.progress_bar.setValue(stats['labeled_images'])
    
    def import_images(self):
        if not self.project_manager.current_project:
            QMessageBox.warning(self, "提示", "请先打开一个项目")
            return
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图像文件", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        
        if file_paths:
            added = self.project_manager.add_images(file_paths)
            self.refresh_image_list()
            QMessageBox.information(self, "导入完成", f"成功导入 {added} 张图像")
    
    def on_image_double_clicked(self, item):
        image_path = item.data(Qt.UserRole)
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path: str):
        self.current_image_path = Path(image_path)
        
        if self.canvas.set_image(image_path):
            self.image_info_label.setText(f"📷 {self.current_image_path.name}")
            
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
    
    def save_annotations(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "没有当前图像")
            return
        
        annotations = self.canvas.get_yolo_annotations()
        label_path = self.get_label_path()
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        self.refresh_image_list()
        self.image_info_label.setText(f"✅ 已保存: {self.current_image_path.name}")
    
    def clear_annotations(self):
        self.canvas.clear_annotations()
        self.update_annotation_list()
    
    def delete_image(self):
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
            self.image_info_label.setText("图像已删除")
    
    def prev_image(self):
        current_row = self.image_list.currentRow()
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
        self.save_annotations()
        self.next_image()
    
    def on_class_changed(self, index):
        self.canvas.set_current_class(index)
    
    def on_annotation_clicked(self, item):
        index = self.annotation_list.row(item)
        self.canvas.selected_box_index = index
        self.canvas.draw_boxes()
    
    def delete_annotation(self):
        if self.canvas.selected_box_index >= 0:
            del self.canvas.current_boxes[self.canvas.selected_box_index]
            self.canvas.selected_box_index = -1
            self.canvas.draw_boxes()
            self.update_annotation_list()
    
    def update_annotation_list(self):
        self.annotation_list.clear()
        
        for i, (rect, class_idx) in enumerate(self.canvas.current_boxes):
            class_name = self.canvas.classes[class_idx] if class_idx < len(self.canvas.classes) else f"Class {class_idx}"
            item_text = f"{i+1}. {class_name} ({rect.width()}x{rect.height()})"
            self.annotation_list.addItem(item_text)


class TrainingPanel(QWidget):
    training_started = pyqtSignal()
    training_finished = pyqtSignal(bool)
    
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.trainer = None
        self.training_worker = None
        self.gpu_available = device_manager.is_gpu_available()
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        if not self.gpu_available:
            # GPU不可用时显示警告面板
            warning_panel = self.create_gpu_warning_panel()
            layout.addWidget(warning_panel)
        else:
            left_panel = self.create_config_panel()
            layout.addWidget(left_panel, 1)
            
            right_panel = self.create_monitor_panel()
            layout.addWidget(right_panel, 2)
    
    def create_gpu_warning_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; }}")
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignCenter)
        
        warning_icon = QLabel("⚠️")
        warning_icon.setStyleSheet("font-size: 72px;")
        warning_icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning_icon)
        
        warning_title = QLabel("GPU 不可用")
        warning_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #F44336;")
        warning_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning_title)
        
        error_msg = device_manager.get_gpu_error_message() or "未检测到NVIDIA GPU"
        warning_text = QLabel(f"模型训练功能需要NVIDIA GPU支持\n\n{error_msg}")
        warning_text.setStyleSheet("font-size: 14px; color: #757575;")
        warning_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning_text)
        
        layout.addStretch()
        
        return panel
    
    def create_config_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; border-right: 1px solid {Styles.BORDER}; }}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        
        header = QLabel("训练配置")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #212121;")
        layout.addWidget(header)
        
        model_group = QGroupBox("模型选择")
        model_group.setStyleSheet(Styles.GROUP_BOX)
        model_layout = QFormLayout(model_group)
        
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['yolov8', 'yolov5', 'yolov7', 'yolov9', 'yolov10'])
        self.model_type_combo.setStyleSheet(Styles.INPUT)
        self.model_type_combo.currentTextChanged.connect(self.update_model_sizes)
        self.model_type_combo.setToolTip(
            "【模型类型】\n"
            "选择YOLO目标检测模型的版本\n\n"
            "• YOLOv8: 最新版本，推荐使用，性能均衡\n"
            "• YOLOv5: 成熟稳定，社区资源丰富\n"
            "• YOLOv7: 训练效率高，适合快速实验\n"
            "• YOLOv9: 精度更高，适合精度优先场景\n"
            "• YOLOv10: 无NMS设计，推理速度更快\n\n"
            "建议: 新项目推荐使用YOLOv8"
        )
        model_layout.addRow("模型类型:", self.model_type_combo)
        
        self.model_size_combo = QComboBox()
        self.model_size_combo.setStyleSheet(Styles.INPUT)
        self.update_model_sizes('yolov8')
        self.model_size_combo.setToolTip(
            "【模型尺寸】\n"
            "决定模型的大小和复杂度\n\n"
            "• n (nano): 最小最快，适合边缘设备\n"
            "  - 参数量约3M，速度最快\n"
            "  - 适合移动端、嵌入式设备\n\n"
            "• s (small): 小型模型\n"
            "  - 参数量约11M，速度与精度平衡\n"
            "  - 适合实时检测场景\n\n"
            "• m (medium): 中型模型，推荐\n"
            "  - 参数量约25M，精度较高\n"
            "  - 适合通用检测任务\n\n"
            "• l (large): 大型模型\n"
            "  - 参数量约43M，精度高\n"
            "  - 需要更多GPU资源\n\n"
            "• x (extra): 最大最准\n"
            "  - 参数量约68M，精度最高\n"
            "  - 需要强大的GPU支持\n\n"
            "建议: 初次训练推荐使用n或s快速验证"
        )
        model_layout.addRow("模型尺寸:", self.model_size_combo)
        
        layout.addWidget(model_group)
        
        train_group = QGroupBox("训练参数")
        train_group.setStyleSheet(Styles.GROUP_BOX)
        train_layout = QFormLayout(train_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setStyleSheet(Styles.INPUT)
        self.epochs_spin.setToolTip(
            "【训练轮数 (Epochs)】\n"
            "完整遍历整个数据集的次数\n\n"
            "推荐取值范围:\n"
            "• 快速测试: 10-30轮\n"
            "• 常规训练: 100-300轮\n"
            "• 高精度需求: 300-500轮\n\n"
            "影响:\n"
            "• 轮数过少: 模型欠拟合，精度低\n"
            "• 轮数过多: 可能过拟合，浪费时间\n\n"
            "最佳实践:\n"
            "• 配合早停(patience)使用\n"
            "• 观察loss曲线判断是否收敛\n"
            "• 数据量大时可减少轮数"
        )
        train_layout.addRow("训练轮数:", self.epochs_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        self.batch_spin.setStyleSheet(Styles.INPUT)
        self.batch_spin.setToolTip(
            "【批次大小 (Batch Size)】\n"
            "每次训练迭代使用的样本数量\n\n"
            "推荐取值范围:\n"
            "• 4GB显存: batch=4-8\n"
            "• 8GB显存: batch=16-32\n"
            "• 16GB显存: batch=32-64\n"
            "• 24GB+显存: batch=64-128\n\n"
            "影响:\n"
            "• 批次大: 训练稳定，显存占用高\n"
            "• 批次小: 显存占用低，可能不稳定\n\n"
            "最佳实践:\n"
            "• 从较小值开始逐步增大\n"
            "• 观察GPU显存使用率\n"
            "• 批次大小建议为2的幂次"
        )
        train_layout.addRow("批次大小:", self.batch_spin)
        
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setStyleSheet(Styles.INPUT)
        self.imgsz_spin.setToolTip(
            "【图像尺寸 (Image Size)】\n"
            "训练时图像缩放后的尺寸\n\n"
            "推荐取值:\n"
            "• 快速训练: 320-416\n"
            "• 标准训练: 640 (推荐)\n"
            "• 高精度: 1024-1280\n\n"
            "影响:\n"
            "• 尺寸大: 检测小目标能力强，显存占用高\n"
            "• 尺寸小: 训练速度快，可能漏检小目标\n\n"
            "最佳实践:\n"
            "• 根据目标大小选择\n"
            "• 小目标检测建议640以上\n"
            "• 必须是32的倍数"
        )
        train_layout.addRow("图像尺寸:", self.imgsz_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setStyleSheet(Styles.INPUT)
        self.lr_spin.setToolTip(
            "【学习率 (Learning Rate)】\n"
            "控制模型参数更新的步长\n\n"
            "推荐取值范围:\n"
            "• 保守训练: 0.001-0.005\n"
            "• 标准训练: 0.01 (推荐)\n"
            "• 快速收敛: 0.01-0.1\n\n"
            "影响:\n"
            "• 学习率大: 收敛快，可能震荡或发散\n"
            "• 学习率小: 收敛慢，可能陷入局部最优\n\n"
            "最佳实践:\n"
            "• 从默认值0.01开始\n"
            "• 训练不稳定时降低学习率\n"
            "• 可使用学习率调度器自动调整"
        )
        train_layout.addRow("学习率:", self.lr_spin)
        
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 200)
        self.patience_spin.setValue(50)
        self.patience_spin.setStyleSheet(Styles.INPUT)
        self.patience_spin.setToolTip(
            "【早停耐心值 (Patience)】\n"
            "验证集指标连续多少轮无改善后停止训练\n\n"
            "推荐取值:\n"
            "• 快速实验: 10-20轮\n"
            "• 常规训练: 50轮 (推荐)\n"
            "• 充分训练: 100轮\n\n"
            "影响:\n"
            "• 值小: 可能过早停止，欠拟合\n"
            "• 值大: 训练时间长，可能过拟合\n\n"
            "最佳实践:\n"
            "• 配合epochs使用\n"
            "• 数据量大时可增大\n"
            "• 防止无效训练，节省时间"
        )
        train_layout.addRow("早停耐心值:", self.patience_spin)
        
        layout.addWidget(train_group)
        
        data_group = QGroupBox("数据集")
        data_group.setStyleSheet(Styles.GROUP_BOX)
        data_layout = QVBoxLayout(data_group)
        
        self.dataset_info_label = QLabel("未加载数据集")
        self.dataset_info_label.setStyleSheet("color: #616161;")
        self.dataset_info_label.setToolTip(
            "【数据集信息】\n"
            "显示当前项目的数据集统计\n\n"
            "包含:\n"
            "• 总图像数: 项目中的图像总数\n"
            "• 已标注: 已完成标注的图像数\n"
            "• 标注框: 所有标注框的总数"
        )
        data_layout.addWidget(self.dataset_info_label)
        
        self.btn_split = QPushButton("划分数据集")
        self.btn_split.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_split.clicked.connect(self.split_dataset)
        self.btn_split.setToolTip(
            "【划分数据集】\n"
            "将已标注的数据划分为训练集和验证集\n\n"
            "划分比例:\n"
            "• 训练集: 80%\n"
            "• 验证集: 20%\n\n"
            "注意:\n"
            "• 划分前请确保已完成标注\n"
            "• 每次划分会重新随机分配\n"
            "• 建议至少标注50张以上再划分"
        )
        data_layout.addWidget(self.btn_split)
        
        layout.addWidget(data_group)
        
        layout.addStretch()
        
        self.btn_start = QPushButton("🚀 开始训练")
        self.btn_start.setStyleSheet(Styles.BUTTON_SUCCESS)
        self.btn_start.setMinimumHeight(50)
        self.btn_start.clicked.connect(self.start_training)
        self.btn_start.setToolTip(
            "【开始训练】\n"
            "启动模型训练流程\n\n"
            "训练前请确保:\n"
            "• 已打开项目\n"
            "• 已完成数据标注(至少10张)\n"
            "• 已划分数据集\n"
            "• GPU显存充足\n\n"
            "训练过程中:\n"
            "• 可查看实时日志\n"
            "• 可随时停止训练\n"
            "• 自动保存最佳模型"
        )
        layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("⏹ 停止训练")
        self.btn_stop.setStyleSheet(Styles.BUTTON_DANGER)
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setVisible(False)
        self.btn_stop.setToolTip(
            "【停止训练】\n"
            "手动停止当前训练\n\n"
            "注意:\n"
            "• 停止后会保存当前模型\n"
            "• 建议等待当前epoch完成\n"
            "• 不可恢复训练进度"
        )
        layout.addWidget(self.btn_stop)
        
        return panel
    
    def create_monitor_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; }}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        
        header = QLabel("训练监控")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #212121;")
        layout.addWidget(header)
        
        progress_group = QGroupBox("训练进度")
        progress_group.setStyleSheet(Styles.GROUP_BOX)
        progress_layout = QVBoxLayout(progress_group)
        
        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setStyleSheet(Styles.PROGRESS)
        self.train_progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.train_progress_bar)
        
        self.train_status_label = QLabel("等待开始训练...")
        self.train_status_label.setStyleSheet("color: #616161;")
        progress_layout.addWidget(self.train_status_label)
        
        layout.addWidget(progress_group)
        
        log_group = QGroupBox("训练日志")
        log_group.setStyleSheet(Styles.GROUP_BOX)
        log_layout = QVBoxLayout(log_group)
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("""
            QTextEdit {
                background-color: #263238;
                color: #ECEFF1;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
                border: 1px solid #37474F;
                border-radius: 4px;
            }
        """)
        log_layout.addWidget(self.train_log)
        
        layout.addWidget(log_group, 1)
        
        result_group = QGroupBox("训练结果")
        result_group.setStyleSheet(Styles.GROUP_BOX)
        result_layout = QVBoxLayout(result_group)
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["指标", "值"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #F5F5F5;
                padding: 5px;
                border: none;
                border-bottom: 1px solid #E0E0E0;
                font-weight: bold;
            }
        """)
        result_layout.addWidget(self.result_table)
        
        layout.addWidget(result_group)
        
        return panel
    
    def update_model_sizes(self, model_type):
        self.model_size_combo.clear()
        
        sizes_map = {
            'yolov5': ['n', 's', 'm', 'l', 'x'],
            'yolov7': ['tiny', 's', 'm', 'l', 'x'],
            'yolov8': ['n', 's', 'm', 'l', 'x'],
            'yolov9': ['s', 'm', 'c', 'e'],
            'yolov10': ['n', 's', 'm', 'b', 'l', 'x']
        }
        
        if model_type in sizes_map:
            self.model_size_combo.addItems(sizes_map[model_type])
    
    def load_project_info(self):
        if not self.project_manager.current_project:
            return
        
        project = self.project_manager.current_project
        
        model_type_index = self.model_type_combo.findText(project.model_config.name)
        if model_type_index >= 0:
            self.model_type_combo.setCurrentIndex(model_type_index)
        
        model_size_index = self.model_size_combo.findText(project.model_config.size)
        if model_size_index >= 0:
            self.model_size_combo.setCurrentIndex(model_size_index)
        
        self.epochs_spin.setValue(project.training_config.epochs)
        self.batch_spin.setValue(project.training_config.batch_size)
        self.imgsz_spin.setValue(project.training_config.image_size)
        self.lr_spin.setValue(project.training_config.learning_rate)
        self.patience_spin.setValue(project.training_config.patience)
        
        self.update_dataset_info()
    
    def update_dataset_info(self):
        if not self.project_manager.current_project:
            return
        
        stats = self.project_manager.get_annotation_stats()
        self.dataset_info_label.setText(
            f"总图像: {stats['total_images']}\n"
            f"已标注: {stats['labeled_images']}\n"
            f"标注框: {stats['total_annotations']}"
        )
    
    def split_dataset(self):
        if not self.project_manager.current_project:
            QMessageBox.warning(self, "提示", "请先打开一个项目")
            return
        
        try:
            splitter = DatasetSplitter(self.project_manager.current_project)
            train_count, val_count = splitter.split_dataset()
            
            self.dataset_info_label.setText(
                f"训练集: {train_count} 张\n"
                f"验证集: {val_count} 张\n"
                f"总计: {train_count + val_count} 张"
            )
            
            self.log_message(f"✅ 数据集划分完成: 训练集 {train_count}, 验证集 {val_count}")
            QMessageBox.information(self, "划分完成", f"训练集: {train_count}张\n验证集: {val_count}张")
            
        except Exception as e:
            QMessageBox.critical(self, "划分失败", str(e))
    
    def start_training(self):
        if not device_manager.is_gpu_available():
            QMessageBox.critical(self, "错误", f"GPU不可用，无法进行训练:\n{device_manager.get_gpu_error_message()}")
            return
            
        if not self.project_manager.current_project:
            QMessageBox.warning(self, "提示", "请先打开一个项目")
            return
        
        stats = self.project_manager.get_annotation_stats()
        if stats['labeled_images'] < 10:
            QMessageBox.warning(self, "提示", f"已标注图像不足 ({stats['labeled_images']} < 10)")
            return
        
        dataset_yaml = self.project_manager.current_project.base_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            reply = QMessageBox.question(
                self, "数据集未划分",
                "数据集尚未划分，是否现在划分？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.split_dataset()
            else:
                return
        
        model_type = self.model_type_combo.currentText()
        model_size = self.model_size_combo.currentText()
        
        self.project_manager.update_model_config(model_type, model_size)
        
        self.project_manager.current_project.training_config = TrainingConfig(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            image_size=self.imgsz_spin.value(),
            learning_rate=self.lr_spin.value(),
            patience=self.patience_spin.value()
        )
        
        self.trainer = ModelTrainer(self.project_manager.current_project)
        
        self.training_worker = TrainingWorker(self.trainer)
        self.training_worker.progress.connect(self.log_message)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.metrics_update.connect(self.on_metrics_update)
        
        self.btn_start.setVisible(False)
        self.btn_stop.setVisible(True)
        self.train_progress_bar.setRange(0, 0)
        
        self.log_message(f"🚀 开始训练: {model_type}{model_size}")
        self.training_started.emit()
        self.training_worker.start()
    
    def stop_training(self):
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认停止",
                "确定要停止训练吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.training_worker.stop()
                self.training_worker.terminate()
                self.log_message("⚠️ 训练已手动停止")
                self.on_training_finished(False, "训练已停止")
    
    def on_training_finished(self, success: bool, message: str):
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(False)
        self.train_progress_bar.setRange(0, 100)
        
        if success:
            self.log_message(f"✅ {message}")
            self.train_status_label.setText("训练完成!")
            self.train_status_label.setStyleSheet(f"color: {Styles.SUCCESS}; font-weight: bold;")
        else:
            self.log_message(f"❌ {message}")
            self.train_status_label.setText(f"训练失败: {message}")
            self.train_status_label.setStyleSheet(f"color: {Styles.ERROR};")
        
        self.training_finished.emit(success)
    
    def on_metrics_update(self, metrics: dict):
        self.result_table.setRowCount(len(metrics))
        
        for i, (key, value) in enumerate(metrics.items()):
            self.result_table.setItem(i, 0, QTableWidgetItem(key))
            if value is not None:
                self.result_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            else:
                self.result_table.setItem(i, 1, QTableWidgetItem("N/A"))
    
    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.train_log.append(f"[{timestamp}] {message}")
    
    def on_project_loaded(self):
        self.load_project_info()
        self.train_log.clear()
        self.result_table.setRowCount(0)
        self.train_status_label.setText("等待开始训练...")
        self.train_status_label.setStyleSheet("color: #616161;")


class ModelPanel(QWidget):
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.trainer = None
        self.onnx_worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        header = QLabel("模型管理")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #212121;")
        layout.addWidget(header)
        
        models_group = QGroupBox("已训练模型")
        models_group.setStyleSheet(Styles.GROUP_BOX)
        models_layout = QVBoxLayout(models_group)
        
        self.model_list = QListWidget()
        self.model_list.setStyleSheet(Styles.LIST_ITEM)
        self.model_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_list.customContextMenuRequested.connect(self.show_model_context_menu)
        models_layout.addWidget(self.model_list)
        
        btn_layout = QHBoxLayout()
        
        self.btn_refresh_models = QPushButton("刷新列表")
        self.btn_refresh_models.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_refresh_models.clicked.connect(self.refresh_model_list)
        btn_layout.addWidget(self.btn_refresh_models)
        
        self.btn_open_folder = QPushButton("打开文件夹")
        self.btn_open_folder.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_open_folder.clicked.connect(self.open_model_folder)
        btn_layout.addWidget(self.btn_open_folder)
        
        models_layout.addLayout(btn_layout)
        
        layout.addWidget(models_group)
        
        onnx_group = QGroupBox("ONNX转换")
        onnx_group.setStyleSheet(Styles.GROUP_BOX)
        onnx_layout = QVBoxLayout(onnx_group)
        
        onnx_info = QLabel("将PyTorch模型转换为ONNX格式，便于部署")
        onnx_info.setStyleSheet("color: #616161;")
        onnx_layout.addWidget(onnx_info)
        
        self.btn_convert_onnx = QPushButton("转换为ONNX")
        self.btn_convert_onnx.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_convert_onnx.clicked.connect(self.convert_to_onnx)
        onnx_layout.addWidget(self.btn_convert_onnx)
        
        self.onnx_progress = QProgressBar()
        self.onnx_progress.setStyleSheet(Styles.PROGRESS)
        self.onnx_progress.setVisible(False)
        onnx_layout.addWidget(self.onnx_progress)
        
        self.onnx_status = QLabel("")
        self.onnx_status.setStyleSheet("color: #616161;")
        onnx_layout.addWidget(self.onnx_status)
        
        layout.addWidget(onnx_group)
        
        report_group = QGroupBox("训练报告")
        report_group.setStyleSheet(Styles.GROUP_BOX)
        report_layout = QVBoxLayout(report_group)
        
        report_info = QLabel("查看训练报告和评估指标")
        report_info.setStyleSheet("color: #616161;")
        report_layout.addWidget(report_info)
        
        self.btn_generate_report = QPushButton("生成报告")
        self.btn_generate_report.setStyleSheet(Styles.BUTTON_SUCCESS)
        self.btn_generate_report.clicked.connect(self.generate_report)
        report_layout.addWidget(self.btn_generate_report)
        
        self.btn_open_report = QPushButton("查看报告")
        self.btn_open_report.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_open_report.clicked.connect(self.open_report)
        report_layout.addWidget(self.btn_open_report)
        
        layout.addWidget(report_group)
        
        layout.addStretch()
    
    def set_trainer(self, trainer):
        self.trainer = trainer
        self.refresh_model_list()
    
    def refresh_model_list(self):
        self.model_list.clear()
        
        if not self.project_manager.current_project:
            return
        
        models_dir = self.project_manager.current_project.base_dir / "models"
        
        pt_files = list(models_dir.glob("*.pt"))
        for pt_file in pt_files:
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            item = QListWidgetItem(f"📦 {pt_file.name} ({size_mb:.1f} MB)")
            item.setData(Qt.UserRole, str(pt_file))
            self.model_list.addItem(item)
        
        onnx_dir = models_dir / "onnx"
        if onnx_dir.exists():
            onnx_files = list(onnx_dir.glob("*.onnx"))
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                item = QListWidgetItem(f"⚡ {onnx_file.name} ({size_mb:.1f} MB)")
                item.setData(Qt.UserRole, str(onnx_file))
                self.model_list.addItem(item)
    
    def show_model_context_menu(self, position):
        item = self.model_list.itemAt(position)
        if not item:
            return
        
        model_path = item.data(Qt.UserRole)
        if not model_path or not Path(model_path).exists():
            return
        
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Styles.BACKGROUND};
                border: 1px solid #E0E0E0;
                padding: 5px;
            }}
            QMenu::item {{
                padding: 8px 25px;
                border-radius: 3px;
            }}
            QMenu::item:selected {{
                background-color: #E3F2FD;
            }}
        """)
        
        open_folder_action = menu.addAction("📁 打开文件位置")
        open_folder_action.triggered.connect(lambda: self.open_model_file_location(model_path))
        
        menu.addSeparator()
        
        delete_action = menu.addAction("🗑️ 删除模型")
        delete_action.triggered.connect(lambda: self.delete_model_file(model_path))
        
        menu.exec_(self.model_list.mapToGlobal(position))
    
    def open_model_file_location(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            QMessageBox.warning(self, "提示", "文件不存在")
            return
        
        import subprocess
        subprocess.run(['explorer', '/select,', str(model_path)])
    
    def delete_model_file(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            QMessageBox.warning(self, "提示", "文件不存在")
            return
        
        file_name = model_path.name
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除模型文件吗？\n\n{file_name}\n\n此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                model_path.unlink()
                
                info_path = model_path.parent / f"{model_path.stem}_conversion_info.json"
                if info_path.exists():
                    info_path.unlink()
                
                self.refresh_model_list()
                QMessageBox.information(self, "成功", f"已删除模型: {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")
    
    def open_model_folder(self):
        if not self.project_manager.current_project:
            return
        
        models_dir = self.project_manager.current_project.base_dir / "models"
        os.startfile(str(models_dir))
    
    def convert_to_onnx(self):
        if self.onnx_worker is not None and self.onnx_worker.isRunning():
            QMessageBox.warning(self, "提示", "正在进行ONNX转换，请等待完成")
            return
        
        selected_items = self.model_list.selectedItems()
        selected_model_path = None
        
        if selected_items:
            selected_model_path = selected_items[0].data(Qt.UserRole)
            if selected_model_path and not selected_model_path.endswith('.pt'):
                selected_model_path = None
        
        model_path = None
        
        if selected_model_path:
            model_path = selected_model_path
        elif self.trainer and self.trainer.best_model_path:
            model_path = str(self.trainer.best_model_path)
        else:
            if not self.project_manager.current_project:
                QMessageBox.warning(self, "提示", "请先打开一个项目")
                return
            
            models_dir = self.project_manager.current_project.base_dir / "models"
            pt_files = list(models_dir.glob("*.pt"))
            
            if not pt_files:
                QMessageBox.warning(self, "提示", "没有可转换的模型，请先完成训练或选择一个.pt模型")
                return
            
            model_names = [f.name for f in pt_files]
            from PyQt5.QtWidgets import QInputDialog
            choice, ok = QInputDialog.getItem(
                self, "选择模型", "请选择要转换的模型:", model_names, 0, False
            )
            
            if ok and choice:
                model_path = str(models_dir / choice)
            else:
                return
        
        if not model_path:
            QMessageBox.warning(self, "提示", "没有可转换的模型，请先选择一个.pt模型")
            return
        
        if not Path(model_path).exists():
            QMessageBox.warning(self, "提示", f"模型文件不存在: {model_path}")
            return
        
        self.btn_convert_onnx.setEnabled(False)
        self.onnx_progress.setVisible(True)
        self.onnx_progress.setRange(0, 0)
        self.onnx_status.setText("正在转换...")
        
        converter = ONNXConverter(self.project_manager.current_project)
        
        self.onnx_worker = ONNXConversionWorker(converter, model_path)
        self.onnx_worker.finished.connect(self.on_onnx_finished)
        self.onnx_worker.start()
    
    def on_onnx_finished(self, success: bool, message: str, onnx_info: dict):
        self.btn_convert_onnx.setEnabled(True)
        self.onnx_progress.setVisible(False)
        
        if success:
            self.onnx_status.setText(f"✅ 转换完成: {Path(message).name}")
            self.onnx_status.setStyleSheet(f"color: {Styles.SUCCESS};")
            self.refresh_model_list()
        else:
            self.onnx_status.setText(f"❌ 转换失败: {message}")
            self.onnx_status.setStyleSheet(f"color: {Styles.ERROR};")
        
        if self.onnx_worker is not None:
            self.onnx_worker.deleteLater()
            self.onnx_worker = None
    
    def generate_report(self):
        if not self.trainer:
            QMessageBox.warning(self, "提示", "没有训练结果，无法生成报告")
            return
        
        try:
            splitter = DatasetSplitter(self.project_manager.current_project)
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
            
            report_gen = ReportGenerator(self.project_manager.current_project)
            report_path = report_gen.generate_full_report(
                training_results, validation_metrics, dataset_stats
            )
            
            QMessageBox.information(self, "报告生成", f"报告已保存至:\n{report_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))
    
    def open_report(self):
        if not self.project_manager.current_project:
            return
        
        reports_dir = self.project_manager.current_project.base_dir / "reports"
        
        if not reports_dir.exists():
            QMessageBox.warning(self, "提示", "报告目录不存在")
            return
        
        report_files = list(reports_dir.glob("*.md"))
        if not report_files:
            QMessageBox.warning(self, "提示", "没有找到报告文件")
            return
        
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        os.startfile(str(latest_report))
    
    def on_project_loaded(self):
        self.refresh_model_list()
        self.onnx_status.setText("")
    
    def on_training_finished(self, success: bool):
        if success:
            self.refresh_model_list()


class InferenceWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str, list)
    
    def __init__(self, inference_engine, image_paths):
        super().__init__()
        self.inference_engine = inference_engine
        self.image_paths = image_paths
        self._is_running = True
    
    def run(self):
        results = []
        total = len(self.image_paths)
        
        for i, image_path in enumerate(self.image_paths):
            if not self._is_running:
                break
            
            success, result = self.inference_engine.predict_image(image_path)
            if success:
                results.append(result)
            
            self.progress.emit(i + 1, total, image_path)
        
        self.finished.emit(True, f"处理完成 {len(results)} 张图像", results)
    
    def stop(self):
        self._is_running = False


class VideoInferenceWorker(QThread):
    progress = pyqtSignal(int, int)
    frame_ready = pyqtSignal(object)
    finished = pyqtSignal(bool, str, dict)
    
    def __init__(self, inference_engine, video_path, output_path):
        super().__init__()
        self.inference_engine = inference_engine
        self.video_path = video_path
        self.output_path = output_path
        self._is_running = True
    
    def run(self):
        def progress_callback(current, total):
            if not self._is_running:
                # 发送终止信号
                raise StopIteration("用户停止")
            self.progress.emit(current, total)
        
        def frame_callback(frame, current, total):
            if self._is_running:
                self.frame_ready.emit(frame)
        
        try:
            success, message, stats = self.inference_engine.predict_video(
                self.video_path, 
                self.output_path,
                progress_callback=progress_callback,
                frame_callback=frame_callback
            )
            if self._is_running:
                self.finished.emit(success, message, stats)
            else:
                self.finished.emit(False, "推理已停止", {})
        except StopIteration:
            self.finished.emit(False, "推理已停止", {})
        except Exception as e:
            self.finished.emit(False, f"错误: {str(e)}", {})
    
    def stop(self):
        self._is_running = False


class InferencePanel(QWidget):
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.inference_engine = None
        self.inference_worker = None
        self.video_worker = None
        self.current_results: List[InferenceResult] = []
        self.current_image_path: Optional[str] = None
        self.gpu_available = device_manager.is_gpu_available()
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        if not self.gpu_available:
            # GPU不可用时显示警告面板
            warning_panel = self.create_gpu_warning_panel()
            layout.addWidget(warning_panel)
        else:
            left_panel = self.create_left_panel()
            layout.addWidget(left_panel, 1)
            
            center_panel = self.create_center_panel()
            layout.addWidget(center_panel, 2)
            
            right_panel = self.create_right_panel()
            layout.addWidget(right_panel, 1)
    
    def create_gpu_warning_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; }}")
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignCenter)
        
        warning_icon = QLabel("⚠️")
        warning_icon.setStyleSheet("font-size: 72px;")
        warning_icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning_icon)
        
        warning_title = QLabel("GPU 不可用")
        warning_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #F44336;")
        warning_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning_title)
        
        error_msg = device_manager.get_gpu_error_message() or "未检测到NVIDIA GPU"
        warning_text = QLabel(f"推理测试功能需要NVIDIA GPU支持\n\n{error_msg}")
        warning_text.setStyleSheet("font-size: 14px; color: #757575;")
        warning_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning_text)
        
        layout.addStretch()
        
        return panel
    
    def create_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; border-right: 1px solid {Styles.BORDER}; }}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        header = QLabel("模型与数据")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #212121;")
        layout.addWidget(header)
        
        model_group = QGroupBox("模型选择")
        model_group.setStyleSheet(Styles.GROUP_BOX)
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(Styles.INPUT)
        self.model_combo.setToolTip(
            "【模型选择】\n"
            "选择要用于推理的模型\n\n"
            "模型类型:\n"
            "• PyTorch (.pt): 原始训练模型\n"
            "• ONNX (.onnx): 导出的部署模型\n\n"
            "建议:\n"
            "• 推理使用 .pt 模型\n"
            "• 部署使用 .onnx 模型"
        )
        model_layout.addWidget(self.model_combo)
        
        btn_layout = QHBoxLayout()
        
        self.btn_refresh_models = QPushButton("刷新")
        self.btn_refresh_models.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_refresh_models.clicked.connect(self.refresh_models)
        btn_layout.addWidget(self.btn_refresh_models)
        
        self.btn_load_model = QPushButton("加载")
        self.btn_load_model.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_load_model.clicked.connect(self.load_model)
        btn_layout.addWidget(self.btn_load_model)
        
        model_layout.addLayout(btn_layout)
        
        self.model_status = QLabel("未加载模型")
        self.model_status.setStyleSheet("color: #757575; font-size: 11px;")
        model_layout.addWidget(self.model_status)
        
        layout.addWidget(model_group)
        
        data_group = QGroupBox("测试数据")
        data_group.setStyleSheet(Styles.GROUP_BOX)
        data_layout = QVBoxLayout(data_group)
        
        self.btn_select_image = QPushButton("选择图像")
        self.btn_select_image.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_select_image.clicked.connect(self.select_image)
        self.btn_select_image.setToolTip(
            "【选择图像】\n"
            "选择单张或多张图像进行推理\n\n"
            "支持格式:\n"
            "• JPG / JPEG\n"
            "• PNG\n"
            "• BMP\n\n"
            "支持多选批量处理"
        )
        data_layout.addWidget(self.btn_select_image)
        
        self.btn_select_video = QPushButton("选择视频")
        self.btn_select_video.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_select_video.clicked.connect(self.select_video)
        self.btn_select_video.setToolTip(
            "【选择视频】\n"
            "选择视频文件进行推理\n\n"
            "支持格式:\n"
            "• MP4\n"
            "• AVI\n"
            "• MOV\n\n"
            "处理后会生成标注视频"
        )
        data_layout.addWidget(self.btn_select_video)
        
        self.btn_select_folder = QPushButton("选择文件夹")
        self.btn_select_folder.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_select_folder.setToolTip(
            "【选择文件夹】\n"
            "选择包含图像的文件夹\n\n"
            "功能:\n"
            "• 批量处理文件夹内所有图像\n"
            "• 支持递归搜索子文件夹\n"
            "• 自动识别图像格式"
        )
        data_layout.addWidget(self.btn_select_folder)
        
        layout.addWidget(data_group)
        
        config_group = QGroupBox("推理参数")
        config_group.setStyleSheet(Styles.GROUP_BOX)
        config_layout = QFormLayout(config_group)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setStyleSheet(Styles.INPUT)
        self.conf_spin.setToolTip(
            "【置信度阈值】\n"
            "检测结果的最小置信度\n\n"
            "取值范围: 0.01 - 1.0\n\n"
            "• 值越高: 检测越严格，误检少但可能漏检\n"
            "• 值越低: 检测越宽松，召回率高但误检多\n\n"
            "推荐值: 0.25 - 0.5"
        )
        config_layout.addRow("置信度阈值:", self.conf_spin)
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setStyleSheet(Styles.INPUT)
        self.iou_spin.setToolTip(
            "【IOU阈值】\n"
            "非极大值抑制(NMS)的IOU阈值\n\n"
            "取值范围: 0.01 - 1.0\n\n"
            "• 值越高: 保留更多重叠框\n"
            "• 值越低: 更激进地去除重叠框\n\n"
            "推荐值: 0.45 - 0.7"
        )
        config_layout.addRow("IOU阈值:", self.iou_spin)
        
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setStyleSheet(Styles.INPUT)
        self.imgsz_spin.setToolTip(
            "【图像尺寸】\n"
            "推理时图像缩放尺寸\n\n"
            "• 值越大: 检测小目标能力越强，速度越慢\n"
            "• 值越小: 速度越快，可能漏检小目标\n\n"
            "推荐值: 640 (标准)\n"
            "必须为32的倍数"
        )
        config_layout.addRow("图像尺寸:", self.imgsz_spin)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(['自动', 'GPU'])
        self.device_combo.setStyleSheet(Styles.INPUT)
        self.device_combo.setToolTip(
            "【推理设备】\n"
            "选择推理使用的计算设备\n\n"
            "• 自动: 自动选择最佳GPU\n"
            "• GPU: 使用显卡推理\n\n"
            "注意: 此应用程序仅支持GPU推理"
        )
        config_layout.addRow("推理设备:", self.device_combo)
        
        layout.addWidget(config_group)
        
        layout.addStretch()
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: #1a1a1a; }}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        toolbar = QFrame()
        toolbar.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; border-bottom: 1px solid {Styles.BORDER}; }}")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        self.btn_run_inference = QPushButton("开始推理")
        self.btn_run_inference.setStyleSheet(Styles.BUTTON_SUCCESS)
        self.btn_run_inference.clicked.connect(self.run_inference)
        self.btn_run_inference.setToolTip(
            "【开始推理】\n"
            "对选中的图像/视频进行推理\n\n"
            "前提条件:\n"
            "• 已加载模型\n"
            "• 已选择测试数据"
        )
        toolbar_layout.addWidget(self.btn_run_inference)
        
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setStyleSheet(Styles.BUTTON_DANGER)
        self.btn_stop.clicked.connect(self.stop_inference)
        self.btn_stop.setEnabled(False)
        toolbar_layout.addWidget(self.btn_stop)
        
        toolbar_layout.addStretch()
        
        self.image_info_label = QLabel("请选择图像或视频进行推理")
        self.image_info_label.setStyleSheet("color: #616161;")
        toolbar_layout.addWidget(self.image_info_label)
        
        layout.addWidget(toolbar)
        
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #1a1a1a; color: #616161;")
        self.display_label.setMinimumSize(400, 300)
        self.display_label.setText("推理结果将在此显示")
        layout.addWidget(self.display_label, 1)
        
        progress_frame = QFrame()
        progress_frame.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; }}")
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(10, 5, 10, 5)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(Styles.PROGRESS)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #616161; font-size: 11px;")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_frame)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {Styles.BACKGROUND}; border-left: 1px solid {Styles.BORDER}; }}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        header = QLabel("结果与统计")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #212121;")
        layout.addWidget(header)
        
        stats_group = QGroupBox("性能统计")
        stats_group.setStyleSheet(Styles.GROUP_BOX)
        stats_layout = QFormLayout(stats_group)
        
        self.stats_images = QLabel("0")
        self.stats_images.setStyleSheet("color: #1976D2; font-weight: bold;")
        stats_layout.addRow("处理图像:", self.stats_images)
        
        self.stats_time = QLabel("0 ms")
        self.stats_time.setStyleSheet("color: #1976D2; font-weight: bold;")
        stats_layout.addRow("平均耗时:", self.stats_time)
        
        self.stats_fps = QLabel("0 FPS")
        self.stats_fps.setStyleSheet("color: #1976D2; font-weight: bold;")
        stats_layout.addRow("平均FPS:", self.stats_fps)
        
        self.stats_detections = QLabel("0")
        self.stats_detections.setStyleSheet("color: #1976D2; font-weight: bold;")
        stats_layout.addRow("检测总数:", self.stats_detections)
        
        self.stats_gpu = QLabel("0 MB")
        self.stats_gpu.setStyleSheet("color: #1976D2; font-weight: bold;")
        stats_layout.addRow("GPU显存:", self.stats_gpu)
        
        layout.addWidget(stats_group)
        
        results_group = QGroupBox("检测结果")
        results_group.setStyleSheet(Styles.GROUP_BOX)
        results_layout = QVBoxLayout(results_group)
        
        self.results_list = QListWidget()
        self.results_list.setStyleSheet(Styles.LIST_ITEM)
        self.results_list.setToolTip(
            "【检测结果列表】\n"
            "显示当前图像的检测结果\n\n"
            "信息包含:\n"
            "• 类别名称\n"
            "• 置信度\n"
            "• 边界框坐标"
        )
        results_layout.addWidget(self.results_list)
        
        layout.addWidget(results_group)
        
        export_group = QGroupBox("导出结果")
        export_group.setStyleSheet(Styles.GROUP_BOX)
        export_layout = QVBoxLayout(export_group)
        
        self.btn_export_image = QPushButton("保存标注图像")
        self.btn_export_image.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_export_image.clicked.connect(self.export_annotated_image)
        self.btn_export_image.setToolTip(
            "【保存标注图像】\n"
            "保存带有检测标注的图像\n\n"
            "输出:\n"
            "• 包含边界框的图像\n"
            "• 类别标签和置信度"
        )
        export_layout.addWidget(self.btn_export_image)
        
        self.btn_export_json = QPushButton("导出JSON")
        self.btn_export_json.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_export_json.clicked.connect(self.export_json)
        self.btn_export_json.setToolTip(
            "【导出JSON】\n"
            "导出检测结果为JSON格式\n\n"
            "包含内容:\n"
            "• 所有图像的检测结果\n"
            "• 性能统计数据\n"
            "• 推理配置参数"
        )
        export_layout.addWidget(self.btn_export_json)
        
        self.btn_export_csv = QPushButton("导出CSV")
        self.btn_export_csv.setStyleSheet(Styles.BUTTON_OUTLINE)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_csv.setToolTip(
            "【导出CSV】\n"
            "导出检测结果为CSV表格\n\n"
            "适合:\n"
            "• 数据分析\n"
            "• Excel处理\n"
            "• 后续处理"
        )
        export_layout.addWidget(self.btn_export_csv)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return panel
    
    def on_project_loaded(self):
        self.refresh_models()
        self.clear_results()
    
    def refresh_models(self):
        self.model_combo.clear()
        
        if not self.project_manager.current_project:
            return
        
        project_dir = self.project_manager.current_project.base_dir
        models_dir = project_dir / "models"
        
        if not models_dir.exists():
            return
        
        pt_files = list(models_dir.glob("*.pt"))
        for pt_file in sorted(pt_files, key=lambda x: x.stat().st_mtime, reverse=True):
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            self.model_combo.addItem(f"📦 {pt_file.name} ({size_mb:.1f} MB)", str(pt_file))
        
        onnx_dir = models_dir / "onnx"
        if onnx_dir.exists():
            onnx_files = list(onnx_dir.glob("*.onnx"))
            for onnx_file in sorted(onnx_files, key=lambda x: x.stat().st_mtime, reverse=True):
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                self.model_combo.addItem(f"⚡ {onnx_file.name} ({size_mb:.1f} MB)", str(onnx_file))
    
    def load_model(self):
        model_path = self.model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "提示", "请先选择模型")
            return
        
        if not device_manager.is_gpu_available():
            QMessageBox.critical(self, "错误", f"GPU不可用: {device_manager.get_gpu_error_message()}")
            return
        
        if self.inference_engine is None:
            self.inference_engine = YOLOInference(self.project_manager.current_project)
        
        device_map = {'自动': 'auto', 'GPU': 'cuda:0'}
        device = device_map.get(self.device_combo.currentText(), 'auto')
        
        success, message = self.inference_engine.load_model(model_path, device)
        
        if success:
            self.model_status.setText(f"✅ {Path(model_path).name}")
            self.model_status.setStyleSheet(f"color: {Styles.SUCCESS}; font-size: 11px;")
            QMessageBox.information(self, "成功", message)
        else:
            self.model_status.setText("❌ 加载失败")
            self.model_status.setStyleSheet(f"color: {Styles.ERROR}; font-size: 11px;")
            QMessageBox.critical(self, "错误", message)
    
    def select_image(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图像文件", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        
        if file_paths:
            self.current_image_path = file_paths[0]
            if len(file_paths) == 1:
                self.image_info_label.setText(f"📷 {Path(file_paths[0]).name}")
                self.display_image(file_paths[0])
            else:
                self.image_info_label.setText(f"📷 已选择 {len(file_paths)} 张图像")
            self.current_results = []
    
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_info_label.setText(f"🎬 {Path(file_path).name}")
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        
        if folder_path:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(folder_path).glob(f"*{ext}"))
                image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            if image_files:
                self.current_image_path = str(folder_path)
                self.image_info_label.setText(f"📁 文件夹: {len(image_files)} 张图像")
                self.current_results = []
            else:
                QMessageBox.warning(self, "提示", "文件夹中没有找到图像文件")
    
    def display_image(self, image_path: str):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.display_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.display_label.setPixmap(scaled)
    
    def display_numpy_image(self, image):
        # 转换 BGR 到 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QPixmap.fromImage(
            QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        )
        scaled = q_image.scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.display_label.setPixmap(scaled)
    
    def run_inference(self):
        if not device_manager.is_gpu_available():
            QMessageBox.critical(self, "错误", f"GPU不可用: {device_manager.get_gpu_error_message()}")
            return
        
        if self.inference_engine is None or not self.inference_engine.is_loaded:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        
        config = InferenceConfig(
            conf_threshold=self.conf_spin.value(),
            iou_threshold=self.iou_spin.value(),
            image_size=self.imgsz_spin.value()
        )
        self.inference_engine.set_config(config)
        self.inference_engine.reset_stats()
        
        if self.current_image_path is None:
            QMessageBox.warning(self, "提示", "请先选择图像或视频")
            return
        
        path = Path(self.current_image_path)
        
        if path.is_file() and path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
            self.run_video_inference(str(path))
        else:
            self.run_image_inference()
    
    def run_image_inference(self):
        self.btn_run_inference.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        
        image_paths = []
        path = Path(self.current_image_path) if self.current_image_path else None
        
        if path and path.is_file():
            image_paths = [str(path)]
        elif path and path.is_dir():
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            for ext in image_extensions:
                image_paths.extend([str(p) for p in path.glob(f"*{ext}")])
                image_paths.extend([str(p) for p in path.glob(f"*{ext.upper()}")])
        
        if not image_paths:
            self.btn_run_inference.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.progress_bar.setVisible(False)
            QMessageBox.warning(self, "提示", "没有找到可处理的图像")
            return
        
        self.inference_worker = InferenceWorker(self.inference_engine, image_paths)
        self.inference_worker.progress.connect(self.on_inference_progress)
        self.inference_worker.finished.connect(self.on_inference_finished)
        self.inference_worker.start()
    
    def run_video_inference(self, video_path: str):
        self.btn_run_inference.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        
        output_dir = self.project_manager.current_project.base_dir / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"inference_{timestamp}.mp4")
        
        self.video_worker = VideoInferenceWorker(self.inference_engine, video_path, output_path)
        self.video_worker.progress.connect(self.on_video_progress)
        self.video_worker.frame_ready.connect(self.on_frame_ready)
        self.video_worker.finished.connect(self.on_video_finished)
        self.video_worker.start()
    
    def on_inference_progress(self, current: int, total: int, image_path: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"处理中: {Path(image_path).name} ({current}/{total})")
    
    def on_inference_finished(self, success: bool, message: str, results: list):
        self.btn_run_inference.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success and results:
            self.current_results = results
            self.update_stats()
            
            if len(results) == 1:
                self.display_result(results[0])
            
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "提示", message if message else "推理失败")
    
    def on_video_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"处理帧: {current}/{total}")
    
    def on_frame_ready(self, frame):
        self.display_numpy_image(frame)
    
    def on_video_finished(self, success: bool, message: str, stats: dict):
        self.btn_run_inference.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.stats_images.setText(str(stats.get('total_frames', 0)))
            self.stats_time.setText(f"{stats.get('avg_inference_time_ms', 0):.1f} ms")
            self.stats_fps.setText(f"{stats.get('avg_fps', 0):.1f} FPS")
            self.stats_detections.setText(str(stats.get('total_detections', 0)))
            
            QMessageBox.information(self, "完成", f"视频处理完成\n输出: {message}")
        else:
            QMessageBox.critical(self, "错误", message)
    
    def stop_inference(self):
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            # 不使用 wait()，避免主线程阻塞
        
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            # 不使用 wait()，避免主线程阻塞
        
        self.btn_run_inference.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("已停止")
    
    def display_result(self, result: InferenceResult):
        import cv2
        
        image = cv2.imread(result.image_path)
        if image is None:
            return
        
        annotated = self.inference_engine._draw_detections(image, result.detections)
        self.display_numpy_image(annotated)
        
        self.update_results_list(result)
    
    def update_results_list(self, result: InferenceResult):
        self.results_list.clear()
        
        for i, det in enumerate(result.detections):
            item_text = f"{i+1}. {det.class_name} ({det.confidence:.2f})"
            self.results_list.addItem(item_text)
    
    def update_stats(self):
        if self.inference_engine:
            stats = self.inference_engine.get_stats()
            self.stats_images.setText(str(stats.total_images))
            self.stats_time.setText(f"{stats.avg_inference_time * 1000:.1f} ms")
            self.stats_fps.setText(f"{stats.avg_fps:.1f} FPS")
            self.stats_detections.setText(str(stats.total_detections))
            self.stats_gpu.setText(f"{stats.gpu_memory_used:.1f} MB")
    
    def clear_results(self):
        self.current_results = []
        self.results_list.clear()
        self.stats_images.setText("0")
        self.stats_time.setText("0 ms")
        self.stats_fps.setText("0 FPS")
        self.stats_detections.setText("0")
        self.stats_gpu.setText("0 MB")
        self.display_label.clear()
        self.display_label.setText("推理结果将在此显示")
    
    def export_annotated_image(self):
        if not self.current_results:
            QMessageBox.warning(self, "提示", "没有推理结果可导出")
            return
        
        output_dir = self.project_manager.current_project.base_dir / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for result in self.current_results:
            if result.image_path:
                output_name = f"{Path(result.image_path).stem}_detected.jpg"
                output_path = str(output_dir / output_name)
                
                success, msg = self.inference_engine.save_annotated_image(
                    result.image_path, output_path
                )
                if success:
                    saved_count += 1
        
        QMessageBox.information(
            self, "导出完成", 
            f"已保存 {saved_count} 张标注图像到:\n{output_dir}"
        )
    
    def export_json(self):
        if not self.current_results:
            QMessageBox.warning(self, "提示", "没有推理结果可导出")
            return
        
        output_dir = self.project_manager.current_project.base_dir / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"inference_results_{timestamp}.json")
        
        if self.inference_engine.export_results_json(self.current_results, output_path):
            QMessageBox.information(self, "导出成功", f"JSON文件已保存到:\n{output_path}")
        else:
            QMessageBox.critical(self, "导出失败", "无法导出JSON文件")
    
    def export_csv(self):
        if not self.current_results:
            QMessageBox.warning(self, "提示", "没有推理结果可导出")
            return
        
        output_dir = self.project_manager.current_project.base_dir / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"inference_results_{timestamp}.csv")
        
        if self.inference_engine.export_results_csv(self.current_results, output_path):
            QMessageBox.information(self, "导出成功", f"CSV文件已保存到:\n{output_path}")
        else:
            QMessageBox.critical(self, "导出失败", "无法导出CSV文件")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_manager = ProjectManager()
        self.trainer = None
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        self.setWindowTitle("YOLO标注与训练一体化工具")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Styles.BACKGROUND};
            }}
            QToolTip {{
                background-color: #37474F;
                color: #ECEFF1;
                border: 1px solid #546E7A;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                line-height: 1.5;
            }}
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        header = self.create_header()
        main_layout.addWidget(header)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(Styles.TAB_WIDGET)
        
        self.project_panel = ProjectPanel(self.project_manager)
        self.tab_widget.addTab(self.project_panel, "📁 项目管理")
        
        self.annotation_panel = AnnotationPanel(self.project_manager)
        self.tab_widget.addTab(self.annotation_panel, "✏️ 数据标注")
        
        self.training_panel = TrainingPanel(self.project_manager)
        self.tab_widget.addTab(self.training_panel, "🎯 模型训练")
        
        self.model_panel = ModelPanel(self.project_manager)
        self.tab_widget.addTab(self.model_panel, "📦 模型管理")
        
        self.inference_panel = InferencePanel(self.project_manager)
        self.tab_widget.addTab(self.inference_panel, "🔍 推理测试")
        
        main_layout.addWidget(self.tab_widget, 1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setStyleSheet(Styles.PROGRESS)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_header(self) -> QWidget:
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Styles.PRIMARY}, stop:1 {Styles.PRIMARY_DARK});
            }}
            QLabel {{
                color: white;
            }}
        """)
        header.setMinimumHeight(60)
        header.setMaximumHeight(80)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(20)
        
        title = QLabel("YOLO标注与训练一体化工具")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        title.setMinimumHeight(30)
        layout.addWidget(title)
        
        layout.addStretch()
        
        version_label = QLabel("v1.0.0")
        version_label.setStyleSheet("font-size: 12px;")
        version_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        layout.addWidget(version_label)
        
        return header
    
    def setup_connections(self):
        self.project_panel.project_loaded.connect(self.on_project_loaded)
        self.training_panel.training_finished.connect(self.on_training_finished)
    
    def on_project_loaded(self):
        self.annotation_panel.load_project_info()
        self.training_panel.on_project_loaded()
        self.model_panel.on_project_loaded()
        self.inference_panel.on_project_loaded()
        
        self.status_label.setText(f"已加载项目: {self.project_manager.current_project.project_name}")
        self.tab_widget.setCurrentIndex(1)
    
    def on_training_finished(self, success: bool):
        if success and self.training_panel.trainer:
            self.model_panel.set_trainer(self.training_panel.trainer)
            self.trainer = self.training_panel.trainer


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    print("Starting application...")
    
    try:
        window = MainWindow()
        print("Window created successfully")
        window.show()
        print("Window shown, entering main loop...")
        ret = app.exec_()
        print(f"Application exited with code: {ret}")
        sys.exit(ret)
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
