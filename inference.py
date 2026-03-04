import os
import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from device_manager import device_manager

import torch


@dataclass
class InferenceConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    device: str = 'auto'
    image_size: int = 640
    half_precision: bool = False


@dataclass
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    bbox_xyxy: List[int]
    
    def to_dict(self) -> Dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': round(self.confidence, 4),
            'bbox': [round(x, 2) for x in self.bbox],
            'bbox_xyxy': self.bbox_xyxy
        }


@dataclass
class InferenceResult:
    image_path: str
    image_width: int
    image_height: int
    detections: List[DetectionResult]
    inference_time: float
    fps: float
    
    def to_dict(self) -> Dict:
        return {
            'image_path': self.image_path,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'detections': [d.to_dict() for d in self.detections],
            'inference_time_ms': round(self.inference_time * 1000, 2),
            'fps': round(self.fps, 2),
            'num_detections': len(self.detections)
        }


@dataclass
class PerformanceStats:
    total_images: int = 0
    total_time: float = 0.0
    avg_inference_time: float = 0.0
    avg_fps: float = 0.0
    total_detections: int = 0
    avg_detections_per_image: float = 0.0
    gpu_memory_used: float = 0.0
    cpu_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_images': self.total_images,
            'total_time_s': round(self.total_time, 3),
            'avg_inference_time_ms': round(self.avg_inference_time * 1000, 2),
            'avg_fps': round(self.avg_fps, 2),
            'total_detections': self.total_detections,
            'avg_detections_per_image': round(self.avg_detections_per_image, 2),
            'gpu_memory_used_mb': round(self.gpu_memory_used, 2),
            'cpu_percent': round(self.cpu_percent, 2)
        }


class ONNXInferenceEngine:
    def __init__(self, model_path: str, classes: List[str], config: InferenceConfig):
        self.model_path = model_path
        self.classes = classes
        self.config = config
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.using_gpu = False
        
    def load(self) -> Tuple[bool, str]:
        try:
            import onnxruntime as ort
        except ImportError as e:
            return False, f"onnxruntime未安装: {str(e)}"
        except Exception as e:
            return False, f"onnxruntime加载失败: {str(e)}"
        
        try:
            available_providers = ort.get_available_providers()
            print(f"[ONNX] 可用的执行提供者: {available_providers}")
        except Exception as e:
            print(f"[ONNX] 无法获取可用提供者: {e}")
            available_providers = []
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers_to_try = []
        
        if 'CUDAExecutionProvider' in available_providers:
            providers_to_try.append(['CUDAExecutionProvider', 'CPUExecutionProvider'])
        providers_to_try.append(['CPUExecutionProvider'])
        
        last_error = None
        for providers in providers_to_try:
            try:
                print(f"[ONNX] 尝试使用提供者: {providers}")
                
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options,
                    providers=providers
                )
                
                actual_providers = self.session.get_providers()
                print(f"[ONNX] 实际使用的提供者: {actual_providers}")
                
                if 'CUDAExecutionProvider' in actual_providers:
                    self.using_gpu = True
                    print("[ONNX] ✅ CUDA GPU推理已启用")
                else:
                    self.using_gpu = False
                    print("[ONNX] ⚠️ 使用CPU推理")
                
                input_info = self.session.get_inputs()[0]
                self.input_name = input_info.name
                self.input_shape = input_info.shape
                print(f"[ONNX] 输入名称: {self.input_name}, 形状: {self.input_shape}")
                
                self.output_names = [o.name for o in self.session.get_outputs()]
                print(f"[ONNX] 输出名称: {self.output_names}")
                
                device_name = "GPU (CUDA)" if self.using_gpu else "CPU"
                return True, f"ONNX模型加载成功 (设备: {device_name})"
                
            except Exception as e:
                last_error = e
                print(f"[ONNX] 提供者 {providers} 失败: {e}")
                if self.session is not None:
                    del self.session
                    self.session = None
                continue
        
        import traceback
        traceback.print_exc()
        return False, f"ONNX模型加载失败: {str(last_error)}"
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        target_size = self.config.image_size
        h, w = img.shape[:2]
        
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                         cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        img_input = img_padded.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input, scale, (left, top)
    
    def postprocess(self, outputs: List[np.ndarray], original_shape: Tuple[int, int], 
                    scale: float, pad: Tuple[int, int]) -> List[DetectionResult]:
        detections = []
        
        output = outputs[0]
        
        if output.ndim == 3:
            output = output[0]
        
        output = output.transpose(1, 0)
        
        boxes = output[:, :4]
        scores = output[:, 4:]
        
        if scores.shape[1] > 1:
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
        else:
            class_ids = np.zeros(len(scores), dtype=np.int32)
            confidences = scores[:, 0]
        
        mask = confidences >= self.config.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        if len(boxes) > 0:
            boxes_xyxy = np.copy(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            
            indices = self._nms(boxes_xyxy, confidences, self.config.iou_threshold)
            
            left, top = pad
            orig_h, orig_w = original_shape[:2]
            
            for i in indices:
                x1, y1, x2, y2 = boxes_xyxy[i]
                
                x1 = (x1 - left) / scale
                y1 = (y1 - top) / scale
                x2 = (x2 - left) / scale
                y2 = (y2 - top) / scale
                
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                cls_id = int(class_ids[i])
                conf = float(confidences[i])
                
                if cls_id < len(self.classes):
                    class_name = self.classes[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                x_center = (x1 + x2) / 2 / orig_w
                y_center = (y1 + y2) / 2 / orig_h
                w = (x2 - x1) / orig_w
                h = (y2 - y1) / orig_h
                
                detection = DetectionResult(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=[x_center, y_center, w, h],
                    bbox_xyxy=[int(x1), int(y1), int(x2), int(y2)]
                )
                detections.append(detection)
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def predict(self, image: np.ndarray) -> Tuple[bool, List[DetectionResult], float]:
        try:
            original_shape = image.shape
            img_input, scale, pad = self.preprocess(image)
            
            start_time = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: img_input})
            inference_time = time.time() - start_time
            
            detections = self.postprocess(outputs, original_shape, scale, pad)
            
            return True, detections, inference_time
            
        except Exception as e:
            print(f"[ONNX推理错误] {e}")
            import traceback
            traceback.print_exc()
            return False, [], 0
    
    def cleanup(self):
        if self.session is not None:
            del self.session
            self.session = None


class YOLOInference:
    def __init__(self, project_config=None):
        self.project_config = project_config
        self.model = None
        self.onnx_engine = None
        self.model_path: Optional[str] = None
        self.classes: List[str] = []
        self.config = InferenceConfig()
        self.performance_stats = PerformanceStats()
        self.is_onnx = False
        
        if not device_manager.is_gpu_available():
            raise RuntimeError(f"GPU不可用，无法初始化推理引擎: {device_manager.get_gpu_error_message()}")
        
        if project_config:
            self.classes = project_config.classes
    
    @property
    def is_loaded(self) -> bool:
        if self.is_onnx:
            return self.onnx_engine is not None and self.onnx_engine.session is not None
        else:
            return self.model is not None
    
    def load_model(self, model_path: str, device: str = 'auto') -> Tuple[bool, str]:
        if not device_manager.is_gpu_available():
            return False, f"GPU不可用: {device_manager.get_gpu_error_message()}"
        
        try:
            self.model_path = model_path
            self.is_onnx = model_path.endswith('.onnx')
            
            if self.is_onnx:
                return self._load_onnx_model(model_path)
            else:
                return self._load_pytorch_model(model_path, device)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"模型加载失败: {str(e)}"
    
    def _load_pytorch_model(self, model_path: str, device: str = 'auto') -> Tuple[bool, str]:
        try:
            device = self._resolve_device(device)
            self.config.device = device
            print(f"[推理] 使用设备: {device}")
            
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(device)
            
            if hasattr(self.model, 'names') and self.model.names:
                model_classes = self.model.names
                if not self.classes:
                    self.classes = [model_classes[i] for i in sorted(model_classes.keys())]
                else:
                    self._validate_classes(model_classes)
            
            return True, f"PyTorch模型加载成功: {Path(model_path).name} (设备: {device})"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"PyTorch模型加载失败: {str(e)}"
    
    def _load_onnx_model(self, model_path: str) -> Tuple[bool, str]:
        try:
            if not self.classes:
                info_path = Path(model_path).parent / f"{Path(model_path).stem}_conversion_info.json"
                if info_path.exists():
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                        self.classes = info.get('classes', [])
                        print(f"[ONNX] 从转换信息加载类别: {self.classes}")
            
            self.onnx_engine = ONNXInferenceEngine(model_path, self.classes, self.config)
            success, msg = self.onnx_engine.load()
            
            if success:
                self.config.device = 'CUDA'
                return True, f"ONNX模型加载成功: {Path(model_path).name}"
            else:
                return False, msg
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"ONNX模型加载失败: {str(e)}"
    
    def _validate_classes(self, model_classes: Dict[int, str]):
        print(f"[推理] 项目类别: {self.classes}")
        print(f"[推理] 模型类别: {model_classes}")
        
        model_class_list = [model_classes[i] for i in sorted(model_classes.keys())]
        
        if set(self.classes) != set(model_class_list):
            print(f"[警告] 项目类别与模型类别不匹配!")
            print(f"[警告] 项目: {self.classes}")
            print(f"[警告] 模型: {model_class_list}")
            
            self.classes = model_class_list
            print(f"[推理] 已自动使用模型类别: {self.classes}")
    
    def set_config(self, config: InferenceConfig):
        self.config = config
    
    def set_classes(self, classes: List[str]):
        self.classes = classes
    
    def predict_image(self, image_path: str) -> Tuple[bool, InferenceResult]:
        if not device_manager.is_gpu_available():
            print(f"[错误] GPU不可用: {device_manager.get_gpu_error_message()}")
            return False, None
        
        if self.is_onnx:
            return self._predict_image_onnx(image_path)
        else:
            return self._predict_image_pytorch(image_path)
    
    def _predict_image_pytorch(self, image_path: str) -> Tuple[bool, InferenceResult]:
        if self.model is None:
            print("[错误] PyTorch模型未加载")
            return False, None
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[错误] 无法读取图像: {image_path}")
                return False, None
            
            height, width = image.shape[:2]
            
            start_time = time.time()
            
            device = self._resolve_device(self.config.device)
            results = self.model.predict(
                source=image,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                imgsz=self.config.image_size,
                device=device,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            detections = self._parse_results(results, width, height)
            
            print(f"[PyTorch推理] 图像: {Path(image_path).name}")
            print(f"[PyTorch推理] 尺寸: {width}x{height}")
            print(f"[PyTorch推理] 耗时: {inference_time*1000:.1f}ms, FPS: {fps:.1f}")
            print(f"[PyTorch推理] 检测到 {len(detections)} 个目标")
            for det in detections:
                print(f"  - {det.class_name}: {det.confidence:.2f}")
            
            result = InferenceResult(
                image_path=image_path,
                image_width=width,
                image_height=height,
                detections=detections,
                inference_time=inference_time,
                fps=fps
            )
            
            self._update_stats(result)
            
            return True, result
            
        except Exception as e:
            print(f"[PyTorch推理错误] {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def _predict_image_onnx(self, image_path: str) -> Tuple[bool, InferenceResult]:
        if self.onnx_engine is None:
            print("[错误] ONNX模型未加载")
            return False, None
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[错误] 无法读取图像: {image_path}")
                return False, None
            
            height, width = image.shape[:2]
            
            success, detections, inference_time = self.onnx_engine.predict(image)
            
            if not success:
                return False, None
            
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            print(f"[ONNX推理] 图像: {Path(image_path).name}")
            print(f"[ONNX推理] 尺寸: {width}x{height}")
            print(f"[ONNX推理] 耗时: {inference_time*1000:.1f}ms, FPS: {fps:.1f}")
            print(f"[ONNX推理] 检测到 {len(detections)} 个目标")
            for det in detections:
                print(f"  - {det.class_name}: {det.confidence:.2f}")
            
            result = InferenceResult(
                image_path=image_path,
                image_width=width,
                image_height=height,
                detections=detections,
                inference_time=inference_time,
                fps=fps
            )
            
            self._update_stats(result)
            
            return True, result
            
        except Exception as e:
            print(f"[ONNX推理错误] {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def predict_batch(self, image_paths: List[str], progress_callback=None) -> List[InferenceResult]:
        # 检查GPU是否可用
        if not device_manager.is_gpu_available():
            raise RuntimeError(f"GPU不可用: {device_manager.get_gpu_error_message()}")
        
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            success, result = self.predict_image(image_path)
            if success:
                results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, image_path)
        
        return results
    
    def predict_video(self, video_path: str, output_path: str = None, 
                      progress_callback=None, frame_callback=None) -> Tuple[bool, str, Dict]:
        if not device_manager.is_gpu_available():
            return False, f"GPU不可用: {device_manager.get_gpu_error_message()}", {}
        
        if self.is_onnx:
            return self._predict_video_onnx(video_path, output_path, progress_callback, frame_callback)
        else:
            return self._predict_video_pytorch(video_path, output_path, progress_callback, frame_callback)
    
    def _predict_video_pytorch(self, video_path: str, output_path: str = None, 
                               progress_callback=None, frame_callback=None) -> Tuple[bool, str, Dict]:
        if self.model is None:
            return False, "PyTorch模型未加载", {}
        
        cap = None
        writer = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "无法打开视频文件", {}
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            all_detections = []
            frame_count = 0
            total_inference_time = 0
            
            device = self._resolve_device(self.config.device)
            print(f"[PyTorch视频推理] 使用设备: {device}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                results = self.model.predict(
                    source=frame,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_detections,
                    imgsz=self.config.image_size,
                    device=device,
                    verbose=False
                )
                
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                detections = self._parse_results(results, width, height)
                all_detections.extend([d.to_dict() for d in detections])
                
                annotated_frame = self._draw_detections(frame, detections)
                
                if writer:
                    writer.write(annotated_frame)
                
                if frame_callback:
                    frame_callback(annotated_frame, frame_count, total_frames)
                
                if progress_callback:
                    progress_callback(frame_count + 1, total_frames)
                
                frame_count += 1
            
            cap.release()
            if writer:
                writer.release()
            
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
            
            stats = {
                'total_frames': frame_count,
                'total_detections': len(all_detections),
                'avg_fps': round(avg_fps, 2),
                'avg_inference_time_ms': round(avg_inference_time * 1000, 2),
                'video_fps': fps,
                'resolution': f"{width}x{height}"
            }
            
            return True, output_path or "处理完成", stats
            
        except StopIteration as e:
            if cap:
                cap.release()
            if writer:
                writer.release()
            return False, str(e), {}
        except Exception as e:
            if cap:
                cap.release()
            if writer:
                writer.release()
            return False, f"视频处理错误: {str(e)}", {}
    
    def _predict_video_onnx(self, video_path: str, output_path: str = None,
                            progress_callback=None, frame_callback=None) -> Tuple[bool, str, Dict]:
        if self.onnx_engine is None:
            return False, "ONNX模型未加载", {}
        
        cap = None
        writer = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "无法打开视频文件", {}
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            all_detections = []
            frame_count = 0
            total_inference_time = 0
            
            print(f"[ONNX视频推理] 开始处理")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                success, detections, inference_time = self.onnx_engine.predict(frame)
                
                if success:
                    total_inference_time += inference_time
                    all_detections.extend([d.to_dict() for d in detections])
                    
                    annotated_frame = self._draw_detections(frame, detections)
                else:
                    annotated_frame = frame
                
                if writer:
                    writer.write(annotated_frame)
                
                if frame_callback:
                    frame_callback(annotated_frame, frame_count, total_frames)
                
                if progress_callback:
                    progress_callback(frame_count + 1, total_frames)
                
                frame_count += 1
            
            cap.release()
            if writer:
                writer.release()
            
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
            
            stats = {
                'total_frames': frame_count,
                'total_detections': len(all_detections),
                'avg_fps': round(avg_fps, 2),
                'avg_inference_time_ms': round(avg_inference_time * 1000, 2),
                'video_fps': fps,
                'resolution': f"{width}x{height}"
            }
            
            return True, output_path or "处理完成", stats
            
        except Exception as e:
            if cap:
                cap.release()
            if writer:
                writer.release()
            return False, f"ONNX视频处理错误: {str(e)}", {}
    
    def predict_numpy(self, image: np.ndarray) -> Tuple[bool, InferenceResult]:
        if not device_manager.is_gpu_available():
            return False, None
        
        if self.is_onnx:
            return self._predict_numpy_onnx(image)
        else:
            return self._predict_numpy_pytorch(image)
    
    def _predict_numpy_pytorch(self, image: np.ndarray) -> Tuple[bool, InferenceResult]:
        if self.model is None:
            return False, None
        
        try:
            height, width = image.shape[:2]
            
            start_time = time.time()
            
            device = self._resolve_device(self.config.device)
            results = self.model.predict(
                source=image,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                imgsz=self.config.image_size,
                device=device,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            detections = self._parse_results(results, width, height)
            
            result = InferenceResult(
                image_path="",
                image_width=width,
                image_height=height,
                detections=detections,
                inference_time=inference_time,
                fps=fps
            )
            
            return True, result
            
        except Exception as e:
            print(f"PyTorch推理错误: {e}")
            return False, None
    
    def _predict_numpy_onnx(self, image: np.ndarray) -> Tuple[bool, InferenceResult]:
        if self.onnx_engine is None:
            return False, None
        
        try:
            height, width = image.shape[:2]
            
            success, detections, inference_time = self.onnx_engine.predict(image)
            
            if not success:
                return False, None
            
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            result = InferenceResult(
                image_path="",
                image_width=width,
                image_height=height,
                detections=detections,
                inference_time=inference_time,
                fps=fps
            )
            
            return True, result
            
        except Exception as e:
            print(f"ONNX推理错误: {e}")
            return False, None
    
    def _parse_results(self, results, img_width: int, img_height: int) -> List[DetectionResult]:
        detections = []
        
        if not results or len(results) == 0:
            print("[推理] 没有推理结果")
            return detections
        
        result = results[0]
        
        if result.boxes is None:
            print("[推理] 没有检测框")
            return detections
        
        boxes = result.boxes
        print(f"[推理] 原始检测框数量: {len(boxes)}")
        print(f"[推理] 当前类别列表: {self.classes}")
        
        if hasattr(result, 'names') and result.names:
            print(f"[推理] 结果中的类别映射: {result.names}")
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            w = (x2 - x1) / img_width
            h = (y2 - y1) / img_height
            
            if cls_id < len(self.classes):
                class_name = self.classes[cls_id]
            else:
                if hasattr(result, 'names') and result.names and cls_id in result.names:
                    class_name = result.names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                print(f"[警告] 类别ID {cls_id} 超出范围(类别数:{len(self.classes)})，使用默认名称: {class_name}")
            
            detection = DetectionResult(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox=[x_center, y_center, w, h],
                bbox_xyxy=[int(x1), int(y1), int(x2), int(y2)]
            )
            detections.append(detection)
        
        return detections
    
    def _draw_detections(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        在图像上绘制检测结果，支持中文显示
        """
        # 转换为PIL图像以支持中文
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        colors = [
            (255, 82, 82), (76, 175, 80), (33, 150, 243), (255, 193, 7),
            (156, 39, 176), (0, 188, 212), (255, 87, 34), (139, 195, 74),
            (63, 81, 181), (121, 85, 72), (233, 30, 99), (0, 150, 136)
        ]
        
        # 尝试加载中文字体
        try:
            # Windows系统常用中文字体路径
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                "C:/Windows/Fonts/simhei.ttf",    # 黑体
                "C:/Windows/Fonts/simsun.ttc",    # 宋体
                "C:/Windows/Fonts/STZHONGS.TTF",  # 华文中宋
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, 16)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            color = colors[det.class_id % len(colors)]
            x1, y1, x2, y2 = det.bbox_xyxy
            
            # 绘制边界框 (PIL使用RGB颜色)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # 准备标签文本
            label = f"{det.class_name} {det.confidence:.2f}"
            
            # 获取文本大小
            try:
                bbox = draw.textbbox((x1, y1), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width, text_height = draw.textsize(label, font=font)
            
            # 绘制标签背景
            draw.rectangle(
                [x1, y1 - text_height - 8, x1 + text_width + 4, y1],
                fill=color
            )
            
            # 绘制文本 (白色)
            draw.text((x1 + 2, y1 - text_height - 6), label, font=font, fill=(255, 255, 255))
        
        # 转换回OpenCV格式
        annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return annotated
    
    def _update_stats(self, result: InferenceResult):
        self.performance_stats.total_images += 1
        self.performance_stats.total_time += result.inference_time
        self.performance_stats.total_detections += len(result.detections)
        
        n = self.performance_stats.total_images
        self.performance_stats.avg_inference_time = self.performance_stats.total_time / n
        self.performance_stats.avg_fps = n / self.performance_stats.total_time if self.performance_stats.total_time > 0 else 0
        self.performance_stats.avg_detections_per_image = self.performance_stats.total_detections / n
        
        if torch.cuda.is_available():
            self.performance_stats.gpu_memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    def reset_stats(self):
        self.performance_stats = PerformanceStats()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_stats(self) -> PerformanceStats:
        return self.performance_stats
    
    def export_results_json(self, results: List[InferenceResult], output_path: str) -> bool:
        try:
            data = {
                'export_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_path': self.model_path,
                'config': {
                    'conf_threshold': self.config.conf_threshold,
                    'iou_threshold': self.config.iou_threshold,
                    'image_size': self.config.image_size,
                    'device': self.config.device
                },
                'performance_stats': self.performance_stats.to_dict(),
                'results': [r.to_dict() for r in results]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"导出JSON失败: {e}")
            return False
    
    def export_results_csv(self, results: List[InferenceResult], output_path: str) -> bool:
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'image_path', 'image_width', 'image_height',
                    'class_id', 'class_name', 'confidence',
                    'bbox_x_center', 'bbox_y_center', 'bbox_width', 'bbox_height',
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                    'inference_time_ms', 'fps'
                ])
                
                for result in results:
                    for det in result.detections:
                        writer.writerow([
                            result.image_path,
                            result.image_width,
                            result.image_height,
                            det.class_id,
                            det.class_name,
                            det.confidence,
                            det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                            det.bbox_xyxy[0], det.bbox_xyxy[1], det.bbox_xyxy[2], det.bbox_xyxy[3],
                            round(result.inference_time * 1000, 2),
                            round(result.fps, 2)
                        ])
            
            return True
            
        except Exception as e:
            print(f"导出CSV失败: {e}")
            return False
    
    def save_annotated_image(self, image_path: str, output_path: str) -> Tuple[bool, str]:
        try:
            success, result = self.predict_image(image_path)
            if not success:
                return False, "推理失败"
            
            image = cv2.imread(image_path)
            annotated = self._draw_detections(image, result.detections)
            
            cv2.imwrite(output_path, annotated)
            
            return True, output_path
            
        except Exception as e:
            return False, f"保存失败: {str(e)}"
    
    def _resolve_device(self, device: str) -> str:
        """解析设备 - 强制GPU"""
        return device_manager.resolve_device(device)
    
    def get_available_models(self, project_dir: Path) -> List[Dict]:
        models = []
        models_dir = project_dir / "models"
        
        if not models_dir.exists():
            return models
        
        pt_files = list(models_dir.glob("*.pt"))
        for pt_file in pt_files:
            stat = pt_file.stat()
            models.append({
                'name': pt_file.name,
                'path': str(pt_file),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                'type': 'PyTorch'
            })
        
        onnx_dir = models_dir / "onnx"
        if onnx_dir.exists():
            onnx_files = list(onnx_dir.glob("*.onnx"))
            for onnx_file in onnx_files:
                stat = onnx_file.stat()
                models.append({
                    'name': onnx_file.name,
                    'path': str(onnx_file),
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    'type': 'ONNX'
                })
        
        return sorted(models, key=lambda x: x['modified'], reverse=True)
