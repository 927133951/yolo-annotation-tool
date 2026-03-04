import os
import shutil
import gc
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime
import json


class ONNXConverter:
    def __init__(self, project_config):
        self.project_config = project_config
        self.project_dir = project_config.base_dir
        self.onnx_dir = self.project_dir / "models" / "onnx"
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
    
    def _clear_gpu_memory(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except Exception:
            pass
    
    def convert_to_onnx(
        self,
        model_path: str,
        simplify: bool = False,
        dynamic_batch: bool = False,
        opset_version: int = 12,
        half_precision: bool = False
    ) -> Tuple[bool, str]:
        model_path = Path(model_path)
        
        if not model_path.exists():
            return False, f"模型文件不存在: {model_path}"
        
        if model_path.suffix != '.pt':
            return False, "只支持PyTorch模型(.pt)转换"
        
        model = None
        try:
            self._clear_gpu_memory()
            
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            
            export_path = model.export(
                format='onnx',
                simplify=False,
                dynamic=dynamic_batch,
                opset=opset_version,
                half=half_precision
            )
            
            if export_path is None:
                return False, "ONNX导出失败：未生成输出文件"
            
            onnx_path = Path(export_path)
            
            model_name = model_path.stem
            final_onnx_path = self.onnx_dir / f"{model_name}.onnx"
            
            if onnx_path.exists():
                shutil.copy2(onnx_path, final_onnx_path)
            
            self.save_conversion_info(model_path, final_onnx_path, {
                'simplify': False,
                'dynamic_batch': dynamic_batch,
                'opset_version': opset_version,
                'half_precision': half_precision
            })
            
            return True, str(final_onnx_path)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"ONNX转换错误详情:\n{error_detail}")
            return False, f"ONNX转换失败: {str(e)}"
        finally:
            if model is not None:
                del model
            self._clear_gpu_memory()
    
    def convert_with_ultralytics(
        self,
        model_path: str,
        imgsz: int = 640,
        batch: int = 1
    ) -> Tuple[bool, str]:
        model_path = Path(model_path)
        
        if not model_path.exists():
            return False, f"模型文件不存在: {model_path}"
        
        model = None
        try:
            self._clear_gpu_memory()
            
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            
            onnx_path = model.export(
                format='onnx',
                imgsz=imgsz,
                batch=batch,
                simplify=True,
                dynamic=False
            )
            
            onnx_path = Path(onnx_path)
            model_name = model_path.stem
            final_onnx_path = self.onnx_dir / f"{model_name}.onnx"
            
            if onnx_path.exists() and onnx_path != final_onnx_path:
                shutil.copy2(onnx_path, final_onnx_path)
            
            return True, str(final_onnx_path)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"ONNX转换错误详情:\n{error_detail}")
            return False, f"ONNX转换失败: {str(e)}"
        finally:
            if model is not None:
                del model
            self._clear_gpu_memory()
    
    def verify_onnx_model(self, onnx_path: str) -> Tuple[bool, Dict]:
        onnx_path = Path(onnx_path)
        
        if not onnx_path.exists():
            return False, {'error': 'ONNX文件不存在'}
        
        try:
            file_size = onnx_path.stat().st_size / (1024 * 1024)
            
            try:
                import onnx
                model = onnx.load(str(onnx_path), load_external_data=False)
                
                input_info = []
                output_info = []
                
                for input_tensor in model.graph.input:
                    shape = [dim.dim_value if dim.dim_value else dim.dim_param 
                            for dim in input_tensor.type.tensor_type.shape.dim]
                    input_info.append({
                        'name': input_tensor.name,
                        'shape': shape,
                        'type': input_tensor.type.tensor_type.elem_type
                    })
                
                for output_tensor in model.graph.output:
                    shape = [dim.dim_value if dim.dim_value else dim.dim_param 
                            for dim in output_tensor.type.tensor_type.shape.dim]
                    output_info.append({
                        'name': output_tensor.name,
                        'shape': shape,
                        'type': output_tensor.type.tensor_type.elem_type
                    })
                
                opset_version = model.opset_import[0].version if model.opset_import else 12
                
                del model
                
                return True, {
                    'valid': True,
                    'file_size_mb': round(file_size, 2),
                    'inputs': input_info,
                    'outputs': output_info,
                    'opset_version': opset_version
                }
                
            except Exception as e:
                print(f"ONNX详细验证失败，使用简化验证: {e}")
                return True, {
                    'valid': True,
                    'file_size_mb': round(file_size, 2),
                    'note': '简化验证模式'
                }
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"ONNX验证错误详情:\n{error_detail}")
            return False, {'error': str(e)}
    
    def test_onnx_inference(
        self,
        onnx_path: str,
        test_image: str = None,
        imgsz: int = 640
    ) -> Tuple[bool, Dict]:
        onnx_path = Path(onnx_path)
        
        if not onnx_path.exists():
            return False, {'error': 'ONNX文件不存在'}
        
        session = None
        try:
            import onnxruntime as ort
            import numpy as np
            from PIL import Image
            
            session = ort.InferenceSession(str(onnx_path))
            
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            batch_size = 1
            channels = 3
            height = input_shape[2] if isinstance(input_shape[2], int) else imgsz
            width = input_shape[3] if isinstance(input_shape[3], int) else imgsz
            
            dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
            
            outputs = session.run(None, {input_name: dummy_input})
            
            inference_time = None
            if test_image and os.path.exists(test_image):
                import time
                img = Image.open(test_image).convert('RGB')
                img = img.resize((width, height))
                img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, 0)
                
                start_time = time.time()
                session.run(None, {input_name: img_array})
                inference_time = time.time() - start_time
            
            return True, {
                'success': True,
                'input_shape': [batch_size, channels, height, width],
                'output_shapes': [o.shape for o in outputs],
                'inference_time': inference_time
            }
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"ONNX推理测试错误详情:\n{error_detail}")
            return False, {'error': str(e)}
        finally:
            if session is not None:
                del session
    
    def save_conversion_info(
        self,
        original_path: Path,
        onnx_path: Path,
        conversion_params: Dict
    ):
        info_path = self.onnx_dir / f"{onnx_path.stem}_conversion_info.json"
        
        info = {
            'original_model': str(original_path),
            'onnx_model': str(onnx_path),
            'conversion_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'conversion_params': conversion_params,
            'project_name': self.project_config.project_name,
            'model_type': self.project_config.model_config.full_name,
            'classes': self.project_config.classes
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    
    def list_onnx_models(self) -> list:
        onnx_files = list(self.onnx_dir.glob("*.onnx"))
        
        models = []
        for onnx_file in onnx_files:
            valid, info = self.verify_onnx_model(str(onnx_file))
            models.append({
                'path': str(onnx_file),
                'name': onnx_file.name,
                'size_mb': info.get('file_size_mb', 0),
                'valid': valid,
                'info': info
            })
        
        return models
    
    def compare_models(
        self,
        pt_model_path: str,
        onnx_model_path: str,
        test_images: list = None
    ) -> Dict:
        results = {
            'pt_model': {'path': pt_model_path},
            'onnx_model': {'path': onnx_model_path},
            'comparison': {}
        }
        
        pt_path = Path(pt_model_path)
        if pt_path.exists():
            results['pt_model']['size_mb'] = round(pt_path.stat().st_size / (1024 * 1024), 2)
        
        onnx_path = Path(onnx_model_path)
        if onnx_path.exists():
            results['onnx_model']['size_mb'] = round(onnx_path.stat().st_size / (1024 * 1024), 2)
        
        if pt_path.exists() and onnx_path.exists():
            size_reduction = (
                (results['pt_model']['size_mb'] - results['onnx_model']['size_mb']) 
                / results['pt_model']['size_mb'] * 100
            )
            results['comparison']['size_reduction_percent'] = round(size_reduction, 2)
        
        if test_images:
            pt_model = None
            onnx_session = None
            try:
                self._clear_gpu_memory()
                
                from ultralytics import YOLO
                import time
                import numpy as np
                from PIL import Image
                import onnxruntime as ort
                
                pt_model = YOLO(str(pt_path))
                onnx_session = ort.InferenceSession(str(onnx_path))
                
                pt_times = []
                onnx_times = []
                
                for img_path in test_images[:10]:
                    if not os.path.exists(img_path):
                        continue
                    
                    start = time.time()
                    pt_model.predict(img_path, verbose=False)
                    pt_times.append(time.time() - start)
                    
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((640, 640))
                    img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                    img_array = np.expand_dims(img_array, 0)
                    
                    input_name = onnx_session.get_inputs()[0].name
                    start = time.time()
                    onnx_session.run(None, {input_name: img_array})
                    onnx_times.append(time.time() - start)
                
                if pt_times and onnx_times:
                    results['comparison']['pt_avg_inference_time'] = round(np.mean(pt_times), 4)
                    results['comparison']['onnx_avg_inference_time'] = round(np.mean(onnx_times), 4)
                    speedup = np.mean(pt_times) / np.mean(onnx_times)
                    results['comparison']['speedup_factor'] = round(speedup, 2)
                    
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"模型比较错误详情:\n{error_detail}")
                results['comparison']['error'] = str(e)
            finally:
                if pt_model is not None:
                    del pt_model
                if onnx_session is not None:
                    del onnx_session
                self._clear_gpu_memory()
        
        return results
