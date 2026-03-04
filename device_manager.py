from typing import Optional


class DeviceManager:
    """全局设备管理工具 - 强制GPU模式"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化设备管理器 - 强制GPU"""
        self.torch_available = False
        self.cuda_available = False
        self.cuda_device_count = 0
        self.gpu_error_message = None
        
        try:
            import torch
            self.torch = torch
            self.torch_available = True
            self._check_cuda_availability()
        except Exception as e:
            self.gpu_error_message = f"PyTorch 加载失败: {e}"
            print(f"[设备管理] {self.gpu_error_message}")
    
    def _check_cuda_availability(self):
        """检查CUDA可用性 - 强制要求GPU"""
        if self.torch_available:
            self.cuda_available = self.torch.cuda.is_available()
            self.cuda_device_count = self.torch.cuda.device_count() if self.cuda_available else 0
            
            if self.cuda_available:
                print(f"[设备管理] CUDA 可用，检测到 {self.cuda_device_count} 个GPU设备")
                for i in range(self.cuda_device_count):
                    device_name = self.torch.cuda.get_device_name(i)
                    print(f"[设备管理] GPU {i}: {device_name}")
            else:
                self.gpu_error_message = "CUDA 不可用！此应用程序需要NVIDIA GPU支持。"
                print(f"[设备管理] {self.gpu_error_message}")
        else:
            self.gpu_error_message = "PyTorch 未正确加载，无法检测GPU。"
            print(f"[设备管理] {self.gpu_error_message}")
    
    def is_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        return self.cuda_available and self.torch_available
    
    def get_gpu_error_message(self) -> Optional[str]:
        """获取GPU错误信息"""
        return self.gpu_error_message
    
    def get_best_device(self) -> str:
        """获取最佳设备 - 强制GPU"""
        if not self.is_gpu_available():
            raise RuntimeError(f"GPU不可用: {self.gpu_error_message}")
        
        if self.cuda_device_count > 1:
            return ','.join([f'cuda:{i}' for i in range(self.cuda_device_count)])
        else:
            return 'cuda:0'
    
    def resolve_device(self, device: str = 'auto') -> str:
        """解析设备参数 - 强制GPU，不允许CPU"""
        if not self.is_gpu_available():
            raise RuntimeError(f"GPU不可用，无法进行推理或训练: {self.gpu_error_message}")
        
        if device == 'auto':
            return self.get_best_device()
        elif device == 'cpu':
            raise RuntimeError("此应用程序不允许使用CPU模式，请使用GPU。")
        elif device.isdigit():
            device_id = int(device)
            if device_id >= self.cuda_device_count:
                raise RuntimeError(f"GPU设备 {device_id} 不存在，可用设备: 0-{self.cuda_device_count-1}")
            return f'cuda:{device}'
        elif device.startswith('cuda'):
            return device
        
        return device
    
    def get_device_info(self) -> dict:
        """获取设备信息"""
        device_names = []
        if self.is_gpu_available():
            device_names = [self.torch.cuda.get_device_name(i) for i in range(self.cuda_device_count)]
        
        return {
            'torch_available': self.torch_available,
            'cuda_available': self.cuda_available,
            'cuda_device_count': self.cuda_device_count,
            'best_device': self.get_best_device() if self.is_gpu_available() else None,
            'device_names': device_names,
            'gpu_error': self.gpu_error_message
        }
    
    def warmup_device(self):
        """预热设备，确保设备可用"""
        if not self.is_gpu_available():
            raise RuntimeError(f"GPU不可用: {self.gpu_error_message}")
        
        try:
            x = self.torch.randn(1, 1).cuda()
            print("[设备管理] GPU 预热成功")
            return True
        except Exception as e:
            print(f"[设备管理] GPU 预热失败: {e}")
            return False


# 创建全局设备管理器实例
device_manager = DeviceManager()
