# 🔍 YOLO标注与训练自动化工具

一个基于PyQt5开发的YOLO目标检测模型标注、训练、推理一体化工具。支持多种YOLO模型的完整工作流程，包括数据标注、模型训练、模型推理、ONNX转换等功能。

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-MultiVersion-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ 功能特性

### 📝 数据标注
- 支持矩形框标注
- 支持多类别标注
- 支持YOLO格式导出
- 标注可视化预览

### 🏋️ 模型训练
- **支持多种YOLO版本**：
  - YOLOv5 (n/s/m/l/x) - 经典稳定版本
  - YOLOv8 (n/s/m/l/x) - 推荐使用 ⭐
  - YOLOv9 (c/e) - 高精度版本
  - YOLOv10 (n/s/m/b/l/x) - 最新版本
  - YOLO11 (n/s/m/l/x) - 最新一代
- 支持自定义训练参数
- 实时训练进度监控
- 自动保存最佳模型
- 训练报告生成

### 🔎 推理测试
- 支持图像/视频推理
- 支持批量图像推理
- 支持GPU加速推理
- **支持ONNX模型GPU推理**
- 实时FPS显示
- 中文标签支持

### 🔄 模型转换
- PyTorch模型转ONNX格式
- ONNX模型验证
- 支持ONNX GPU推理

### 📊 其他功能
- 项目管理
- 数据集自动划分
- 训练报告生成
- 模型管理（删除、打开位置）

## 🖥️ 系统要求

### 硬件要求
- **CPU**: 4核心以上
- **内存**: 8GB以上
- **GPU**: NVIDIA显卡（推荐RTX 3060及以上）
- **硬盘**: 10GB以上可用空间

### 软件要求
- **操作系统**: Windows 10/11 64位
- **Python**: 3.9 - 3.12
- **CUDA**: 12.1+（GPU版本）
- **NVIDIA驱动**: 最新版本

## 🚀 快速开始

### 方法一：一键部署（推荐）

1. **克隆仓库**
```bash
git clone https://github.com/your-username/yolo-annotation-tool.git
cd yolo-annotation-tool
```

2. **运行启动脚本**
双击 `run.bat` 文件，脚本将自动：
- 检测Python环境
- 创建虚拟环境
- 安装所有依赖
- 启动程序

### 方法二：手动安装

1. **创建虚拟环境**
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. **安装PyTorch GPU版本**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. **安装ONNX Runtime GPU版本**
```bash
pip install onnxruntime-gpu==1.19.2
```

4. **安装其他依赖**
```bash
pip install -r requirements.txt
```

5. **运行程序**
```bash
python app.py
```

## 📖 使用指南

### 1. 创建项目

1. 点击「新建项目」按钮
2. 输入项目名称（英文）
3. 设置检测类别（逗号分隔）
4. 选择项目保存位置

### 2. 数据标注

1. 打开项目后进入「标注工具」
2. 选择图像文件夹
3. 使用鼠标绘制边界框
4. 选择对应类别
5. 保存标注结果

### 3. 数据集划分

1. 进入「数据管理」
2. 设置训练集/验证集/测试集比例
3. 点击「划分数据集」
4. 自动生成YOLO格式配置

### 4. 模型训练

1. 进入「模型训练」
2. 选择模型类型和版本：
   | 模型 | 尺寸 | 说明 |
   |------|------|------|
   | YOLOv5 | n/s/m/l/x | 经典稳定，兼容性好 |
   | YOLOv8 | n/s/m/l/x | 推荐，性能均衡 ⭐ |
   | YOLOv9 | c/e | 高精度，适合复杂场景 |
   | YOLOv10 | n/s/m/b/l/x | 最新版本，NMS-free |
   | YOLO11 | n/s/m/l/x | 最新一代，性能最优 |
3. 设置训练参数：
   - 训练轮数（Epochs）
   - 批次大小（Batch Size）
   - 图像尺寸（Image Size）
   - 学习率（Learning Rate）
4. 点击「开始训练」

### 5. 推理测试

1. 进入「推理测试」
2. 选择训练好的模型（.pt或.onnx）
3. 点击「加载模型」
4. 选择图像或视频
5. 点击「开始推理」

### 6. ONNX转换

1. 进入「模型管理」
2. 选择要转换的.pt模型
3. 点击「转换为ONNX」
4. 转换完成后可用于部署

## 📁 项目结构

```
yolo-annotation-tool/
├── .venv/                   # 虚拟环境
├── projects/                # 项目目录
│   └── your_project/
│       ├── images/          # 图像文件
│       ├── labels/          # 标注文件
│       ├── models/          # 训练模型
│       │   └── onnx/        # ONNX模型
│       ├── config.json      # 项目配置
│       └── dataset.yaml     # 数据集配置
├── runs/                    # 训练结果
├── app.py                   # 主程序入口
├── annotator.py             # 标注工具
├── trainer.py               # 训练模块
├── inference.py             # 推理模块
├── onnx_converter.py        # ONNX转换
├── device_manager.py        # 设备管理
├── config.py                # 配置管理
├── project_manager.py       # 项目管理
├── dataset_splitter.py      # 数据集划分
├── report_generator.py      # 报告生成
├── requirements.txt         # 依赖列表
├── run.bat                  # 一键启动脚本
└── README.md                # 说明文档
```

## ⚙️ 配置说明

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| Epochs | 训练轮数 | 100 |
| Batch Size | 批次大小 | 16 |
| Image Size | 输入图像尺寸 | 640 |
| Learning Rate | 学习率 | 0.01 |
| Patience | 早停耐心值 | 50 |

### 推理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| Confidence | 置信度阈值 | 0.25 |
| IoU Threshold | IoU阈值 | 0.45 |
| Max Detections | 最大检测数 | 300 |

### 模型尺寸说明

| 尺寸 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| n (nano) | 最小 | 最快 | 较低 | 边缘设备、实时检测 |
| s (small) | 小 | 快 | 中等 | 移动端、嵌入式 |
| m (medium) | 中 | 中等 | 较高 | 通用场景 ⭐ |
| l (large) | 大 | 较慢 | 高 | 服务器部署 |
| x (extra) | 最大 | 最慢 | 最高 | 高精度需求 |

## 🔧 常见问题

### Q: GPU不可用怎么办？
A: 请确保：
1. 安装了NVIDIA显卡驱动
2. 安装了CUDA Toolkit 12.x
3. 安装了PyTorch GPU版本

### Q: ONNX模型加载失败？
A: 请确保安装了正确版本的onnxruntime-gpu：
```bash
pip install onnxruntime-gpu==1.19.2
```

### Q: 训练时显存不足？
A: 尝试减小batch_size或使用更小的模型（如yolov8n）

### Q: 中文显示乱码？
A: 程序已内置中文字体支持，如仍有问题请确保系统安装了微软雅黑字体

### Q: 如何选择合适的YOLO版本？
A: 
- **YOLOv8**: 推荐大多数用户使用，性能均衡
- **YOLOv5**: 需要兼容旧项目时使用
- **YOLOv9**: 追求高精度时使用
- **YOLOv10/YOLO11**: 想体验最新特性时使用

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO框架
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI框架
- [ONNX Runtime](https://onnxruntime.ai/) - 推理引擎

## 📮 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。

---

⭐ 如果这个项目对你有帮助，请给一个Star支持一下！
"# YOLO-Annotation-and-Training-Model-Integrated-Software" 
