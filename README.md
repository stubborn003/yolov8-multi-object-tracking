# 智慧交通检测系统

## 项目简介

本项目实现了一个完整的智慧交通检测系统，具备视频目标跟踪、异常行为预警、数据统计分析及可视化展示功能。系统利用YOLO模型进行实时车辆检测，结合多目标跟踪算法实现车辆轨迹追踪，能够监测特定区域的车辆行为并发出预警，同时计算车速和统计车流量，通过图形化界面直观展示结果，并支持将统计信息存储到本地MySQL数据库中。

## 功能特点

- ✅ **实时车辆检测与跟踪**：基于YOLO模型和多目标跟踪算法，实现视频流/摄像头的实时车辆检测与追踪
- ✅ **车速计算与分析**：通过车辆轨迹计算实时车速，并进行统计分析
- ✅ **车流量统计与可视化**：实时统计车流量数据，并通过图表直观展示
- ✅ **异常行为预警**：检测车辆异常行为（如超速、闯红灯等）并发出语音警报
- ✅ **图形化界面**：基于PyQt5实现友好的用户界面，支持视频播放、参数设置和结果展示
- ✅ **数据持久化**：将交通统计信息存储到本地MySQL数据库，支持历史数据查询与分析
- ✅ **多视频源支持**：支持本地视频文件和摄像头输入

## 项目结构

```
智慧交通检测系统/
├── 核心文件
│   ├── traffic_detection_system.py  # 主程序入口，整合各模块功能
│   ├── object_tracking.py          # 目标跟踪核心算法
│   ├── voice_alert.py              # 语音警报功能实现
│   ├── ui_main_window.py           # 图形化界面实现
│   └── database_integration.py     # 数据库集成模块
├── 工具类
│   └── database_utils.py           # 数据库工具类
├── 模型和数据
│   ├── best.pt                     # YOLO模型权重文件
│   ├── vehicles.yaml               # 车辆检测配置文件
│   ├── car_test3.mp4               # 测试视频文件
│   ├── result.mp4                  # 处理结果视频
│   └── warning_frames/             # 警告帧存储目录
├── 配置文件
│   └── requirements.txt            # 项目依赖库
└── README.md                       # 项目说明文档
```

## 文件详细说明

### 核心文件

1. **traffic_detection_system.py**：
   - 主程序入口，整合各模块功能
   - 搭建图形化界面，处理用户交互
   - 初始化检测和跟踪参数
   - 处理视频帧，更新数据统计
   - 调用语音警报和数据库存储功能

2. **object_tracking.py**：
   - 实现多目标跟踪算法
   - 计算目标交并比(IoU)
   - 非极大值抑制处理
   - 车辆速度计算
   - 异常行为检测

3. **voice_alert.py**：
   - 初始化语音引擎
   - 定义语音警告函数
   - 实现异常行为的语音提示

4. **ui_main_window.py**：
   - 基于PyQt5的图形化界面实现
   - 视频播放控件
   - 数据统计图表
   - 参数设置面板

5. **database_integration.py**：
   - 数据库集成接口
   - 简化主程序中的数据库操作
   - 提供统计信息存储方法

### 工具类

6. **database_utils.py**：
   - MySQL数据库连接管理
   - 数据库表结构创建
   - 数据插入和查询操作

### 模型和数据文件

7. **best.pt**：
   - YOLO模型权重文件
   - 用于车辆检测和跟踪

8. **vehicles.yaml**：
   - 车辆检测配置文件
   - 定义类别名称和模型参数

9. **car_test3.mp4**：
   - 测试视频文件
   - 用于系统功能演示

10. **result.mp4**：
    - 处理结果视频存储路径
    - 包含检测和跟踪结果的视频

11. **warning_frames/**：
    - 存储异常行为警告帧的目录
    - 保存触发警报时的视频帧

## 环境要求

- Python 3.10+
- MySQL 5.7+
- CUDA 11.8+ (可选，用于GPU加速)
- TensorRT 8.6.1 (可选，用于推理加速)

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据库

1. 启动MySQL服务
2. 创建数据库：
   ```sql
   CREATE DATABASE traffic_stats;
   ```
3. 数据库默认配置：
   - 主机：localhost
   - 用户名：root
   - 密码：《自己设置的密码》
   - 数据库：traffic_stats（如有导入不成功，可以手动创建数据库）

   如果需要修改数据库配置，请编辑 `database_utils.py` 文件中的连接参数。

### 3. 运行程序

```bash
python traffic_detection_system.py
```

### 4. 使用说明

1. 启动程序后，点击"选择视频文件"按钮加载测试视频
2. 或点击"使用摄像头"按钮使用实时摄像头输入
3. 点击"开始处理"按钮开始车辆检测和跟踪
4. 在界面上可以查看：
   - 实时视频流和检测结果
   - 车速统计图表
   - 车流量统计信息
   - 异常行为警告
5. 处理完成后，结果视频将保存为 `result.mp4`

## 数据库功能

### 数据库结构

系统使用MySQL数据库存储交通统计信息，表结构如下：

```sql
CREATE TABLE traffic_statistics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    avg_speed FLOAT,
    total_vehicles INT,
    current_vehicles INT,
    frame_count INT,
    inference_speed FLOAT,
    fps FLOAT
);
```

### 数据存储内容

- `timestamp`：数据记录时间戳
- `avg_speed`：平均车速
- `total_vehicles`：累计车辆数
- `current_vehicles`：当前画面中的车辆数
- `frame_count`：处理的视频帧数
- `inference_speed`：模型推理速度
- `fps`：处理帧率

## 技术栈

- **目标检测**：YOLOv8
- **目标跟踪**：多目标跟踪算法
- **图形界面**：PyQt5
- **图像处理**：OpenCV
- **数据分析**：NumPy, Pandas
- **可视化**：Matplotlib, pyqtgraph
- **数据库**：MySQL
- **语音合成**：pyttsx3

## 注意事项

1. 确保MySQL服务正在运行，并且数据库配置正确
2. 首次运行时会自动创建数据库表结构
3. 模型文件 `best.pt` 需要与主程序在同一目录下
4. 处理大型视频文件时可能需要较长时间
5. GPU加速需要正确安装CUDA和TensorRT环境
6. 语音警报功能需要系统支持语音合成

## 许可证

本项目仅供学习和研究使用。