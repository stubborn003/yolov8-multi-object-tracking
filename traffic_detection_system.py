import cv2  # OpenCV：视频读取、保存、帧处理
import torch  # PyTorch：深度学习模型推理
import numpy as np  # 数值计算，处理坐标和速度
import os  # 文件路径和文件夹操作
import time  # 时间戳，计算处理时间
from ultralytics import YOLO  # YOLOv8目标检测与跟踪模型
from PyQt5 import QtWidgets, QtGui, QtCore  # PyQt5：GUI界面
from PyQt5.QtWidgets import QFileDialog, QMessageBox  # 文件选择对话框、提示框
from ui_main_window import Ui_MainWindow  # Qt Designer生成的UI文件
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Matplotlib嵌入PyQt5
from matplotlib.figure import Figure  # Matplotlib绘图
import matplotlib as mpl  # Matplotlib配置
from database_integration import DBIntegration  # 数据库集成
from voice_alert import play_voice_alert  # 语音警报
from object_tracking import initialize_tracking, process_frame  # 跟踪和警报功能

# 启用cuDNN自动优化卷积运算速度（适合固定输入尺寸的视频检测）
torch.backends.cudnn.benchmark = True

# 解决Matplotlib中文显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


class SpeedAnalyzer:
    """
    车辆速度分析器
    作用：
        接收YOLOv8跟踪结果（track_id + 中心坐标），计算每辆车的瞬时速度并进行平均。
    算法原理：
        1. 保存每辆车的位置和时间戳
        2. 相邻帧之间计算像素距离 → 转换为实际距离
        3. 用时间差计算瞬时速度
        4. 用滑动窗口（最近5次速度）做简单滤波，减少抖动
    """
    def __init__(self, pixels_per_meter=5):
        self.tracks = {}  # 存储每辆车的轨迹信息
        self.speeds = {}  # 存储每辆车的平滑速度（km/h）
        self.pixels_per_meter = pixels_per_meter  # 像素到米的比例（需要根据场景校准）
        self.all_tracked_vehicles = set()  # 累计跟踪的车辆ID

    def update(self, track_id, center, timestamp):
        """
        更新车辆位置并计算速度
        参数：
            track_id: YOLOv8跟踪输出的车辆ID
            center: 当前帧车辆边界框中心坐标 (x, y)
            timestamp: 当前帧时间戳
        流程：
            1. 如果是新车辆，初始化轨迹数据
            2. 否则，计算与上一帧的位置差和时间差
            3. 像素距离转换为实际距离
            4. 计算瞬时速度，用滑动窗口保存最近5次速度
            5. 更新平均速度
        """
        self.all_tracked_vehicles.add(track_id)

        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'prev_pos': center,
                'prev_time': timestamp,
                'speeds': np.zeros(5),  # 滑动窗口保存最近5次速度
                'speed_index': 0,  # 当前速度数组索引
                'last_seen': timestamp,
                'positions': [center]
            }
            return

        prev_pos = self.tracks[track_id]['prev_pos']
        prev_time = self.tracks[track_id]['prev_time']
        self.tracks[track_id]['last_seen'] = timestamp
        self.tracks[track_id]['positions'].append(center)

        time_diff = timestamp - prev_time
        if time_diff <= 0.001:  # 避免时间差太小导致速度异常
            return

        # 计算欧几里得距离（像素）
        distance_pixels = np.linalg.norm(center - prev_pos)
        distance_meters = distance_pixels / self.pixels_per_meter
        speed_m_per_s = distance_meters / time_diff
        speed_km_per_h = speed_m_per_s * 3.6  # m/s → km/h

        if 0 < speed_km_per_h < 200:  # 速度过滤
            tracks_data = self.tracks[track_id]
            speeds = tracks_data['speeds']
            speed_index = tracks_data['speed_index']
            speeds[speed_index] = speed_km_per_h
            tracks_data['speed_index'] = (speed_index + 1) % len(speeds)

            valid_speeds = speeds[speeds > 0]
            if len(valid_speeds) > 0:
                self.speeds[track_id] = np.mean(valid_speeds)
            else:
                self.speeds[track_id] = 0

        self.tracks[track_id]['prev_pos'] = center
        self.tracks[track_id]['prev_time'] = timestamp

    def calculate_average_speed(self):
        """计算所有车辆的平均速度（过滤掉0值）"""
        if not self.speeds:
            return 0
        valid_speeds = [v for v in self.speeds.values() if v > 0]
        if not valid_speeds:
            return 0
        return np.mean(valid_speeds)

    def get_vehicle_count(self):
        """返回累计跟踪的车辆总数"""
        return len(self.all_tracked_vehicles)


class MainApp(QtWidgets.QMainWindow):
    """
    智慧交通检测系统主类
    作用：
        视频/摄像头实时车辆检测、跟踪、速度计算、流量统计、UI显示
    主要技术：
        - YOLOv8（目标检测 + ByteTrack跟踪）
        - OpenCV（视频读取、保存、帧处理）
        - PyQt5（GUI界面显示、交互）
        - Matplotlib（实时车辆流量图绘制）
    运行流程：
        1. 选择视频或摄像头
        2. 点击开始 → 初始化视频捕获
        3. 定时器循环读取帧 → YOLO检测+跟踪
        4. 调用SpeedAnalyzer计算速度
        5. 更新UI：视频帧、统计信息、流量图
        6. 点击停止 → 释放资源
    """
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  # 加载UI界面

        self.setMinimumSize(900, 650)  # 设置窗口最小尺寸
        self.setWindowTitle("智慧交通检测系统")  # 设置窗口标题

        # 绑定按钮事件
        self.ui.open_video_btn.clicked.connect(self.select_video_file)  # 选择视频文件
        self.ui.open_camera_btn.clicked.connect(self.use_camera)  # 使用摄像头

        self.create_control_buttons()  # 创建开始/停止按钮
        self.create_status_display()  # 创建流量图和统计显示区域

        # 设备选择（GPU优先）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载YOLO模型（优先加载自定义模型best.pt，否则加载官方yolov8n.pt）
        model_pt_path = "best.pt"
        if not os.path.exists(model_pt_path):
            self.add_warning("使用默认模型")
            model_pt_path = 'yolov8n.pt'

        try:
            self.model = YOLO(model_pt_path).to(self.device)
            print("模型加载成功")
        except Exception as e:
            self.add_warning(f"模型加载失败: {str(e)}")
            return

        # 初始化参数
        self.RESULT_PATH = "result.mp4"  # 处理结果保存路径
        self.WARNING_FOLDER = "warning_frames"  # 异常帧保存文件夹
        self.VIDEO_PATH = ""  # 视频文件路径

        self.camera_index = 0  # 默认摄像头索引
        self.using_camera = False  # 是否使用摄像头
        self.processing = False  # 是否正在处理视频

        if not os.path.exists(self.WARNING_FOLDER):
            os.makedirs(self.WARNING_FOLDER)  # 创建异常帧保存文件夹

        self.speed_analyzer = SpeedAnalyzer(pixels_per_meter=5)  # 初始化速度分析器
        self.db_integration = DBIntegration()  # 初始化数据库集成

        self.frame_count = 0  # 帧计数器
        self.last_results = None  # 上一帧检测结果
        self.video_label = None  # 视频显示标签
        self.timer = QtCore.QTimer(self)  # 定时器，控制帧处理频率
        self.timer.timeout.connect(self.update_frame)  # 定时器绑定帧处理函数

        # 统计数据
        self.current_vehicles = 0  # 当前帧车辆数
        self.flow_x = []  # 流量图时间轴
        self.flow_y = []  # 流量图车辆数轴

        self.inference_times = []  # 推理时间列表
        self.frame_times = []  # 帧处理时间列表

        # 跟踪和警报相关变量
        self.track_history = None
        self.entered_ids = None
        self.entry_time = None
        self.warned_ids = None
        self.count_passed = None
        self.count_exited = None
        self.polygon_points = None
        self.polygon_points1 = None

        self.init_video_label()  # 初始化视频显示区域

    def init_video_label(self):
        """初始化视频显示区域（清空原有控件，创建新的占位标签）"""
        layout = self.ui.video_frame.layout()
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.video_label = QtWidgets.QLabel("请选择视频源或摄像头")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                font: 16pt "华文新魏";
                color: #6c757d;
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
            }
        """)
        layout.addWidget(self.video_label)

    def create_control_buttons(self):
        """创建开始/停止按钮并绑定事件"""
        button_frame = self.ui.open_video_btn.parent().parent()
        button_layout = button_frame.layout()

        if button_layout.count() > 2:
            for i in range(button_layout.count() - 1, 1, -1):
                item = button_layout.itemAt(i)
                if isinstance(item, QtWidgets.QSpacerItem):
                    button_layout.removeItem(item)

        self.start_btn = QtWidgets.QPushButton("▶ 开始处理")
        self.start_btn.setStyleSheet("""
            QPushButton {
                font: 12pt "华文新魏";
                padding: 10px;
                background-color: #81c784;
                color: #ffffff;
                border-radius: 5px;
                margin-top: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #66bb6a;
            }
            QPushButton:disabled {
                background-color: #a5d6a7;
            }
            QPushButton:pressed {
                background-color: #4caf50;
            }
        """)
        self.start_btn.clicked.connect(self.start_processing)

        self.stop_btn = QtWidgets.QPushButton("■ 停止处理")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                font: 12pt "华文新魏";
                padding: 10px;
                background-color: #e57373;
                color: #ffffff;
                border-radius: 5px;
                margin-top: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ef5350;
            }
            QPushButton:disabled {
                background-color: #ef9a9a;
            }
            QPushButton:pressed {
                background-color: #f44336;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()

    def create_status_display(self):
        """创建车辆流量图（Matplotlib嵌入PyQt5）"""
        layout = QtWidgets.QVBoxLayout(self.ui.chart_placeholder)
        layout.setContentsMargins(5, 5, 5, 5)

        self.flow_figure = Figure(figsize=(4, 3), dpi=100)
        self.flow_figure.patch.set_facecolor('#f8f9fa')
        self.flow_canvas = FigureCanvas(self.flow_figure)
        self.flow_ax = self.flow_figure.add_subplot(111)

        self.flow_ax.set_facecolor('#ffffff')
        self.flow_ax.spines['top'].set_visible(False)
        self.flow_ax.spines['right'].set_visible(False)
        self.flow_ax.spines['left'].set_color('#adb5bd')
        self.flow_ax.spines['bottom'].set_color('#adb5bd')
        self.flow_ax.tick_params(colors='#6c757d', labelsize=8)

        self.flow_ax.set_xlabel('时间', fontsize=9, color='#495057')
        self.flow_ax.set_ylabel('车辆数', fontsize=9, color='#495057')
        self.flow_ax.set_title('车辆流量图', fontsize=11, color='#343a40', pad=10)

        layout.addWidget(self.flow_canvas)

        self.flow_x = []
        self.flow_y = []

        self.ui.warning_text.setStyleSheet("""
            QTextEdit {
                font: 10pt "Microsoft YaHei";
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                color: #495057;
            }
        """)

    def add_warning(self, message):
        """添加系统日志/警告信息到文本框"""
        current_time = time.strftime("%H:%M:%S")
        self.ui.warning_text.append(f"[{current_time}] {message}")

        scrollbar = self.ui.warning_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        lines = self.ui.warning_text.toPlainText().split('\n')
        if len(lines) > 30:
            self.ui.warning_text.setText('\n'.join(lines[-30:]))

    def select_video_file(self):
        """选择视频文件并更新显示"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )

        if file_path:
            self.VIDEO_PATH = file_path
            self.using_camera = False
            filename = os.path.basename(file_path)

            self.video_label.setText(f"已选择视频:\n{filename}")
            self.video_label.setStyleSheet("""
                QLabel {
                    font: 14pt "华文新魏";
                    color: #81c784;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 20px;
                }
            """)

            self.add_warning(f"选择视频文件: {filename}")

    def use_camera(self):
        """切换到摄像头并更新显示"""
        self.using_camera = True
        self.VIDEO_PATH = f"摄像头{self.camera_index}"

        self.video_label.setText(f"使用摄像头 {self.camera_index}")
        self.video_label.setStyleSheet("""
            QLabel {
                font: 14pt "华文新魏";
                color: #7986cb;
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
            }
        """)

        self.add_warning(f"切换至摄像头: {self.camera_index}")

    def start_processing(self):
        """开始处理视频/摄像头"""
        if self.using_camera or (self.VIDEO_PATH and os.path.exists(self.VIDEO_PATH)):
            self.processing = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.frame_count = 0
            self.speed_analyzer = SpeedAnalyzer(pixels_per_meter=5)

            self.flow_x = []
            self.flow_y = []
            self.flow_ax.clear()
            self.flow_ax.set_xlabel('时间', fontsize=9, color='#495057')
            self.flow_ax.set_ylabel('车辆数', fontsize=9, color='#495057')
            self.flow_ax.set_title('车辆流量图', fontsize=11, color='#343a40', pad=10)
            self.flow_canvas.draw()

            # 初始化跟踪和警报相关变量
            (self.videowriter, self.track_history, self.entered_ids, 
             self.entry_time, self.warned_ids, self.count_passed, 
             self.count_exited, self.polygon_points, self.polygon_points1, 
             self.fps, self.frame_width, self.frame_height) = initialize_tracking(
                self.VIDEO_PATH, self.RESULT_PATH, self.WARNING_FOLDER
            )

            # 记录处理开始时间
            self.process_start_time = time.time()

            self.setup_video()
            self.add_warning("开始处理视频")
        else:
            QMessageBox.warning(self, "提示", "请先选择视频文件或摄像头")

    def stop_processing(self):
        """停止处理视频/摄像头"""
        self.processing = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_current_process()
        self.add_warning("已停止处理")

    def setup_video(self):
        """初始化视频捕获和保存器"""
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

        if self.using_camera:
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                self.add_warning("摄像头打开失败，请检查摄像头连接")
                self.stop_processing()
                return
            self.fps = 30
            self.frame_width = 640
            self.frame_height = 480
        else:
            self.capture = cv2.VideoCapture(self.VIDEO_PATH)
            if not self.capture.isOpened():
                self.add_warning("视频打开失败，请检查文件路径")
                self.stop_processing()
                return

            self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.fps <= 0:
                self.fps = 30

        if not self.using_camera:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.videowriter = cv2.VideoWriter(
                self.RESULT_PATH,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
        else:
            self.videowriter = None

        update_interval = max(33, int(1000 / min(30, self.fps)))
        self.timer.start(update_interval)

        self.add_warning(f"视频源设置完成: {self.frame_width}x{self.frame_height} @ {self.fps}fps")

    def update_frame(self):
        """处理单帧视频：检测、跟踪、速度计算、UI更新"""
        if not self.processing:
            return

        frame_start_time = time.time()

        success, frame = self.capture.read()
        if not success:
            if self.using_camera:
                self.add_warning("摄像头读取失败")
            else:
                self.add_warning("视频读取完成")
                self.stop_current_process()
            return

        try:
            # 使用 process_frame 函数处理帧，集成警报功能
            (annotated_frame, self.count_passed, self.count_exited, 
             self.entered_ids, self.entry_time, self.warned_ids, 
             self.track_history) = process_frame(
                frame, self.model, self.videowriter, self.track_history, 
                self.entered_ids, self.entry_time, self.warned_ids, 
                self.count_passed, self.count_exited, self.polygon_points, 
                self.polygon_points1, play_voice_alert, self.WARNING_FOLDER,
                warning_display=self.ui.warning_text
            )

            inference_time = time.time() - frame_start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 10:
                self.inference_times = self.inference_times[-10:]

        except Exception as e:
            self.add_warning(f"推理错误: {str(e)}")
            print(f"推理错误: {e}")
            return

        if self.videowriter is not None:
            self.videowriter.write(annotated_frame)

        # 计算车辆数
        self.current_vehicles = self.count_passed - self.count_exited
        
        # 更新速度分析器中的数据
        timestamp = time.time()
        for track_id, track in self.track_history.items():
            if track:
                # 获取最新的位置
                x, y = track[-1]
                center = np.array([x, y], dtype=np.float32)
                # 更新速度分析器
                self.speed_analyzer.update(track_id, center, timestamp)

        self.update_ui_display(annotated_frame)
        self.frame_count += 1

        if self.frame_count % 5 == 0:
            self.update_status_and_chart()

        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 10:
            self.frame_times = self.frame_times[-10:]

    def update_ui_display(self, frame):
        """将OpenCV帧转换为Qt显示格式并更新"""
        try:
            display_width = self.ui.video_frame.width() - 20
            display_height = self.ui.video_frame.height() - 20

            if display_width > 10 and display_height > 10:
                frame_height, frame_width = frame.shape[:2]
                aspect_ratio = frame_width / frame_height

                display_aspect = display_width / display_height

                if display_aspect > aspect_ratio:
                    new_height = display_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = display_width
                    new_height = int(new_width / aspect_ratio)

                resized_frame = cv2.resize(frame, (new_width, new_height))

                height, width, channel = resized_frame.shape
                bytesPerLine = 3 * width

                if not resized_frame.flags['C_CONTIGUOUS']:
                    resized_frame = np.ascontiguousarray(resized_frame)

                qImg = QtGui.QImage(
                    resized_frame.data,
                    width,
                    height,
                    bytesPerLine,
                    QtGui.QImage.Format_RGB888
                ).rgbSwapped()

                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.video_label.setPixmap(pixmap)
                self.video_label.setStyleSheet("border: none;")

        except Exception as e:
            print(f"显示更新错误: {e}")

    def update_status_and_chart(self):
        """更新统计信息和流量图"""
        try:
            avg_speed = self.speed_analyzer.calculate_average_speed()
            total_vehicles = self.speed_analyzer.get_vehicle_count()

            scrollbar = self.ui.stats_text.verticalScrollBar()
            scroll_position = scrollbar.value()

            inference_speed = 0
            if self.inference_times:
                inference_speed = np.mean(self.inference_times) * 1000

            stats_html = f"""
                <div style='font-family: "Microsoft YaHei"; font-size: 11pt; color: #495057;'>
                    <h3 style='color: #343a40; margin-top: 0;'>📊 实时统计</h3>
                    <p style='margin: 5px 0;'>🚗 <b>平均车速:</b> <span style='color: #81c784;'>{avg_speed:.1f} km/h</span></p>
                    <p style='margin: 5px 0;'>📈 <b>累计车辆:</b> <span style='color: #7986cb;'>{total_vehicles}</span></p>
                    <p style='margin: 5px 0;'>👁️ <b>当前车辆:</b> <span style='color: #a1887f;'>{self.current_vehicles}</span></p>
                    <p style='margin: 5px 0;'>⏱️ <b>处理帧数:</b> <span style='color: #4db6ac;'>{self.frame_count}</span></p>
                    <p style='margin: 5px 0;'>⚡ <b>推理速度:</b> <span style='color: #ffb74d;'>{inference_speed:.1f} ms</span></p>
                </div>
                """

            was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10

            self.ui.stats_text.setHtml(stats_html)

            if was_at_bottom:
                scrollbar.setValue(scrollbar.maximum())
            else:
                scrollbar.setValue(min(scroll_position, scrollbar.maximum()))

            current_time = time.strftime("%H:%M:%S")
            self.flow_x.append(current_time)
            self.flow_y.append(self.current_vehicles)

            if len(self.flow_x) > 15:
                self.flow_x = self.flow_x[-15:]
                self.flow_y = self.flow_y[-15:]

            self.update_flow_graph()

            fps_text = f"FPS: {1 / np.mean(self.frame_times):.1f}" if self.frame_times else "等待数据..."
            status_text = f"📍 车辆检测 | 🚗 {self.current_vehicles} 辆车 | ⚡ {fps_text} | 📍 智慧交通检测系统"
            self.ui.statusBar.showMessage(status_text)

            # 存储统计信息到数据库
            self.db_integration.store_statistics(
                self.speed_analyzer,
                self.current_vehicles,
                self.frame_count,
                self.inference_times,
                self.frame_times
            )

        except Exception as e:
            print(f"状态更新错误: {e}")

    def update_flow_graph(self):
        """更新车辆流量图"""
        try:
            self.flow_ax.clear()

            if len(self.flow_x) > 0:
                x_indices = list(range(len(self.flow_x)))

                if len(self.flow_x) > 1:
                    line_color = '#7986cb'
                    self.flow_ax.plot(x_indices, self.flow_y, '-', linewidth=2,
                                      color=line_color,
                                      marker='o', markersize=4,
                                      markerfacecolor='white', markeredgecolor=line_color)

                    self.flow_ax.fill_between(x_indices, self.flow_y, 0, alpha=0.1, color=line_color)
                else:
                    self.flow_ax.plot(x_indices, self.flow_y, 'o', markersize=8, color='#7986cb')

                if len(self.flow_x) > 1:
                    step = max(1, len(self.flow_x) // 4)
                    tick_indices = list(range(0, len(self.flow_x), step))
                    if tick_indices:
                        tick_labels = [self.flow_x[i] for i in tick_indices]
                        self.flow_ax.set_xticks(tick_indices)
                        self.flow_ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

                if self.flow_y:
                    y_max = max(self.flow_y) if max(self.flow_y) > 0 else 5
                    self.flow_ax.set_ylim(0, y_max * 1.2)
                    y_ticks = list(range(0, int(y_max) + 2, max(1, int(y_max / 3))))
                    if y_ticks:
                        self.flow_ax.set_yticks(y_ticks)

                self.flow_ax.grid(True, linestyle='--', alpha=0.3, color='#e0e0e0')

            self.flow_ax.set_xlabel('时间', fontsize=9, color='#495057')
            self.flow_ax.set_ylabel('车辆数', fontsize=9, color='#495057')
            self.flow_ax.set_title('车辆流量图', fontsize=11, color='#343a40', pad=10)

            self.flow_figure.tight_layout()
            self.flow_canvas.draw()

        except Exception as e:
            print(f"图表更新错误: {e}")

    def stop_current_process(self):
        """释放资源，停止处理"""
        if not self.processing:
            return

        if self.timer.isActive():
            self.timer.stop()

        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

        if hasattr(self, 'videowriter') and self.videowriter is not None:
            self.videowriter.release()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.processing = False

        avg_speed = self.speed_analyzer.calculate_average_speed()
        total_vehicles = self.speed_analyzer.get_vehicle_count()

        if self.frame_count > 0:
            avg_fps = self.frame_count / (
                time.time() - self.process_start_time if hasattr(self, 'process_start_time') else 1)
            summary = f"处理完成: {self.frame_count}帧, 平均{avg_speed:.1f}km/h, 共{total_vehicles}车, 平均{avg_fps:.1f}FPS"
        else:
            summary = f"处理完成: {self.frame_count}帧, 平均{avg_speed:.1f}km/h, 共{total_vehicles}车"

        self.add_warning(summary)

        self.video_label.clear()
        self.video_label.setText("处理完成\n请选择新的视频源")
        self.video_label.setStyleSheet("""
            QLabel {
                font: 14pt "华文新魏";
                color: #81c784;
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
            }
        """)

    def resizeEvent(self, event):
        """窗口大小改变事件（未实现特殊功能）"""
        super().resizeEvent(event)

    def closeEvent(self, event):
        """窗口关闭事件，确保释放资源"""
        self.stop_current_process()
        # 关闭数据库连接
        if hasattr(self, 'db_integration'):
            self.db_integration.close()
        event.accept()


if __name__ == "__main__":
    import sys

    try:
        app = QtWidgets.QApplication(sys.argv)
        main_app = MainApp()
        main_app.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback

        traceback.print_exc()