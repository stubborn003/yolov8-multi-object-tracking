from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    """主窗口界面类，用于构建智慧交通检测系统的GUI"""

    def setupUi(self, MainWindow):
        """
        设置主窗口的UI界面

        参数:
            MainWindow: QMainWindow实例，作为主窗口
        """
        # 设置主窗口基本属性
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1425, 887)

        # 设置主窗口样式表
        MainWindow.setStyleSheet("""
            QMainWindow {
                background-color: #e9ecef;
            }
            #centralwidget{
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.732955, y2:0.801, 
                                stop:0.282486 rgba(168, 182, 201, 180), 
                                stop:0.840909 rgba(227, 230, 232, 0));
                border-radius: 10px;
            }
        """)

        # 创建中央部件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 创建主垂直布局
        main_vlayout = QtWidgets.QVBoxLayout(self.centralwidget)
        main_vlayout.setContentsMargins(10, 10, 10, 10)
        main_vlayout.setSpacing(10)

        # 创建标题标签
        self.label_2 = QtWidgets.QLabel("智慧交通检测系统")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setStyleSheet("""
            font: 24pt "华文新魏";
            color: #495057;
            padding: 10px;
            background-color: rgba(173, 181, 189, 0.8);
            border-radius: 8px;
        """)
        main_vlayout.addWidget(self.label_2)

        # 创建内容水平布局（视频区域 + 控制面板）
        content_hlayout = QtWidgets.QHBoxLayout()
        content_hlayout.setSpacing(10)

        # ==================== 视频显示区域 ====================
        self.video_frame = QtWidgets.QFrame()
        self.video_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(248, 249, 250, 0.95);
                border-radius: 8px;
                border: 2px solid rgba(173, 181, 189, 0.4);
            }
        """)
        video_layout = QtWidgets.QVBoxLayout(self.video_frame)
        video_layout.setContentsMargins(5, 5, 5, 5)

        # 视频占位标签
        self.video_label_placeholder = QtWidgets.QLabel("视频显示区域")
        self.video_label_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label_placeholder.setStyleSheet("""
            font: 14pt "华文新魏";
            color: #6c757d;
        """)
        video_layout.addWidget(self.video_label_placeholder)

        # 将视频区域添加到内容布局（占70%宽度）
        content_hlayout.addWidget(self.video_frame, 7)

        # ==================== 控制面板 ====================
        control_panel = QtWidgets.QFrame()
        control_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(248, 249, 250, 0.95);
                border-radius: 8px;
                border: 2px solid rgba(173, 181, 189, 0.4);
            }
        """)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setSpacing(10)

        # 按钮框架
        button_frame = QtWidgets.QFrame()
        button_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(233, 236, 239, 0.9);
                border-radius: 6px;
                padding: 10px;
            }
        """)
        button_layout = QtWidgets.QVBoxLayout(button_frame)
        button_layout.setSpacing(8)

        # 选择视频文件按钮
        self.open_video_btn = QtWidgets.QPushButton("📁 选择视频文件")
        self.open_video_btn.setStyleSheet("""
            QPushButton {
                font: 12pt "华文新魏";
                padding: 10px;
                background-color: #95a5a6;
                color: #ffffff;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #6c7a7d;
            }
        """)

        # 切换摄像头按钮
        self.open_camera_btn = QtWidgets.QPushButton("📷 切换摄像头")
        self.open_camera_btn.setStyleSheet("""
            QPushButton {
                font: 12pt "华文新魏";
                padding: 10px;
                background-color: #7986cb;
                color: #ffffff;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #5c6bc0;
            }
            QPushButton:pressed {
                background-color: #3f51b5;
            }
        """)

        # 添加按钮到按钮布局
        button_layout.addWidget(self.open_video_btn)
        button_layout.addWidget(self.open_camera_btn)
        button_layout.addStretch()  # 添加弹性空间

        # 将按钮框架添加到控制面板
        control_layout.addWidget(button_frame)

        # 统计信息框架
        stats_frame = QtWidgets.QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(233, 236, 239, 0.9);
                border-radius: 6px;
                padding: 10px;
            }
        """)
        stats_layout = QtWidgets.QVBoxLayout(stats_frame)

        # 统计信息标题
        self.stats_label = QtWidgets.QLabel("实时统计信息")
        self.stats_label.setAlignment(QtCore.Qt.AlignCenter)
        self.stats_label.setStyleSheet("font: 14pt '华文新魏'; color: #495057;")
        stats_layout.addWidget(self.stats_label)

        # 统计信息文本框
        self.stats_text = QtWidgets.QTextEdit()
        self.stats_text.setReadOnly(True)  # 设置为只读
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                font: 11pt 'Microsoft YaHei';
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                color: #495057;
            }
        """)
        stats_layout.addWidget(self.stats_text)

        # 将统计信息框架添加到控制面板
        control_layout.addWidget(stats_frame)

        # 图表框架
        chart_frame = QtWidgets.QFrame()
        chart_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(233, 236, 239, 0.9);
                border-radius: 6px;
                padding: 10px;
            }
        """)
        chart_layout = QtWidgets.QVBoxLayout(chart_frame)

        # 图表标题
        self.chart_label = QtWidgets.QLabel("车辆流量图")
        self.chart_label.setAlignment(QtCore.Qt.AlignCenter)
        self.chart_label.setStyleSheet("font: 14pt '华文新魏'; color: #495057;")
        chart_layout.addWidget(self.chart_label)

        # 图表占位符
        self.chart_placeholder = QtWidgets.QWidget()
        self.chart_placeholder.setMinimumHeight(200)
        self.chart_placeholder.setStyleSheet("""
            background-color: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 4px;
        """)
        chart_layout.addWidget(self.chart_placeholder)

        # 将图表框架添加到控制面板
        control_layout.addWidget(chart_frame)

        # 系统日志框架
        warning_frame = QtWidgets.QFrame()
        warning_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(233, 236, 239, 0.9);
                border-radius: 6px;
                padding: 10px;
            }
        """)
        warning_layout = QtWidgets.QVBoxLayout(warning_frame)

        # 系统日志标题
        self.warning_label = QtWidgets.QLabel("系统日志")
        self.warning_label.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_label.setStyleSheet("font: 14pt '华文新魏'; color: #495057;")
        warning_layout.addWidget(self.warning_label)

        # 系统日志文本框
        self.warning_text = QtWidgets.QTextEdit()
        self.warning_text.setReadOnly(True)  # 设置为只读
        self.warning_text.setMaximumHeight(150)
        self.warning_text.setStyleSheet("""
            QTextEdit {
                font: 10pt 'Microsoft YaHei';
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                color: #495057;
            }
        """)
        warning_layout.addWidget(self.warning_text)

        # 将系统日志框架添加到控制面板
        control_layout.addWidget(warning_frame)

        # 设置控制面板中各部分的拉伸比例
        control_layout.setStretch(0, 1)  # 按钮区域
        control_layout.setStretch(1, 1)  # 统计信息
        control_layout.setStretch(2, 2)  # 图表区域
        control_layout.setStretch(3, 1)  # 系统日志

        # 将控制面板添加到内容布局（占30%宽度）
        content_hlayout.addWidget(control_panel, 3)

        # 将内容布局添加到主布局
        main_vlayout.addLayout(content_hlayout, 1)

        # 创建状态栏
        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setStyleSheet("""
            QStatusBar {
                background-color: rgba(173, 181, 189, 0.9);
                color: #495057;
                font: 10pt 'Microsoft YaHei';
                border-top: 1px solid #adb5bd;
            }
        """)
        main_vlayout.addWidget(self.statusBar)

        # 设置中央部件
        MainWindow.setCentralWidget(self.centralwidget)

        # 重新翻译UI文本
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        """
        设置UI界面的文本内容

        参数:
            MainWindow: QMainWindow实例
        """
        _translate = QtCore.QCoreApplication.translate

        # 设置窗口标题
        MainWindow.setWindowTitle(_translate("MainWindow", "智慧交通检测系统"))

        # 设置标题标签文本
        self.label_2.setText(_translate("MainWindow", "智慧交通检测系统"))

        # 设置视频占位文本
        self.video_label_placeholder.setText(_translate("MainWindow", "视频显示区域 - 请选择视频源"))

        # 设置统计信息标题
        self.stats_label.setText(_translate("MainWindow", "📊 实时统计信息"))

        # 设置图表标题
        self.chart_label.setText(_translate("MainWindow", "📈 车辆流量图"))

        # 设置系统日志标题
        self.warning_label.setText(_translate("MainWindow", "📝 系统日志"))