#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库集成脚本
用于将main.py中的统计信息存储到MySQL数据库中
"""

import time
import numpy as np
from database_utils import DatabaseUtils

class DBIntegration:
    """
    数据库集成类
    用于将交通检测系统的统计信息存储到数据库
    """
    def __init__(self):
        """
        初始化数据库集成
        """
        self.db = DatabaseUtils()
        self.process_start_time = time.time()
    
    def store_statistics(self, speed_analyzer, current_vehicles, frame_count, inference_times, frame_times):
        """
        存储统计信息到数据库
        :param speed_analyzer: 速度分析器实例
        :param current_vehicles: 当前车辆数
        :param frame_count: 处理帧数
        :param inference_times: 推理时间列表
        :param frame_times: 帧处理时间列表
        :return: 是否存储成功
        """
        try:
            # 计算平均车速
            avg_speed = speed_analyzer.calculate_average_speed()
            
            # 获取累计车辆数
            total_vehicles = speed_analyzer.get_vehicle_count()
            
            # 计算平均推理速度
            inference_speed = 0
            if inference_times:
                inference_speed = np.mean(inference_times) * 1000  # 转换为毫秒
            
            # 计算帧率
            fps = 0
            if frame_times:
                avg_frame_time = np.mean(frame_times)
                if avg_frame_time > 0:
                    fps = 1 / avg_frame_time
            
            # 存储到数据库
            result = self.db.insert_statistics(
                avg_speed=avg_speed,
                total_vehicles=total_vehicles,
                current_vehicles=current_vehicles,
                frame_count=frame_count,
                inference_speed=inference_speed,
                fps=fps
            )
            
            return result
        except Exception as e:
            print(f"存储统计信息失败: {e}")
            return False
    
    def close(self):
        """
        关闭数据库连接
        """
        if self.db:
            self.db.close()


# 示例用法
if __name__ == "__main__":
    """
    示例用法，展示如何在main.py中集成
    """
    print("数据库集成示例")
    print("================")
    print("在main.py中集成的方法：")
    print("1. 在文件顶部导入DBIntegration")
    print("   from db_integration import DBIntegration")
    print("\n2. 在MainApp类的__init__方法中初始化")
    print("   self.db_integration = DBIntegration()")
    print("\n3. 在update_status_and_chart方法中添加存储调用")
    print("   self.db_integration.store_statistics(")
    print("       self.speed_analyzer,")
    print("       self.current_vehicles,")
    print("       self.frame_count,")
    print("       self.inference_times,")
    print("       self.frame_times")
    print("   )")
    print("\n4. 在closeEvent方法中关闭连接")
    print("   if hasattr(self, 'db_integration'):")
    print("       self.db_integration.close()")
    print("\n这样就可以在不修改main.py核心逻辑的情况下，将统计信息存储到数据库中。")