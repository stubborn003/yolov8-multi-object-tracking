import pymysql
import time
import datetime

class DatabaseUtils:
    def __init__(self, host='localhost', user='root', password='123456', db='traffic_stats'):
        """
        初始化数据库连接
        :param host: 数据库主机地址
        :param user: 数据库用户名
        :param password: 数据库密码
        :param db: 数据库名称
        """
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.connection = None
        self.cursor = None
        self.latest_data = None  # 最新数据
        self.last_write_time = time.time()  # 上次写入时间
        self.write_interval = 5  # 写入间隔（秒）
        self.connect()
        self.create_table()
    
    def connect(self):
        """
        连接到MySQL数据库
        """
        try:
            # 连接数据库
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.connection.cursor()
            
            # 创建数据库（如果不存在）
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db}")
            self.cursor.execute(f"USE {self.db}")
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
    
    def create_table(self):
        """
        创建流量统计表（先删除旧表再创建新表，确保字段结构正确）
        """
        try:
            # 先删除旧表（如果存在）
            drop_sql = "DROP TABLE IF EXISTS traffic_statistics"
            self.cursor.execute(drop_sql)
            
            # 创建新表，添加中文注释
            create_sql = """
            CREATE TABLE traffic_statistics (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT '记录ID',
                timestamp DATETIME NOT NULL COMMENT '时间戳',
                avg_speed FLOAT NOT NULL COMMENT '平均车速',
                total_vehicles INT NOT NULL COMMENT '累计车辆数',
                current_vehicles INT NOT NULL COMMENT '当前车辆数',
                frame_count INT NOT NULL COMMENT '处理帧数',
                inference_speed FLOAT NOT NULL COMMENT '推理速度',
                fps FLOAT NOT NULL COMMENT '帧率'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            self.cursor.execute(create_sql)
            self.connection.commit()
            print("表创建成功")
        except Exception as e:
            print(f"表创建失败: {e}")
    
    def insert_statistics(self, avg_speed, total_vehicles, current_vehicles, frame_count, inference_speed, fps):
        """
        插入统计数据（保存最新数据，达到时间间隔后写入）
        :param avg_speed: 平均车速
        :param total_vehicles: 累计车辆数
        :param current_vehicles: 当前车辆数
        :param frame_count: 处理帧数
        :param inference_speed: 推理速度
        :param fps: 帧率
        """
        timestamp = datetime.datetime.now()
        # 只保存最新的数据
        self.latest_data = {
            'timestamp': timestamp,
            'avg_speed': avg_speed,
            'total_vehicles': total_vehicles,
            'current_vehicles': current_vehicles,
            'frame_count': frame_count,
            'inference_speed': inference_speed,
            'fps': fps
        }
        
        # 检查是否需要写入数据
        return self.check_and_write()
    
    def check_and_write(self):
        """
        检查是否达到写入间隔，如果是则写入最新数据
        """
        current_time = time.time()
        if current_time - self.last_write_time >= self.write_interval:
            return self.write_latest_data()
        return True
    
    def write_latest_data(self):
        """
        写入最新的数据
        """
        if not self.latest_data:
            return True
        
        try:
            sql = """
            INSERT INTO traffic_statistics 
            (timestamp, avg_speed, total_vehicles, current_vehicles, frame_count, inference_speed, fps) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            # 写入最新数据
            self.cursor.execute(sql, (
                self.latest_data['timestamp'],
                self.latest_data['avg_speed'],
                self.latest_data['total_vehicles'],
                self.latest_data['current_vehicles'],
                self.latest_data['frame_count'],
                self.latest_data['inference_speed'],
                self.latest_data['fps']
            ))
            self.connection.commit()
            
            # 更新上次写入时间
            self.last_write_time = time.time()
            return True
        except Exception as e:
            print(f"数据写入失败: {e}")
            return False
    
    def get_statistics(self, limit=10):
        """
        获取最近的统计数据
        :param limit: 限制返回条数
        :return: 统计数据列表
        """
        try:
            sql = f"SELECT * FROM traffic_statistics ORDER BY timestamp DESC LIMIT {limit}"
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            print(f"获取数据失败: {e}")
            return []
    
    def close(self):
        """
        关闭数据库连接（先写入最新数据）
        """
        try:
            # 写入最新数据
            if self.latest_data:
                self.write_latest_data()
            
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            print("数据库连接已关闭")
        except Exception as e:
            print(f"关闭连接失败: {e}")


# 示例用法
if __name__ == "__main__":
    # 创建数据库工具实例
    db = DatabaseUtils()
    
    # 模拟插入数据
    db.insert_statistics(
        avg_speed=45.5,
        total_vehicles=120,
        current_vehicles=15,
        frame_count=1000,
        inference_speed=15.2,
        fps=25.5
    )
    
    # 获取最近的数据
    recent_data = db.get_statistics()
    print("最近的统计数据:")
    for data in recent_data:
        print(data)
    
    # 关闭连接
    db.close()