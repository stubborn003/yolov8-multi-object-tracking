import cv2
import numpy as np
from collections import defaultdict
import time
from voice_alert import play_voice_alert
import os


def calculate_iou(box1, box2):
    """
    计算两个边界框之间的交并比（Intersection over Union, IoU）。
    :param box1: 第一个边界框，格式为 (x, y, w, h)，分别表示左上角坐标和宽高
    :param box2: 第二个边界框，格式为 (x, y, w, h)
    :return: 两个边界框的交并比，取值范围为 [0, 1]
    """
    # 计算交集区域的左上角坐标
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    # 计算交集区域的右下角坐标
    x2_int = min(box1[0] + box1[2], box2[0] + box2[2])
    y2_int = min(box1[1] + box1[3], box2[1] + box2[3])

    # 计算交集区域的面积
    intersection_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    # 计算两个边界框各自的面积
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # 计算交并比，避免除零错误
    return intersection_area / float(box1_area + box2_area - intersection_area) if (box1_area + box2_area) > 0 else 0


# 定义需要跟踪的目标类别列表
OBJ_LIST = [0, 1, 2, 3, 4]
# 定义需要发出警报的目标类别列表
ALERT_OBJ_LIST = [0, 2]

def initialize_tracking(video_path, result_path, warning_folder):
    """
    初始化跟踪所需的资源和变量。
    :param video_path: 输入视频文件的路径
    :param result_path: 输出处理后视频文件的路径
    :param warning_folder: 用于保存警告帧图像的文件夹路径
    :return: 包含初始化后各变量的元组
    """
    # 初始化视频写入器为None，后续根据需要创建
    videowriter = None
    # 用于记录每个跟踪目标的历史轨迹，使用defaultdict方便添加新的目标
    track_history = defaultdict(lambda: [])
    # 记录进入特定区域的目标数量
    count_passed = 0
    # 记录离开特定区域的目标数量
    count_exited = 0
    # 存储已经进入特定区域的目标的 ID
    entered_ids = set()
    # 存储每个目标进入警告区域的时间
    entry_time = {}
    # 存储已经发出过警告的目标的 ID
    warned_ids = set()

    # 定义特定区域的多边形顶点坐标
    polygon_points = np.array([[0, 500], [0, 670], [1918, 630], [1918, 463]], np.int32)
    # 定义警告区域的多边形顶点坐标
    polygon_points1 = np.array([[120, 470], [1731, 428], [1918, 133], [1918, 0], [1350, 0]], np.int32)

    return (videowriter, track_history, entered_ids, entry_time, warned_ids, count_passed,
            count_exited, polygon_points, polygon_points1, 30, 1920, 1080)


def nms(boxes, scores, iou_threshold=0.3):
    """
    非极大值抑制（Non-Maximum Suppression, NMS）算法，用于去除重叠的检测框。
    :param boxes: 检测框列表，格式为 (x, y, w, h)
    :param scores: 每个检测框对应的置信度分数
    :param iou_threshold: 交并比阈值，当两个检测框的 IoU 大于该阈值时，会抑制其中一个
    :return: 经过 NMS 处理后保留的检测框的索引列表
    """
    # 如果检测框列表为空，直接返回空列表
    if len(boxes) == 0:
        return []
    # 将检测框转换为 (x1, y1, x2, y2) 格式，方便后续计算
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # 计算每个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照置信度分数降序排序，获取排序后的索引
    order = scores.argsort()[::-1]

    # 用于存储最终保留的检测框索引
    keep = []
    # 循环处理排序后的检测框索引列表
    while order.size > 0:
        # 选取置信度最高的检测框的索引
        i = order[0]
        keep.append(i)
        # 计算其余检测框与当前选取框的交集区域的左上角和右下角坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算交集区域的宽度和高度
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 计算交集区域的面积
        inter = w * h
        # 计算其余检测框与当前选取框的交并比
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到交并比小于等于阈值的检测框的索引
        inds = np.where(ovr <= iou_threshold)[0]
        # 更新排序后的索引列表，去除被抑制的检测框
        order = order[inds + 1]

    return keep


def process_frame(frame, model, videowriter, track_history, entered_ids, entry_time,
                  warned_ids, count_passed, count_exited, polygon_points, polygon_points1,
                  play_voice_alert, warning_folder, warning_display=None):
    """
    处理视频的每一帧，进行目标跟踪和预警处理。
    """
    # 使用模块级定义的目标类别列表

    # 记录已经显示过警告信息的ID
    displayed_warning_ids = set()

    # 使用 YOLO 模型对当前帧进行目标跟踪，只跟踪指定类别的目标，并设置置信度阈值
    results = model.track(frame, persist=True, classes=OBJ_LIST, conf=0.5)

    # 如果有检测结果，绘制检测框；否则使用原始帧图像
    a_frame = results[0].plot(line_width=2) if results[0] is not None else frame

    # 创建一个与帧图像相同大小的掩码，用于绘制特定区域
    mask = np.zeros_like(frame)
    # 在掩码上填充特定区域
    cv2.fillPoly(mask, [polygon_points], (0, 255, 255))
    # 将掩码与帧图像叠加，使特定区域半透明显示
    a_frame = cv2.addWeighted(a_frame, 1, mask, 0.1, 0)

    # 存储当前帧的警告信息
    current_warnings = []

    # 如果检测结果不为空且包含边界框和ID信息
    if results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
        # 获取检测到的目标的边界框信息
        boxes = results[0].boxes.xywh.cpu()
        # 获取检测到的目标的 ID
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # 获取检测到的目标的类别
        track_classes = results[0].boxes.cls.int().cpu().tolist()
        # 获取每个检测框的置信度分数
        scores = results[0].boxes.conf.cpu().numpy()

        # 将检测框信息转换为 numpy 数组
        boxes = boxes.cpu().numpy()

        # 过滤掉过大的检测框
        valid_indices = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            frame_area = frame.shape[0] * frame.shape[1]
            box_area = w * h
            # 过滤条件：检测框面积不超过视频面积的五分之一
            if box_area < frame_area / 5:
                valid_indices.append(i)

        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        track_ids = [track_ids[i] for i in valid_indices]
        track_classes = [track_classes[i] for i in valid_indices]


        keep_indices = nms(boxes, scores, iou_threshold=0.4)

        filtered_boxes = boxes[keep_indices]
        filtered_ids = [track_ids[i] for i in keep_indices]
        filtered_classes = [track_classes[i] for i in keep_indices]

        # 确保每个框只包含一个主要目标
        final_boxes = []
        final_ids = []
        final_classes = []

        for box, track_id, track_class in zip(filtered_boxes, filtered_ids, filtered_classes):
            x, y, w, h = box
            aspect_ratio = w / h

            # 过滤掉不合理的宽高比 (0.2-5.0是合理范围)
            if 0.2 < aspect_ratio < 5.0:
                final_boxes.append(box)
                final_ids.append(track_id)
                final_classes.append(track_class)

        # 使用最终过滤后的结果进行后续处理
        for box, track_id, track_class in zip(final_boxes, final_ids, final_classes):
            x, y, w, h = box
            # 计算边界框的中心点坐标
            center = np.array([x + w / 2, y + h / 2], dtype=np.float32)

            # 获取该目标的跟踪历史
            track = track_history[track_id]
            # 将当前中心点添加到跟踪历史中
            track.append((float(x), float(y)))
            # 只保留最近的 30 个跟踪点，避免内存占用过大
            if len(track) > 30:
                track.pop(0)

            # 将跟踪点转换为适合 OpenCV 绘制的格式
            points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)
            # 在帧图像上绘制目标的跟踪轨迹
            cv2.polylines(a_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)

            # 如果目标还未进入特定区域且当前位于特定区域内
            if track_id not in entered_ids and cv2.pointPolygonTest(polygon_points, center, False) >= 0:
                # 进入特定区域的目标数量加 1
                count_passed += 1
                # 将该目标的 ID 添加到已进入集合中
                entered_ids.add(track_id)

            # 如果目标位于警告区域内
            if cv2.pointPolygonTest(polygon_points1, center, False) >= 0:
                if track_class in ALERT_OBJ_LIST:
                    if track_id not in entry_time:
                        # 记录目标进入警告区域的时间
                        entry_time[track_id] = time.time()
                    else:
                        # 如果目标在警告区域内停留超过 2 秒
                        if time.time() - entry_time[track_id] > 2:
                            # 创建一个与帧图像相同大小的掩码，用于绘制警告区域
                            mask1 = np.zeros_like(frame)
                            # 在掩码上填充警告区域
                            cv2.fillPoly(mask1, [polygon_points1], (0, 0, 255))
                            # 将掩码与帧图像叠加，使警告区域半透明显示
                            a_frame = cv2.addWeighted(a_frame, 1, mask1, 0.2, 0)
                            # 在帧图像上显示警告信息
                            cv2.putText(a_frame, f'warn: ID {track_id}', (int(center[0]), int(center[1] - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                            # 只有当ID不在已显示集合中时才记录警告信息
                            if track_id not in displayed_warning_ids:
                                warning_msg = f"警告：物体 ID {track_id} ({track_class}) 进入了警告区域！"
                                print(warning_msg)
                                current_warnings.append(warning_msg)
                                displayed_warning_ids.add(track_id)  # 标记该ID已显示警告

                            if track_id not in warned_ids:
                                # 保存当前帧图像作为警告帧
                                cv2.imwrite(os.path.join(warning_folder, f"warning_frame_{track_id}.jpg"), frame)
                                # 启动一个新线程播放语音警报
                                import threading
                                voice_thread = threading.Thread(target=play_voice_alert)
                                voice_thread.start()
                                # 将该目标的 ID 添加到已警告集合中
                                warned_ids.add(track_id)
                else:
                    # 如果目标类别不在需要警报的列表中，移除其进入警告区域的时间记录
                    if track_id in entry_time:
                        del entry_time[track_id]
            # 如果目标已经进入特定区域且当前不在特定区域内
            elif track_id in entered_ids and cv2.pointPolygonTest(polygon_points, center, True) < 0:
                # 离开特定区域的目标数量加 1
                count_exited += 1
                # 从已进入集合中移除该目标的 ID
                entered_ids.remove(track_id)
                # 如果该目标有进入警告区域的时间记录，移除该记录
                if track_id in entry_time:
                    del entry_time[track_id]

    # 如果提供了警告显示控件，则更新显示
    if warning_display is not None and current_warnings:
        # 获取当前时间
        current_time = time.strftime("%H:%M:%S")
        # 在每个警告前添加时间戳
        for warning in current_warnings:
            warning_display.append(f"[{current_time}] {warning}")

    return (a_frame, count_passed, count_exited, entered_ids, entry_time, warned_ids,
            track_history)
