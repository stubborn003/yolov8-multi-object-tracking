from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 加载模型（建议使用更大的预训练模型）
    model = YOLO('yolov8n.pt')  # 改用中等尺寸模型提升性能

    # 优化后的训练配置
    results = model.train(
        data="vehicles.yaml",
        # resume=True,
        epochs=80,              # 增加训练轮次
        patience=20,            # 早停耐心值
        imgsz=640,              # 增大图像尺寸
        batch=16,               # 减小batch size以获得更好梯度估计
        device='0',
        optimizer='AdamW',      # 保持
        lr0=0.001,             # 降低初始学习率
        lrf=0.01,              # 最终学习率 = lr0 * lrf
        weight_decay=0.0005,    # 保持
        warmup_epochs=3,        # 学习率预热
        warmup_momentum=0.8,    # 预热期动量
        warmup_bias_lr=0.1,     # 偏置项学习率
        box=7.5,                # 增加box loss权重
        cls=0.5,                # 分类损失权重
        dfl=1.5,                # 分布焦点损失权重
        close_mosaic=10,        # 保持
        hsv_h=0.015,            # 色相增强
        hsv_s=0.7,              # 饱和度增强
        hsv_v=0.4,              # 明度增强
        degrees=10.0,           # 增加旋转角度
        translate=0.1,          # 平移增强
        scale=0.5,              # 缩放增强
        shear=2.0,              # 剪切增强
        perspective=0.0005,     # 透视变换
        fliplr=0.5,             # 增加水平翻转概率
        mosaic=1.0,             # 使用mosaic增强
        mixup=0.1,              # 轻微mixup增强
        copy_paste=0.1,         # 小物体复制粘贴增强
        erasing=0.4,            # 随机擦除
        crop_fraction=0.9,      # 图像裁剪比例
        overlap_mask=True,      # 训练时计算mask重叠
        single_cls=False,       # 多类别训练
        workers=4,              # 增加数据加载线程
        pretrained=True,        # 使用预训练权重
        seed=42,                # 固定随机种子
        deterministic=True      # 确定性训练
    )