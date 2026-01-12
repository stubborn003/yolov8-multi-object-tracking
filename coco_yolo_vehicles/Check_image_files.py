import os
from PIL import Image
from tqdm import tqdm


def find_and_remove_corrupt_images(data_dir):
    corrupt_files = []

    for root, _, files in os.walk(os.path.join(data_dir, 'images')):
        for file in tqdm(files, desc="Checking images"):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    # 验证图片完整性
                    with Image.open(img_path) as img:
                        img.verify()

                    # 尝试读取图片数据
                    with Image.open(img_path) as img:
                        img.load()  # 强制加载所有数据
                except (IOError, OSError, Image.DecompressionBombError) as e:
                    corrupt_files.append(img_path)

    # 删除损坏文件及其对应标签
    removed = []
    for img_path in corrupt_files:
        # 删除图片文件
        os.remove(img_path)

        # 查找并删除对应标签
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt').replace(
            '.jpeg', '.txt')
        if os.path.exists(label_path):
            os.remove(label_path)
            removed.append((img_path, label_path))
        else:
            removed.append((img_path, None))

    return removed


# 使用示例
corrupt_files = find_and_remove_corrupt_images("E:/coco/coco_yolo_vehicles")
print(f"\nFound and removed {len(corrupt_files)} corrupt files:")
for img, lbl in corrupt_files:
    print(f"- Image: {img}")
    if lbl: print(f"  Label: {lbl}")