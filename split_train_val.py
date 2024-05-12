# src/data_processing/split_train_val.py
import os
import random
import shutil


def split_train_val(train_dir, val_dir, val_ratio=0.2):
    # 创建验证集目录
    os.makedirs(val_dir, exist_ok=True)
    classes = os.listdir(train_dir)

    for cls in classes:
        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        os.makedirs(val_cls_dir, exist_ok=True)

        images = os.listdir(train_cls_dir)
        val_count = int(len(images) * val_ratio)
        val_images = random.sample(images, val_count)

        for img in val_images:
            shutil.move(os.path.join(train_cls_dir, img), os.path.join(val_cls_dir, img))


if __name__ == "__main__":
    train_dir = r"C:\Users\IssacY\Desktop\GarbageClassificationProject\data\raw\TrashNet\train"
    val_dir = r"C:\Users\IssacY\Desktop\GarbageClassificationProject\data\raw\TrashNet\val"
    split_train_val(train_dir, val_dir, val_ratio=0.2)
