# -*- coding: utf-8 -*-
"""
将 YOLO 风格的 polygon 分割 txt 标注
转换为 完全符合 COCO 官方格式 的实例分割 JSON。

假设 txt 每一行格式为：
    class_id x1 y1 x2 y2 ... xn yn
其中：
    - class_id ∈ {0,1,2}，分别映射到：
        0 -> maturity1
        1 -> maturity2
        2 -> maturity3
    - x_i, y_i 均为 [0,1] 的相对坐标（相对整张图宽高）
"""

import os
import glob
import json
from PIL import Image
import numpy as np
from datetime import datetime


# -------------------- 一些几何计算函数 -------------------- #
def calculate_polygon_area(polygon: np.ndarray) -> float:
    """用 shoelace 公式计算多边形面积，polygon 形状为 (N, 2)。"""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def calculate_bounding_box(polygon: np.ndarray):
    """根据多边形顶点计算外接矩形 bbox，返回 [x_min, y_min, w, h]。"""
    x_min = float(np.min(polygon[:, 0]))
    y_min = float(np.min(polygon[:, 1]))
    x_max = float(np.max(polygon[:, 0]))
    y_max = float(np.max(polygon[:, 1]))
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    return [x_min, y_min, width, height]


# -------------------- 主转换函数 -------------------- #
def text_to_coco_segmentation(in_labels, in_images, out_json):
    """
    :param in_labels: txt 标签目录，例如 'labels/test'
    :param in_images: 图片目录，例如 'images/test'
    :param out_json:  输出的 COCO json 文件路径
    """

    # 1. COCO 顶层结构初始化
    coco = dict()
    coco["info"] = {
        "description": "Maturity Segmentation Dataset",
        "version": "1.0",
        "year": 2025,
        "contributor": "",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    coco["licenses"] = [
        {
            "id": 1,
            "name": "Unknown",
            "url": ""
        }
    ]
    coco["images"] = []
    coco["annotations"] = []

    # 2. 固定 3 个类别：maturity1/2/3
    coco["categories"] = [
        {"id": 1, "name": "maturity1", "supercategory": "maturity"},
        {"id": 2, "name": "maturity2", "supercategory": "maturity"},
        {"id": 3, "name": "maturity3", "supercategory": "maturity"}
    ]

    # annotation 与 image 的 id 计数器（COCO 习惯从 1 开始）
    ann_id = 1
    img_id = 1

    # 3. 读取所有 txt 文件（按文件名排序，保证顺序稳定）
    txt_files = sorted(glob.glob(os.path.join(in_labels, "*.txt")))

    if len(txt_files) == 0:
        print(f"[警告] 在 {in_labels} 中没有找到任何 .txt 文件")
        return

    for txt_path in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]

        # 默认图片名为 .jpg，你可以按需改成 .png / .jpeg 等
        img_name = base_name + ".jpg"
        img_path = os.path.join(in_images, img_name)

        if not os.path.exists(img_path):
            print(f"[警告] 找不到对应图片：{img_path}，跳过该 txt")
            continue

        # 3.1 读取图片尺寸
        img = Image.open(img_path)
        width, height = img.size

        # 3.2 填写 images 字段
        img_item = {
            "id": img_id,
            "file_name": img_name,
            "width": int(width),
            "height": int(height),
            "license": 1
        }
        coco["images"].append(img_item)

        # 3.3 读取该 txt 内每一个实例
        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                # 至少要有 class_id + 一对坐标
                continue

            # 第一个是类别 id（YOLO 写法，0/1/2）
            cls_id_yolo = int(float(parts[0]))
            if cls_id_yolo not in [0, 1, 2]:
                print(f"[警告] {txt_path} 中出现未知类别 id: {cls_id_yolo}，跳过该行")
                continue

            # 映射到 COCO 类别 id（1,2,3）
            category_id = cls_id_yolo + 1

            coord = [float(x) for x in parts[1:]]
            if len(coord) % 2 != 0:
                print(f"[警告] {txt_path} 中某行顶点数为奇数，跳过该行")
                continue

            # 归一化坐标 -> 像素坐标
            polygon = np.array(coord, dtype=float).reshape(-1, 2)
            polygon[:, 0] *= width
            polygon[:, 1] *= height

            # 计算 area / bbox
            area = calculate_polygon_area(polygon)
            bbox = calculate_bounding_box(polygon)

            # segmentation 按 COCO 要求：list[list[ x1,y1,... ]]
            segmentation = [polygon.flatten().astype(float).tolist()]

            ann_item = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0
            }
            coco["annotations"].append(ann_item)
            ann_id += 1

        print(f"{os.path.basename(txt_path)} 转换完成 (image_id={img_id})")
        img_id += 1

    # 4. 写入 JSON 文件
    os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 已保存 COCO 标注到：{out_json}")
    print(f"  - images 数量: {len(coco['images'])}")
    print(f"  - annotations 数量: {len(coco['annotations'])}")
    print(f"  - categories: maturity1/2/3")


if __name__ == "__main__":
    text_to_coco_segmentation(
        in_labels=r"C:\Users\20433\Desktop\Hickory_Data_new\dataests_final2\labels\test",  # 存 txt 的目录
        in_images=r"C:\Users\20433\Desktop\Hickory_Data_new\dataests_final2\images\test",  # 存图片的目录
        out_json=r"C:\Users\20433\Desktop\Hickory_Data_new\dataests_final2\labels\instances_test.json"
    )
