import os
import cv2
import numpy as np
import pandas as pd

def calculate_iou(image1_path, image2_path):
    # 读取两张二值化图像
    binary_image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    binary_image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # 确保两个图像的大小相同
    if binary_image1.shape != binary_image2.shape:
        raise ValueError("两张图像的大小不一致")

    # 计算交集图像（两个图像的逐元素与操作）
    intersection_image = np.logical_and(binary_image1, binary_image2)

    # 计算并集图像（两个图像的逐元素或操作）
    union_image = np.logical_or(binary_image1, binary_image2)

    # 计算交集和并集的非零像素数量
    intersection_area = np.sum(intersection_image > 0)
    union_area = np.sum(union_image > 0)

    # 计算交集与并集的比值
    intersection_over_union = intersection_area / union_area

    return intersection_area, union_area, intersection_over_union

def process_images(folder1, folder2, output_file):
    # 获取文件夹中的所有图像文件
    images_folder1 = [f for f in os.listdir(folder1) if f.endswith('.png')]
    images_folder2 = [f for f in os.listdir(folder2) if f.endswith('.png')]

    # 创建一个空的DataFrame用于保存结果
    df = pd.DataFrame(columns=['ImageName', 'Intersection', 'Union', 'IoU'])

    # 遍历文件夹中的图像并计算IoU
    for image_name in images_folder1:
        image1_path = os.path.join(folder1, image_name)
        image2_path = os.path.join(folder2, image_name)

        if os.path.exists(image2_path):
            intersection, union, iou = calculate_iou(image1_path, image2_path)
            df = df.append({'ImageName': image_name, 'Intersection': intersection, 'Union': union, 'IoU': iou}, ignore_index=True)

    # 将结果保存到Excel文件中
    df.to_excel(output_file, index=False)

# 用法示例
folder1_path = "F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/1111/1zs"
folder2_path = "F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/1111/5yc"
output_excel_path = "F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/1111/result.xlsx"

process_images(folder1_path, folder2_path, output_excel_path)
