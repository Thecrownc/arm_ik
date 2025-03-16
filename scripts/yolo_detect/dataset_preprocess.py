import os
import shutil
import json
from copyreg import pickle
import pickle
from imaplib import IMAP4
import cv2
from humanfriendly.terminal import output
import numpy as np

def visualize_keypoints(image_path, keypoints, output_path):
    """
    在图像上可视化关键点并保存结果。

    :param image_path: str, 输入图像路径
    :param keypoints: dict, 关键点的字典，每个键对应一个列表，列表内是 (x, y) 坐标
    :param output_path: str, 保存输出图像的路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图像文件未找到: {image_path}")

    # 遍历字典中的关键点并在图像上绘制
    for loc_name, points in keypoints.items():
        for (x, y) in points:  # 每个点是 (x, y)
            # 绘制关键点（圆形）
            cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
            # 绘制索引名称
            cv2.putText(image, loc_name, (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)

    # 保存输出图像
    cv2.imwrite(output_path, image)
    print(f"可视化结果已保存到: {output_path}")

def draw_enlarged_bounding_box(image_path, keypoints, output_path, enlargement_factor=0.2):
    """
    根据关键点在原图上绘制放大后的边界框并保存结果。

    :param image_path: str, 输入图像路径
    :param keypoints: dict, 字典格式的关键点，键是名称，值是 [[x, y]] 列表
    :param output_path: str, 保存绘制结果的路径
    :param enlargement_factor: float, 边界框放大的比例（默认 20%）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图像文件未找到: {image_path}")

    # 提取所有关键点的坐标
    all_points = []
    for coordinates in keypoints.values():
        all_points.extend(coordinates)  # 将每个点的坐标加入列表

    # 转为 NumPy 数组以便计算
    all_points = np.array(all_points)  # [[x1, y1], [x2, y2], ...]

    # 计算边界框
    x_min = int(np.min(all_points[:, 0]))
    y_min = int(np.min(all_points[:, 1]))
    x_max = int(np.max(all_points[:, 0]))
    y_max = int(np.max(all_points[:, 1]))

    # 增加边界框长宽
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, int(x_min - width * enlargement_factor / 2))  # 左边界缩小，避免超出图像范围
    y_min = max(0, int(y_min - height * enlargement_factor / 2))  # 上边界缩小
    x_max = min(image.shape[1], int(x_max + width * enlargement_factor / 2))  # 右边界扩展
    y_max = min(image.shape[0], int(y_max + height * enlargement_factor / 2))  # 下边界扩展


    # 在原图上绘制边界框
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    # 保存结果图像
    cv2.imwrite(output_path, image)
    print(f"绘制放大边界框的图像已保存到: {output_path}")
    return x_min, y_min,x_max, y_max

def convert_to_coco_format(data, image_width, image_height):
    """
    将数据从自定义格式转换为 COCO 标注格式，坐标归一化到 [0, 1] 范围。

    :param data: dict, 包含 'bbox' 和多个关键点信息
                 例如：{'bbox': [260, 364, 312, 390],
                        'loc1': [[265.5, 379.4]],
                        'loc2': [[306.3, 367.1]]}
    :param image_width: int, 图像的宽度
    :param image_height: int, 图像的高度
    :return: list, COCO 格式的标注
    """
    # 提取 bbox 信息
    bbox = data['bbox']  # [x_min, y_min, x_max, y_max]

    # 转换 bbox 为 COCO 格式：[center_x, center_y, width, height]，归一化到 [0, 1]
    x_min, y_min, x_max, y_max = bbox
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    center_x = (x_min + x_max) / 2 / image_width
    center_y = (y_min + y_max) / 2 / image_height
    coco_bbox = [0, center_x, center_y, width, height]  # 类别 ID 默认为 0

    # 提取关键点信息
    keypoints = []
    for loc, coordinates in data.items():
        if loc.startswith("loc"):  # 判断是关键点数据
            x, y = coordinates[0]
            x_norm = x / image_width  # 归一化 x 坐标
            y_norm = y / image_height  # 归一化 y 坐标
            keypoints.extend([x_norm, y_norm, 2])  # COCO 格式：x, y, visibility=2（可见）

    # 如果关键点数不足（例如，COCO 通常需要固定数量的关键点），填充不可见关键点
    num_required_keypoints = 17  # 假设有 17 个关键点
    current_num_keypoints = len(keypoints) // 3
    if current_num_keypoints < num_required_keypoints:
        keypoints.extend([0.0, 0.0, 0] * (num_required_keypoints - current_num_keypoints))

    return coco_bbox + keypoints

if __name__ == "__main__":
    # 合并数据集
    # data_path = '/home/guozz/Datasets/button_datasets'
    # folds = ['high_rgb','left_rgb','right_rgb']
    # target_path = '/home/guozz/Datasets/button_datasets/combined'
    # cnt = 0
    # for fold in folds:
    #     fold_path = os.path.join(data_path,fold)
    #
    #     file_list = os.listdir(fold_path)
    #     nn_list = []
    #     for file in file_list:
    #         a= file.split('.')[0]
    #         if a not in nn_list:
    #             nn_list.append(a)
    #
    #     for nn in nn_list:
    #         cnt = cnt+1
    #         source_file = os.path.join(data_path,fold,nn+'.jpg')
    #         destination_folder = os.path.join(target_path,str(cnt).zfill(4)+'.jpg')
    #         shutil.copy(source_file, destination_folder)
    #
    #         source_file = os.path.join(data_path, fold, nn + '.json')
    #         destination_folder = os.path.join(target_path, str(cnt).zfill(4) + '.json')
    #         shutil.copy(source_file, destination_folder)
    #---------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------
    #可视化，检查标记是否正确
    # data_path = '/home/guozz/Datasets/button_datasets/combined'
    #
    # with open(os.path.join('/home/guozz/Datasets/button_datasets/label.pk',),'rb') as fr:
    #     label = pickle.load(fr)
    # for key in label:
    #     output_path = os.path.join('/home/guozz/Datasets/button_datasets/check',key)
    #     img_path = os.path.join(data_path,key)
    #     visualize_keypoints(img_path,label[key],output_path)
# ----------------------------------------------------------------------------------------
    #添加边界框
    # data_path = '/home/guozz/Datasets/button_datasets/combined'
    #
    # #修改label格式
    # with open(os.path.join('/home/guozz/Datasets/button_datasets/label.pk', ), 'rb') as fr:
    #     label = pickle.load(fr)
    #
    # for key in label:
    #     output_path = os.path.join('/home/guozz/Datasets/button_datasets/bbox',key)
    #     img_path = os.path.join(data_path,key)
    #     x_min, y_min,x_max, y_max = draw_enlarged_bounding_box(img_path,label[key],output_path)
    #     label[key]['bbox'] = [x_min, y_min,x_max, y_max]
    #
    # with open('/home/guozz/Datasets/button_datasets/label_v2.pk','wb') as fw:
    #     pickle.dump(label,fw)

    # data_path = '/home/guozz/Datasets/button_datasets/combined'
    # with open('/home/guozz/Datasets/button_datasets/label_v2.pk','rb') as fr:
    #     label = pickle.load(fr)
    #
    # target_path = "/home/guozz/Datasets/button_datasets/RGB_labels/"
    #
    # for key in label:
    #     output_path = os.path.join(target_path,key.split('.')[0]+'.txt')
    #     coco_format = convert_to_coco_format(label[key], image_width=640, image_height=480)
    #
    #     coco_text = " ".join(map(str, coco_format))
    #
    #     # 保存到指定路径
    #     with open(output_path, 'w') as file:
    #         file.write(coco_text + "\n")  # 每个样本一行
    #     print(f"COCO 格式数据已保存到: {output_path}")

    output_path = "/home/guozz/Datasets/button_datasets/train.txt"
    with open(output_path, 'w') as file:
        for i in range(1, 626 + 1):
            file.write(f"{i:04d}.jpg\n")  # 生成格式为四位数字，例如 0001.jpg
    print(f"文件列表已保存到: {output_path}")
