import cv2
import numpy as np
from typing import Tuple, List, Optional
import os
import time

# 配置参数
CONFIG = {
    'min_area': 3000,  # 最小面积阈值
    'rect_ratio': 0.85,  # 矩形度阈值
    'resize_width': 640,  # 处理图像的宽度
    'run_mode':'debug',
    'colors': {
        'red': {
            'ranges': [
                {'lower': np.array([0, 120, 50]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([170, 120, 50]), 'upper': np.array([180, 255, 255])}
            ],
            'bgr': (0, 0, 255)
        },
        'green': {
            'ranges': [
                {'lower': np.array([35, 120, 50]), 'upper': np.array([85, 255, 255])}
            ],
            'bgr': (0, 255, 0)
        }
    },
    'visualization_dir': 'visualization_steps'  # 可视化结果保存目录
}

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_debug_image(name: str, image: np.ndarray, frame_count: int):
    """保存调试图像"""
    ensure_dir(CONFIG['visualization_dir'])
    filename = os.path.join(CONFIG['visualization_dir'], f'frame_{frame_count:04d}_{name}.jpg')
    cv2.imwrite(filename, image)

def preprocess_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """图像预处理"""
    try:
        # 调整图像大小以提高处理速度
        h, w = img.shape[:2]
        scale = CONFIG['resize_width'] / w
        new_w = CONFIG['resize_width']
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # 使用均值滤波代替双边滤波，速度更快
        img_blur = cv2.medianBlur(img_resized, 5)

        return img_blur, scale
    except Exception as e:
        print(f"图像预处理失败: {str(e)}")
        return img, 1.0

def create_color_mask(hsv_img: np.ndarray, color_ranges: List[dict]) -> np.ndarray:
    """创建颜色掩码"""
    masks = []
    for range_dict in color_ranges:
        mask = cv2.inRange(hsv_img, range_dict['lower'], range_dict['upper'])
        masks.append(mask)

    if len(masks) > 1:
        return cv2.bitwise_or(*masks)
    return masks[0]

def process_contours(contours: List[np.ndarray], color_name: str, result_image: np.ndarray, 
                    scale: float) -> List[Tuple[int, int]]:
    """处理轮廓并返回中心点列表"""
    centers = []
    color_config = CONFIG['colors'][color_name]
    bgr_color = color_config['bgr']

    # 创建轮廓绘制的副本
    contour_image = result_image.copy()

    img_height, img_width = result_image.shape[:2]
    img_center_x = img_width // 2
    img_center_y = img_height // 2

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > CONFIG['min_area'] * (scale ** 2):  # 根据缩放调整面积阈值
            # 计算周长和近似多边形
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # 只处理矩形（顶点数为4的形状）
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                # 将坐标转换回原始尺寸
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w / scale)
                h_orig = int(h / scale)

                # 计算以图像中心为原点的坐标
                cx = x_orig + w_orig // 2 - img_center_x
                cy = img_center_y - (y_orig + h_orig // 2)  # 注意y轴方向翻转
                centers.append((cx, cy))

                # 在原始图像上绘制
                cv2.rectangle(contour_image, (x_orig, y_orig), 
                            (x_orig + w_orig, y_orig + h_orig), bgr_color, 2)

                # 绘制中心点和坐标文本（使用屏幕坐标系进行绘制）
                screen_cx = x_orig + w_orig // 2
                screen_cy = y_orig + h_orig // 2
                cv2.circle(contour_image, (screen_cx, screen_cy), 5, bgr_color, -1)
                cv2.putText(contour_image, f"{color_name}: ({cx}, {cy})", 
                          (screen_cx - 20, screen_cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, bgr_color, 2)

    result_image[:] = contour_image[:]
    return centers

def detect_color_blocks(img: np.ndarray, frame_count: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """检测颜色块并返回中心点"""
    if img is None:
        raise ValueError("输入图像为空")

    try:
        # 保存原始图像
        save_debug_image('original', img, frame_count)

        # 图像预处理
        processed_img, scale = preprocess_image(img)
        save_debug_image('preprocessed', processed_img, frame_count)

        result_image = img.copy()

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        save_debug_image('hsv', hsv, frame_count)

        centers_dict = {}
        for color_name, color_config in CONFIG['colors'].items():
            # 创建颜色掩码
            mask = create_color_mask(hsv, color_config['ranges'])
            save_debug_image(f'mask_{color_name}', mask, frame_count)

            # 形态学操作改善掩码质量
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            save_debug_image(f'mask_{color_name}_morphology', mask, frame_count)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 处理轮廓
            centers = process_contours(contours, color_name, result_image, scale)
            centers_dict[color_name] = centers

        # 保存最终结果
        save_debug_image('final_result', result_image, frame_count)

        # 显示结果
        cv2.imshow("detected_colors", result_image)
        cv2.waitKey(1)

        return centers_dict['red'], centers_dict['green']

    except Exception as e:
        print(f"颜色检测失败: {str(e)}")
        return [], []

def video_detect(video_path: str):
    """视频检测主函数"""


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频读取结束")
                break

            detect_color_blocks(frame, frame_count)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"视频处理出错: {str(e)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "../data/aim_data/aim_1.avi"

    if CONFIG['run_mode'] == 'debug':
    # 清空并创建可视化目录
        if os.path.exists(CONFIG['visualization_dir']):
            for file in os.listdir(CONFIG['visualization_dir']):
                os.remove(os.path.join(CONFIG['visualization_dir'], file))
        ensure_dir(CONFIG['visualization_dir'])

        video_detect(video_path)

    elif CONFIG['run_mode'] == 'release':
        
        video_detect(video_path)
