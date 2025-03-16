#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入必要的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # 数值计算库
from typing import Tuple, List, Optional  # 类型提示
import os  # 操作系统接口
import time  # 时间相关功能
import rospy  # ROS Python客户端库
from sensor_msgs.msg import Image  # ROS图像消息类型
from std_msgs.msg import String  # ROS字符串消息类型
from cv_bridge import CvBridge  # ROS和OpenCV之间的图像转换工具

# 全局配置参数
CONFIG = {
    'min_area': 3000,  # 检测目标的最小面积阈值
    'rect_ratio': 0.85,  # 矩形度阈值，用于判断形状是否为矩形
    'resize_width': 640,  # 处理图像的宽度，用于调整图像大小
    'run_mode':'debug',  # 运行模式：debug或release
    'colors': {  # 颜色配置
        'red': {  # 红色配置
            'ranges': [  # HSV颜色范围
                {'lower': np.array([0, 120, 50]), 'upper': np.array([10, 255, 255])},  # 红色范围1
                {'lower': np.array([170, 120, 50]), 'upper': np.array([180, 255, 255])}  # 红色范围2
            ],
            'bgr': (0, 0, 255)  # BGR颜色值，用于显示
        },
        'green': {  # 绿色配置
            'ranges': [  # HSV颜色范围
                {'lower': np.array([35, 120, 50]), 'upper': np.array([85, 255, 255])}  # 绿色范围
            ],
            'bgr': (0, 255, 0)  # BGR颜色值，用于显示
        }
    },
    'visualization_dir': 'visualization_steps'  # 调试图像保存目录
}

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):  # 检查目录是否存在
        os.makedirs(directory)  # 创建目录

def save_debug_image(name: str, image: np.ndarray, frame_count: int):
    """保存调试图像到指定目录"""
    ensure_dir(CONFIG['visualization_dir'])  # 确保保存目录存在
    filename = os.path.join(CONFIG['visualization_dir'], f'frame_{frame_count:04d}_{name}.jpg')  # 生成文件名
    cv2.imwrite(filename, image)  # 保存图像

def preprocess_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """图像预处理函数"""
    try:
        # 获取图像尺寸
        h, w = img.shape[:2]  # 获取图像高度和宽度
        scale = CONFIG['resize_width'] / w  # 计算缩放比例
        new_w = CONFIG['resize_width']  # 新的宽度
        new_h = int(h * scale)  # 新的高度，保持宽高比
        img_resized = cv2.resize(img, (new_w, new_h))  # 调整图像大小

        # 使用中值滤波去除噪点
        img_blur = cv2.medianBlur(img_resized, 5)  # 5x5中值滤波

        return img_blur, scale  # 返回处理后的图像和缩放比例
    except Exception as e:
        print(f"图像预处理失败: {str(e)}")  # 错误处理
        return img, 1.0  # 返回原始图像和默认缩放比例

def create_color_mask(hsv_img: np.ndarray, color_ranges: List[dict]) -> np.ndarray:
    """创建颜色掩码"""
    masks = []  # 存储不同范围的掩码
    for range_dict in color_ranges:  # 遍历每个颜色范围
        mask = cv2.inRange(hsv_img, range_dict['lower'], range_dict['upper'])  # 创建掩码
        masks.append(mask)  # 添加到掩码列表

    if len(masks) > 1:  # 如果有多个掩码
        return cv2.bitwise_or(*masks)  # 合并所有掩码
    return masks[0]  # 否则返回单个掩码

def process_contours(contours: List[np.ndarray], color_name: str, result_image: np.ndarray, 
                    scale: float) -> List[Tuple[int, int]]:
    """处理轮廓并返回中心点列表"""
    centers = []  # 存储检测到的中心点
    color_config = CONFIG['colors'][color_name]  # 获取颜色配置
    bgr_color = color_config['bgr']  # 获取BGR颜色值

    # 创建结果图像的副本
    contour_image = result_image.copy()  # 复制原始图像

    # 获取图像尺寸和中心点
    img_height, img_width = result_image.shape[:2]  # 获取图像尺寸
    img_center_x = img_width // 2  # 计算图像中心x坐标
    img_center_y = img_height // 2  # 计算图像中心y坐标

    for contour in contours:  # 遍历每个轮廓
        area = cv2.contourArea(contour)  # 计算轮廓面积
        if area > CONFIG['min_area'] * (scale ** 2):  # 根据缩放调整面积阈值
            # 计算轮廓周长和近似多边形
            perimeter = cv2.arcLength(contour, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)  # 多边形近似

            # 只处理矩形（4个顶点）
            if len(approx) == 4:  # 检查是否为4边形
                x, y, w, h = cv2.boundingRect(contour)  # 获取边界矩形
                # 将坐标转换回原始尺寸
                x_orig = int(x / scale)  # 转换x坐标
                y_orig = int(y / scale)  # 转换y坐标
                w_orig = int(w / scale)  # 转换宽度
                h_orig = int(h / scale)  # 转换高度

                # 计算以图像中心为原点的坐标
                cx = x_orig + w_orig // 2 - img_center_x  # 计算中心x坐标
                cy = img_center_y - (y_orig + h_orig // 2)  # 计算中心y坐标（注意y轴方向）
                centers.append((cx, cy))  # 添加中心点

                # 在图像上绘制矩形
                cv2.rectangle(contour_image, (x_orig, y_orig), 
                            (x_orig + w_orig, y_orig + h_orig), bgr_color, 2)

                # 绘制中心点和坐标文本
                screen_cx = x_orig + w_orig // 2  # 计算屏幕坐标系中的x坐标
                screen_cy = y_orig + h_orig // 2  # 计算屏幕坐标系中的y坐标
                cv2.circle(contour_image, (screen_cx, screen_cy), 5, bgr_color, -1)  # 绘制中心点
                cv2.putText(contour_image, f"{color_name}: ({cx}, {cy})", 
                          (screen_cx - 20, screen_cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, bgr_color, 2)  # 绘制坐标文本

    result_image[:] = contour_image[:]  # 更新结果图像
    return centers  # 返回中心点列表

def detect_color_blocks(img: np.ndarray, frame_count: int) -> Dict[str, Any]:
    """检测颜色块并返回结果"""
    if img is None:  # 检查输入图像
        raise ValueError("输入图像为空")

    try:
        # 图像预处理
        processed_img, scale = preprocess_image(img)  # 预处理图像

        result_image = img.copy()  # 创建结果图像副本

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)  # BGR转HSV

        result_json = {"detections": []}  # 初始化结果JSON
        
        for color_name, color_config in CONFIG['colors'].items():  # 遍历每种颜色
            # 创建颜色掩码
            mask = create_color_mask(hsv, color_config['ranges'])  # 创建颜色掩码

            # 形态学操作改善掩码质量
            kernel = np.ones((5, 5), np.uint8)  # 创建形态学操作核
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 处理轮廓
            centers = process_contours(contours, color_name, result_image, scale)
            
            # 构建JSON结果
            if centers:  # 如果检测到目标
                for cx, cy in centers:  # 遍历每个中心点
                    result_json["detections"].append({  # 添加检测结果
                        "color": color_name,
                        "position": {"x": cx, "y": cy}
                    })

        # 显示结果
        cv2.imshow("detected_colors", result_image)  # 显示结果图像
        cv2.waitKey(1)  # 等待1ms

        return result_json  # 返回检测结果

    except Exception as e:
        print(f"颜色检测失败: {str(e)}")  # 错误处理
        return {"detections": []}  # 返回空结果

class ColorDetector:
    """颜色检测器类"""
    def __init__(self):
        rospy.init_node('color_detector', anonymous=True)  # 初始化ROS节点
        self.bridge = CvBridge()  # 创建图像转换桥接器
        self.frame_count = 0  # 初始化帧计数器
        
        # 创建ROS发布者和订阅者
        self.result_pub = rospy.Publisher('/detect_result', String, queue_size=10)  # 创建结果发布者
        self.image_sub = rospy.Subscriber('/camera_f/color/image_raw', Image, self.image_callback)  # 创建图像订阅者
        
        print("Color detector node initialized")  # 打印初始化信息

    def image_callback(self, data):
        """图像回调函数"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # 转换图像格式
            
            # 检测颜色块
            result_strings = detect_color_blocks(cv_image, self.frame_count)  # 检测颜色
            
            # 发布检测结果
            if result_strings:  # 如果有检测结果
                result_msg = String()  # 创建消息
                result_msg.data = result_strings  # 设置消息数据
                self.result_pub.publish(result_msg)  # 发布消息
            
            self.frame_count += 1  # 更新帧计数
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")  # 错误处理

    def run(self):
        """运行ROS节点"""
        rospy.spin()  # 保持节点运行

if __name__ == "__main__":
    try:
        detector = ColorDetector()  # 创建检测器实例
        detector.run()  # 运行检测器
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()  # 关闭所有窗口 