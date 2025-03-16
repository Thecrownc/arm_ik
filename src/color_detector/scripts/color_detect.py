#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
# from cv_bridge import CvBridge
import json
from functools import lru_cache
import time
import sys

# 配置参数 - 预先计算可重用的值
CONFIG = {
    'min_area': 3000,  # 最小面积阈值
    'rect_ratio': 0.85,  # 矩形度阈值
    'resize_width': 640,  # 处理图像的宽度
    'blur_kernel_size': 5,  # 中值滤波核大小
    'morph_kernel': np.ones((5, 5), np.uint8),  # 预计算形态学操作的内核
    'image_total_area': 640*480,
    'fps_calc_interval': 30,  # 每处理这么多帧计算一次FPS
    'colors': {
        'red': {
            'ranges': [
                {'lower': np.array([0, 120, 50], dtype=np.uint8), 'upper': np.array([10, 255, 255], dtype=np.uint8)},
                {'lower': np.array([170, 120, 50], dtype=np.uint8), 'upper': np.array([180, 255, 255], dtype=np.uint8)}
            ],
            'bgr': (0, 0, 255)
        },
        'green': {
            'ranges': [
                {'lower': np.array([35, 120, 50], dtype=np.uint8), 'upper': np.array([85, 255, 255], dtype=np.uint8)}
            ],
            'bgr': (0, 255, 0)
        }
    }
}

# 预先创建常用字体
FONT = cv2.FONT_HERSHEY_SIMPLEX
def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """图像预处理，优化版本"""
    # 更新总面积信息 - 仅在必要时计算一次
    h, w = img.shape[:2]
    CONFIG['image_total_area'] = w * h
    
    # 使用预定义的内核大小进行均值滤波
    return cv2.medianBlur(img, CONFIG['blur_kernel_size'])


def create_color_mask(hsv_img: np.ndarray, color_ranges: List[dict]) -> np.ndarray:
    """创建颜色掩码，优化版本"""
    # 仅处理一个颜色范围的常见情况
    if len(color_ranges) == 1:
        return cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
    
    # 处理多颜色范围的情况
    mask = cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
    for range_dict in color_ranges[1:]:
        mask |= cv2.inRange(hsv_img, range_dict['lower'], range_dict['upper'])
    return mask


@lru_cache(maxsize=1024)
def calculate_area_percentage(area: int) -> float:
    """计算色块面积占总图像的百分比，使用缓存提高性能"""
    total_area = CONFIG['image_total_area']
    percentage = (area / total_area) * 100
    return round(percentage, 2)


def process_contours(contours: List[np.ndarray], color_name: str, result_image: np.ndarray) -> Tuple[List[Tuple[int, int]], Optional[float]]:
    """处理轮廓并返回中心点列表和面积"""
    if not contours:  # 快速检查是否有轮廓
        return [], None
        
    centers = []
    color_config = CONFIG['colors'][color_name]
    bgr_color = color_config['bgr']

    # 无需创建副本，直接在结果图像上绘制
    img_height, img_width = result_image.shape[:2]
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    max_area = 0
    max_contour_info = None
    
    # 查找最大轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > CONFIG['min_area'] and area > max_area:
            # 仅对超过阈值的大轮廓进行多边形逼近
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            # 只处理矩形（顶点数为4的形状）
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                max_area = area
                max_contour_info = (contour, area, x, y, w, h)
    
    # 如果找到符合条件的轮廓
    if max_contour_info:
        contour, area, x, y, w, h = max_contour_info
        
        # 计算面积百分比
        area_percentage = calculate_area_percentage(int(area))
        
        # 计算以图像中心为原点的坐标
        cx = x + w / 2 - img_center_x
        cy = img_center_y - (y + h / 2)  # 注意y轴方向翻转
        centers.append((cx, cy))
        
        # 在原始图像上绘制轮廓 - 比绘制矩形更精确且资源消耗更少
        cv2.drawContours(result_image, [contour], -1, bgr_color, 2)
        
        # 绘制中心点
        screen_cx = x + w / 2
        screen_cy = y + h / 2
        cv2.circle(result_image, (screen_cx, screen_cy), 5, bgr_color, -1)
        
        # 绘制颜色和坐标信息 - 仅绘制必要的文本
        cv2.putText(result_image, f"{color_name}: ({cx}, {cy})", 
                  (screen_cx - 20, screen_cy - 20), FONT, 0.5, bgr_color, 2)
        
        # 添加面积百分比
        cv2.putText(result_image, f"percentage:{area_percentage}", 
                  (10, 15), FONT, 0.5, (255, 0, 255), 1)
                  
        return centers, area
    
    return [], None


def detect_color_blocks(img: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """检测颜色块并返回结果和处理后的图像"""
    if img is None:
        return {}, img
    
    try:
        # 图像预处理
        processed_img = preprocess_image(img)
        
        # 重用输入图像，不创建新副本
        result_image = img
        
        # 转换到HSV颜色空间 - 仅做一次转换
        hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        
        result_json = {}
        
        # 仅遍历可能存在的颜色
        for color_name, color_config in CONFIG['colors'].items():
            # 创建颜色掩码
            mask = create_color_mask(hsv, color_config['ranges'])
            
            # 形态学操作改善掩码质量 - 使用预先计算的内核
            kernel = CONFIG['morph_kernel']
            # 将两个操作合并，减少临时数组的创建
            mask = cv2.morphologyEx(
                cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
                cv2.MORPH_CLOSE, kernel
            )
            
            # 查找轮廓 - 使用CHAIN_APPROX_SIMPLE降低内存占用
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 处理轮廓
            centers, area = process_contours(contours, color_name, result_image)
            
            # 仅在检测到轮廓时构建结果
            if centers and area is not None:
                cx, cy = centers[0]  # 取第一个中心点
                result_json = {
                    "color": color_name,
                    "position": {"x": cx, "y": cy},
                    "area": float(area)  # 确保是可序列化的类型
                }
                # 找到符合条件的色块后退出循环
                break
                
        return result_json, result_image
        
    except Exception as e:
        rospy.logerr(f"颜色检测失败: {str(e)}")
        return {}, img


class ColorDetector:
    def __init__(self):
        """初始化检测器，优化资源使用"""
        # 只初始化一次ROS节点
        rospy.init_node('colorDetect', anonymous=False, disable_signals=True)
        
        # 创建一个可重用的CvBridge实例
        # self.bridge = CvBridge()
        
        # 设置缓冲区大小为较小的值，减少内存使用
        self.result_pub = rospy.Publisher('/detect_result', String, queue_size=2)
        self.image_pub = rospy.Publisher('/camera_r/color/image_detected', Image, queue_size=2)
        
        # 添加一个发布者用于输出当前处理频率
        self.fps_pub = rospy.Publisher('/color_detector/fps', String, queue_size=1)
        
        # 使用较小的队列大小
        self.image_sub = rospy.Subscriber(
            '/camera_r/color/image_raw', 
            Image, 
            self.image_callback,
            queue_size=1,  # 只处理最新的一帧
            buff_size=2*1024*1024  # 2MB缓冲区，适应大多数图像
        )
        

        
        # 添加关闭标志
        self.is_shutdown = False
        
        # 添加频率控制
        self.target_period = 0.1  # 10Hz等于每帧0.1秒
        self.last_callback_time = rospy.Time.now()
        


        
        rospy.loginfo("Color detector node initialized with 10Hz frequency cap")
        
    def image_callback(self, data):
        """处理图像回调，控制为10Hz频率"""
        # 计算自上次处理以来的时间
        now = rospy.Time.now()
        time_since_last = (now - self.last_callback_time).to_sec()
        
        # 如果距离上次处理的时间不足目标周期，则跳过这一帧
        if time_since_last < self.target_period:
            return
        
            
        try:
            # 更新上次处理时间
            self.last_callback_time = now
            
            # 将ROS图像消息转换为OpenCV格式
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = imgmsg_to_cv2(data)
            
            # 检测颜色块
            result_json, result_image = detect_color_blocks(cv_image)
            
            # 只在有结果时发布消息
            if result_json:
                # 发布检测结果 - 预先序列化JSON
                result_msg = String()
                result_msg.data = json.dumps(result_json)
                self.result_pub.publish(result_msg)
                
                # 发布标注后的图像 - 避免不必要的图像转换
                if self.image_pub.get_num_connections() > 0:
                    ros_image = cv2_to_imgmsg(result_image)
                    self.image_pub.publish(ros_image)
            


            
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
            


    def shutdown(self):
        """安全关闭节点"""
        self.is_shutdown = True
        rospy.loginfo("正在安全关闭检测器...")
        
        # 取消订阅
        self.image_sub.unregister()
        
        # 关闭发布者
        self.result_pub.unregister()
        self.image_pub.unregister()
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        rospy.loginfo("检测器已安全关闭")
    
    def run(self):
        """自定义的节点运行循环，支持更好的信号处理"""
        rate = rospy.Rate(10)  # 10Hz的检查频率
        
        try:
            while not rospy.is_shutdown() and not self.is_shutdown:
                # 使用sleep代替rospy.spin()，使我们能够定期检查shutdown标志
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("接收到键盘中断")
        finally:
            self.shutdown()


def signal_handler(sig, frame):
    """处理Ctrl+C等终止信号"""
    rospy.loginfo("接收到终止信号，正在关闭节点...")
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    # 关闭ROS节点
    rospy.signal_shutdown("用户手动关闭")
    # 确保完全退出
    import sys
    sys.exit(0)

if __name__ == "__main__":
    try:
        # 注册信号处理函数
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        detector = ColorDetector()
        rospy.loginfo("色块检测节点已启动，按Ctrl+C可安全退出")
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS中断异常，正在退出...")
        cv2.destroyAllWindows()
    except Exception as e:
        rospy.logerr(f"意外错误: {str(e)}")
        cv2.destroyAllWindows()