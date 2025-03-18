#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import os
import json
from datetime import datetime
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from arm_ik.srv import ColorDetector, ColorDetectorResponse
import numpy as np
from collections import deque

# 配置参数 - 预先计算的可复用值
CONFIG = {
    'min_area': 3000,  # 最小面积阈值
    'rect_ratio': 0.85,  # 矩形比例阈值
    'resize_width': 640,  # 处理图像宽度
    'blur_kernel_size': 5,  # 中值滤波核大小
    'morph_kernel': np.ones((5, 5), np.uint8),  # 形态学操作的预计算核
    'image_total_area': 640*480,  # 图像总面积
    'colors': {
        'red': {  # 红色定义
            'ranges': [  # 红色需要两个HSV范围（跨越了H通道的0）
                {'lower': np.array([0, 120, 50], dtype=np.uint8), 'upper': np.array([10, 255, 255], dtype=np.uint8)},
                {'lower': np.array([170, 120, 50], dtype=np.uint8), 'upper': np.array([180, 255, 255], dtype=np.uint8)}
            ],
            'bgr': (0, 0, 255)  # BGR颜色表示（用于绘图）
        },
        'green': {  # 绿色定义
            'ranges': [  # 绿色的HSV范围
                {'lower': np.array([35, 120, 50], dtype=np.uint8), 'upper': np.array([85, 255, 255], dtype=np.uint8)}
            ],
            'bgr': (0, 255, 0)  # BGR颜色表示（用于绘图）
        }
    },
    # 矩形尺寸过滤条件
    'rect_filter': {
        'width_range': (50, 160),   # 宽度范围（像素）
        'height_range': (50, 160)   # 高度范围（像素）
    },
    # 保存配置
    'save': {
        'base_dir': os.path.expanduser('./color_detection_results'),  # 保存结果的基本目录
        'json_filename': 'detection_results.json',  # JSON文件名
        'image_prefix': 'detection_'  # 图像文件前缀
    },
    # 新增：连续检测配置
    'consecutive_detection': {
        'required_frames': 10,      # 需要连续检测的帧数
        'max_difference': 100        # 允许的cx, cy最大差值
    }
}

# 预创建的通用字体
FONT = cv2.FONT_HERSHEY_SIMPLEX

class ImageProcessingServer:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('detect_color')
        
        # 创建CV桥接器
        self.bridge = CvBridge()
        
        # 创建服务
        self.service = rospy.Service('/detect_color', ColorDetector, self.handle_request)
        
        # 订阅者和接收图像
        self.image_subscriber = None
        self.current_image = None
        self.received_image = False
        
        # 新增：用于存储连续检测结果的队列
        self.detection_results = deque(maxlen=CONFIG['consecutive_detection']['required_frames'])
        
        # 创建一个空的默认响应
        self.empty_response = ColorDetectorResponse()
        
        # 如果不存在，创建结果目录
        self.setup_result_directories()
        
        rospy.loginfo("图像处理服务已启动，等待请求...")
    
    def setup_result_directories(self):
        """创建保存结果的目录"""
        base_dir = CONFIG['save']['base_dir']
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            rospy.loginfo(f"已创建结果目录: {base_dir}")
    
    def image_callback(self, msg):
        """
        回调只负责更新current_image
        """
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
            self.received_image = True
            # rospy.loginfo("已接收图像")
        except CvBridgeError as e:
            rospy.logerr("CvBridge错误: %s", e)

    def create_color_mask(self, hsv_img, color_ranges):
        """创建颜色掩码，优化版本"""
        # 处理单个颜色范围的常见情况
        if len(color_ranges) == 1:
            return cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
        
        # 处理多个颜色范围
        mask = cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
        for range_dict in color_ranges[1:]:
            mask |= cv2.inRange(hsv_img, range_dict['lower'], range_dict['upper'])
        return mask

    def save_detection_results(self, image, result_json, contour=None):
        """保存检测结果到磁盘 - 包括图像和JSON数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间戳格式化
        base_dir = CONFIG['save']['base_dir']
        img_height, img_width = image.shape[:2]
        
        # 保存JSON结果
        if result_json:
            result_json['timestamp'] = timestamp  # 添加时间戳到结果
            result_path = os.path.join(base_dir, f"{timestamp}_{CONFIG['save']['json_filename']}")
            with open(result_path, 'w') as f:
                json.dump(result_json, f, indent=2)
            rospy.loginfo(f"检测结果已保存到 {result_path}")
        
        # 如果有结果则保存标注图像
        if image is not None and result_json:
            # 创建图像的副本用于标注
            # annotated_img = image.copy()
            # 打印坐标用于调试
            # 绘制检测到的位置
            color_bgr = CONFIG['colors'][result_json['color']]['bgr']
            center_x = int(result_json['position']['x'] + image.shape[1]/2)  # 转换为图像坐标系
            center_y = int(image.shape[0]/2 + result_json['position']['y'])  # 转换为图像坐标系
            
            # 在中心点绘制十字标记
            cv2.circle(image, (int(center_x), int(center_y)), 10, (0,0,255), 2)
            
            #在图像中心绘制十字标识
            cv2.drawMarker(image, (int(img_width/2),int(img_height/2)), 
            (255,0,255), markerType=cv2.MARKER_CROSS, 
            markerSize=20, thickness=2)
            
            
            # 绘制中心点
            cv2.circle(image, (int(center_x), int(center_y)), 5, color_bgr, -1)
            
            # 绘制坐标文本
            coord_text = f"({int(result_json['position']['x'])}, {int(result_json['position']['y'])})"
            cv2.putText(image, coord_text,
                    (int(center_x) + 10, int(center_y) - 10),
                    FONT, 0.5, color_bgr, 2)
            
            # 显示颜色名称
            color_name = f"{result_json['color']}"
            cv2.putText(image, color_name,
                    (int(center_x) + 10, int(center_y) + 20),
                    FONT, 0.5, color_bgr, 2)
            
            # 显示尺寸和面积
            size_text = f"Area: {int(result_json['area'])} ,Per:{result_json['area_percentage']}"
            cv2.putText(image, size_text,
                    (int(center_x) + 10, int(center_y) + 40),
                    FONT, 0.5, color_bgr, 2)
            
            # 保存图像
            img_path = os.path.join(base_dir, f"{timestamp}_{CONFIG['save']['image_prefix']}.jpg")
            cv2.imwrite(img_path, image)
            rospy.loginfo(f"标注图像已保存到 {img_path}")
            
            return img_path
        
        return None

    def process_image(self, image, target_color):
        """
        处理图像以检测彩色对象
        """
        
        # 验证目标颜色
        if target_color not in CONFIG['colors']:
            rospy.logerr(f"不支持的颜色: {target_color}")
            return {}
        
        color_config = CONFIG['colors'][target_color]

        # 应用中值滤波进行噪声降低
        filter_image = cv2.medianBlur(image, CONFIG['blur_kernel_size'])

        # 获取图像尺寸
        img_height, img_width = image.shape[:2]
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        total_area = img_height*img_width
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(filter_image, cv2.COLOR_BGR2HSV)

        result_json = {}

        # 创建颜色掩码
        mask = self.create_color_mask(hsv, color_config['ranges'])
        
        # 形态学操作以改善掩码质量 - 使用预计算核
        kernel = CONFIG['morph_kernel']
        # 组合操作以减少临时数组创建
        mask = cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
            cv2.MORPH_CLOSE, kernel
        )
        
        # 查找轮廓 - 使用CHAIN_APPROX_SIMPLE减少内存使用
        # 兼容不同版本的OpenCV
        if cv2.__version__.startswith('3.'):
            # OpenCV 3.x
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # OpenCV 4.x及以上
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 处理轮廓
        contour_result = self.process_contours(contours, img_center_x, img_center_y,total_area)
        
        # 仅在检测到轮廓时构建结果
        if contour_result:
            cx, cy, area,area_percentage = contour_result

            result_json = {
                "color": target_color,
                "position": {"x": cx, "y": cy},
                "area": float(area),
                "area_percentage":float(area_percentage)
            }
            
            return result_json

        return {}
    
    def process_contours(self, contours, img_center_x, img_center_y,total_area):
        """处理检测到的轮廓以找到匹配的矩形"""
        if not contours:  # 轮廓的快速检查
            return None

        # 获取矩形尺寸过滤范围
        width_range = CONFIG['rect_filter']['width_range']
        height_range = CONFIG['rect_filter']['height_range']

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > CONFIG['min_area']:
                # 仅近似超过阈值的大轮廓
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                # 仅处理矩形（有4个顶点的形状）
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)

                    # 计算中心点
                    center_x = x + w / 2
                    center_y = y + h / 2
                
                    # 计算相对于图像中心的坐标（使用常见的计算机视觉坐标系统）
                    cx = center_x - img_center_x
                    # cy = img_center_y - center_y  # 注意y轴方向翻转
                    cy = center_y - img_center_y  # 注意y轴方向翻转
                    
                    area_percentage = (area / total_area) * 100

                    return (cx, cy, area,area_percentage)

        return None

    def check_results_stability(self):
        """
        检查连续检测结果的稳定性
        返回：稳定的结果（如果所有cx, cy差值都在阈值内），或者None
        """
        if len(self.detection_results) < CONFIG['consecutive_detection']['required_frames']:
            return None
            
        # 提取所有cx, cy值
        cx_values = [result['position']['x'] for result in self.detection_results]
        cy_values = [result['position']['y'] for result in self.detection_results]
        
        # 计算最大和最小值的差值
        cx_diff = max(cx_values) - min(cx_values)
        cy_diff = max(cy_values) - min(cy_values)
        
        max_diff = CONFIG['consecutive_detection']['max_difference']
        
        if cx_diff <= max_diff and cy_diff <= max_diff:
            # 如果稳定，返回True
            return True
        
        
        #否则返回false
        return False

    def handle_request(self, req):
        """
        处理服务请求的函数，持续尝试检测直到找到方块
        """
        # 如果已经订阅了，取消订阅
        if self.image_subscriber:
            rospy.loginfo("清理之前的订阅者")
            self.image_subscriber.unregister()
            self.image_subscriber = None
            # 给ROS时间完全清理订阅者
            rospy.sleep(0.1)
        
        topic_name = req.camera_topic
        color_type = req.req_color
        
        rospy.loginfo(f"收到图像处理请求，话题名称: {topic_name}，颜色类型：{color_type}")
        
        # 重置图像接收状态
        self.received_image = False
        self.current_image = None
        
        # 清空检测结果队列
        self.detection_results.clear()
        
        # 回调只接收图像
        self.image_subscriber = rospy.Subscriber(
            topic_name, Image, self.image_callback, queue_size=10)

        # 不断尝试检测
        max_attemp = 300

        for attempt in range(max_attemp):
            # 等待图像
            start_time = rospy.Time.now()

            while not self.received_image and (rospy.Time.now() - start_time) < rospy.Duration(5.0):
                rospy.sleep(0.1)

            # 检查是否收到图像
            if not self.received_image:
                rospy.logwarn(f"第 {attempt+1} 次尝试：未能在超时时间内接收到图像")
                continue

            try:
                # rospy.loginfo(f"第 {attempt+1} 次尝试处理图像")
                
                result_json = self.process_image(self.current_image, color_type)
            
                # 如果当前帧检测到目标
                if result_json:
                    # 添加到检测结果队列
                    self.detection_results.append(result_json)
                    
                    # 检查是否有足够的稳定检测帧
                    
                    if self.check_results_stability():
                        if self.image_subscriber:
                            rospy.loginfo("取消订阅")
                            self.image_subscriber.unregister()
                            self.image_subscriber = None
                            # 给ROS时间清理
                            rospy.sleep(0.1)

                        rospy.loginfo(f"成功检测到稳定的色块，连续{len(self.detection_results)}帧结果稳定")
                        
                        # 将最后一帧的图像和稳定的结果保存到磁盘
                        saved_path = self.save_detection_results(self.current_image, result_json)
                        rospy.loginfo(f"检测结果已保存到 {saved_path}")

                        return ColorDetectorResponse(
                            color=result_json['color'],
                            x=result_json['position']['x'], 
                            y=result_json['position']['y'], 
                            area=result_json['area'], 
                            area_percentage=result_json['area_percentage'], 
                            
                        )
                    else:
                        rospy.loginfo(f"{len(self.detection_results)}/{CONFIG['consecutive_detection']['required_frames']}")
                else:
                    # 如果当前帧未检测到，清空队列重新开始
                    if len(self.detection_results) > 0:
                        rospy.loginfo(f"当前帧未检测到目标，重置队列")
                        self.detection_results.clear()

                # 重置接收状态，准备下一次尝试
                self.received_image = False
                self.current_image = None
                rospy.sleep(0.2)

            except Exception as e:
                rospy.logerr(f"第 {attempt+1} 次尝试图像处理错误: {str(e)}")
                # 重置接收状态
                self.received_image = False
                self.current_image = None
                rospy.sleep(0.2)
                # 清空检测结果队列
                self.detection_results.clear()

        # 如果达到最大尝试次数仍未检测到
        if self.image_subscriber:
            self.image_subscriber.unregister()
            self.image_subscriber = None
            
        rospy.logwarn("已达到最大尝试次数，未能检测到稳定的色块")
        return None
    

if __name__ == '__main__':
    server = ImageProcessingServer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("服务器关闭")