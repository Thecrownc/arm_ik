#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS版彩色方块检测节点
订阅摄像头图像，检测红色和绿色矩形方块，并发布检测结果
"""

import rospy
import cv2
import numpy as np
import json
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

# 配置参数
CONFIG = {
    'min_area': 3000,           # 最小轮廓面积阈值 (像素^2)
    'blur_kernel_size': 5,      # 中值滤波核大小
    'morph_kernel': np.ones((5, 5), np.uint8),  # 预计算形态学操作的内核
    # 颜色定义 - 每种颜色包含HSV范围和显示用的BGR颜色
    'colors': {
        'red': {
            'ranges': [
                # 红色在HSV空间分布在两个区域，需要两个范围
                {'lower': np.array([0, 120, 50], dtype=np.uint8), 'upper': np.array([10, 255, 255], dtype=np.uint8)},
                {'lower': np.array([170, 120, 50], dtype=np.uint8), 'upper': np.array([180, 255, 255], dtype=np.uint8)}
            ],
            'bgr': (0, 0, 255)  # 显示用的BGR颜色值 (红色)
        },
        'green': {
            'ranges': [
                {'lower': np.array([35, 120, 50], dtype=np.uint8), 'upper': np.array([85, 255, 255], dtype=np.uint8)}
            ],
            'bgr': (0, 255, 0)  # 显示用的BGR颜色值 (绿色)
        }
    },
    # 方块尺寸筛选条件
    'rect_filter': {
        'width_range': (60, 100),   # 宽度范围(像素)
        'height_range': (60, 160)   # 高度范围(像素)
    },
    # 连续检测配置
    'required_frames': 5        # 需要连续检测到的帧数
}

# 预先创建常用字体
FONT = cv2.FONT_HERSHEY_SIMPLEX

class ColorDetectorNode:
    """
    ROS颜色检测器节点类
    """
    def __init__(self):
        """初始化ROS节点和相关资源"""
        # 初始化ROS节点
        rospy.init_node('color_detector_node', anonymous=True)
        
        # 创建CV桥接器，用于ROS图像消息和OpenCV图像的转换
        self.bridge = CvBridge()
        
        # 创建图像订阅者，订阅摄像头图像
        self.image_sub = rospy.Subscriber(
            "/camera_r/color/image_raw", 
            Image, 
            self.image_callback, 
            queue_size=1
        )
        
        # 创建检测结果发布者，发布JSON格式的检测结果
        self.result_pub = rospy.Publisher(
            "/detect_result", 
            String, 
            queue_size=10
        )
        
        # 可选：创建处理后的图像发布者，用于调试
        self.processed_image_pub = rospy.Publisher(
            "/color_detection/processed_image", 
            Image, 
            queue_size=1
        )
        
        # 帧计数器
        self.frame_count = 0
        
        # 连续检测到的帧数
        self.consecutive_frames = 0
        
        # 是否显示处理后的图像（调试用）
        self.show_image = rospy.get_param('~show_image', False)
        
        rospy.loginfo("颜色检测器节点已初始化")
    
    def create_color_mask(self, hsv_img, color_ranges):
        """
        创建特定颜色的掩码
        
        Args:
            hsv_img (ndarray): HSV格式的图像
            color_ranges (list): 颜色范围列表，每个元素包含lower和upper阈值
            
        Returns:
            ndarray: 二值掩码，目标颜色区域为白色(255)
        """
        # 性能优化：对于单一范围的情况，直接使用inRange
        if len(color_ranges) == 1:
            return cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
        
        # 对于多范围颜色(如红色)，合并多个掩码
        mask = cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
        for range_dict in color_ranges[1:]:
            mask |= cv2.inRange(hsv_img, range_dict['lower'], range_dict['upper'])
        return mask
    
    def image_callback(self, data):
        """
        处理订阅到的图像消息
        
        Args:
            data (sensor_msgs.msg.Image): ROS图像消息
        """
        try:
            # 将ROS图像消息转换为OpenCV格式
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"图像转换错误: {e}")
            return
        
        # 增加帧计数
        self.frame_count += 1
        
        # 处理图像
        result_image, detection_results = self.process_frame(frame)
        
        # 发布检测结果（JSON格式）
        if detection_results:
            result_json = json.dumps(detection_results)
            self.result_pub.publish(result_json)
        
        # 发布处理后的图像（可选，用于调试）
        if self.processed_image_pub.get_num_connections() > 0:
            try:
                processed_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                self.processed_image_pub.publish(processed_msg)
            except CvBridgeError as e:
                rospy.logerr(f"发布处理后图像错误: {e}")
        
        # 显示处理后的图像（如果启用）
        if self.show_image:
            cv2.imshow("Color Detection", result_image)
            cv2.waitKey(1)
    
    def process_frame(self, frame):
        """
        处理单帧图像，检测并标记特定颜色和形状的目标
        
        Args:
            frame (ndarray): 输入图像帧
            
        Returns:
            tuple: (处理后的图像, 检测结果列表)
        """
        # 创建图像副本以便绘制
        result_image = frame
        
        # 获取图像尺寸和中心点
        img_height, img_width = frame.shape[:2]
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        # 获取方块尺寸筛选范围
        width_range = CONFIG['rect_filter']['width_range']
        height_range = CONFIG['rect_filter']['height_range']

        # 预处理：中值滤波减少噪声
        filter_image = cv2.medianBlur(frame, CONFIG['blur_kernel_size'])
        
        # 转换到HSV颜色空间 (更适合颜色检测)
        hsv = cv2.cvtColor(filter_image, cv2.COLOR_BGR2HSV)
        
        # 当前帧检测到的结果
        current_valid_blocks = []
        
        # 遍历检测每种颜色
        for color_name, color_config in CONFIG['colors'].items():
            # 创建颜色掩码
            mask = self.create_color_mask(hsv, color_config['ranges'])
            
            # 形态学操作改善掩码质量：先开操作(去除噪点)，后闭操作(填充孔洞)
            kernel = CONFIG['morph_kernel']
            mask = cv2.morphologyEx(
                cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
                cv2.MORPH_CLOSE, kernel
            )
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 获取颜色配置
            bgr_color = color_config['bgr']
            
            # 处理每个轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 过滤掉面积过小的轮廓
                if area > CONFIG['min_area']:
                    # 计算轮廓近似多边形 (用于形状分析)
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
                    if len(approx) == 4:
                        # 获取边界矩形
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # 检查是否符合矩形方块的尺寸要求
                        is_width_valid = width_range[0] <= w <= width_range[1]
                        is_height_valid = height_range[0] <= h <= height_range[1]
                        is_height_greater_than_width = h >= w
                        
                        # 只处理符合条件的形状：宽度合适、高度合适、高度大于宽度
                        if is_width_valid and is_height_valid and is_height_greater_than_width:
                            # 计算中心点
                            center_x = x + w / 2
                            center_y = y + h / 2
                            
                            # 计算相对于图像中心的坐标 (采用常见的计算机视觉坐标系)
                            cx = center_x - img_center_x
                            cy = img_center_y - center_y  # 注意y轴方向翻转
                            
                            # 添加检测结果
                            detection_result = {
                                "color": color_name,
                                "position": {
                                    "x": cx,
                                    "y": cy
                                },
                                "area": area
                            }
                            current_valid_blocks.append(detection_result)
                            
                            # 绘制轮廓
                            cv2.drawContours(result_image, [contour], -1, bgr_color, 2)
                            
                            # 绘制中心点
                            cv2.circle(result_image, (center_x, center_y), 5, bgr_color, -1)
                            
                            # 绘制坐标文本
                            coord_text = f"({int(cx)}, {int(cy)})"
                            cv2.putText(result_image, coord_text,
                                    (center_x + 10, center_y - 10),
                                    FONT, 0.5, bgr_color, 2)
                            
                            # 显示颜色名称
                            cv2.putText(result_image, color_name,
                                    (center_x + 10, center_y + 20),
                                    FONT, 0.5, bgr_color, 2)
                            
                            # 显示尺寸和面积
                            size_text = f"Size: {w}x{h}, Area: {int(area)}"
                            cv2.putText(result_image, size_text,
                                    (center_x + 10, center_y + 40),
                                    FONT, 0.5, bgr_color, 2)
        
        # 检查当前帧是否有合格色块
        if current_valid_blocks:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0
        
        # 显示连续检测帧数
        status_text = f"Frame: {self.frame_count} | Consecutive: {self.consecutive_frames}/{CONFIG['required_frames']}"
        cv2.putText(result_image, status_text, (10, 30), FONT, 0.7, (0, 255, 255), 2)
        
        # 只有连续检测到指定帧数才返回结果
        if self.consecutive_frames >= CONFIG['required_frames']:
            return result_image, current_valid_blocks
        else:
            return result_image, []
    
    def run(self):
        """运行ROS节点，处理回调"""
        # 使用rospy.spin()保持节点运行，处理回调
        rospy.loginfo("颜色检测器节点正在运行")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("用户终止节点")
        finally:
            # 关闭所有OpenCV窗口
            if self.show_image:
                cv2.destroyAllWindows()
            rospy.loginfo("颜色检测器节点已关闭")


if __name__ == '__main__':
    try:
        detector_node = ColorDetectorNode()
        detector_node.run()
    except rospy.ROSInterruptException:
        pass