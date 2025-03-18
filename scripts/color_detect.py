#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from arm_ik.srv import ColorDetector, ColorDetectorResponse
import numpy as np

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

class ImageProcessingServer:
    def __init__(self):
        # 初始化ROS节点
        print("init node ok")

        rospy.init_node('detect_color')
        print("init node ok")
        # 创建CV桥接器
        self.bridge = CvBridge()
        print("CvBridge node ok")
        
        # 创建服务
        self.service = rospy.Service('/detect_color', ColorDetector, self.handle_request)
        
        # 订阅者和接收到的图像
        self.image_subscriber = None
        self.current_image = None
        self.received_image = False
        
        rospy.loginfo("图像处理服务已启动，等待请求...")
    
    def image_callback(self, msg):
        """
        当收到图像时的回调函数
        """
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
            self.received_image = True
            rospy.loginfo("已接收图像")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

    def create_color_mask(self, hsv_img, color_ranges):
        """创建颜色掩码，优化版本"""
        # 仅处理一个颜色范围的常见情况
        if len(color_ranges) == 1:
            return cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
        
        # 处理多颜色范围的情况
        mask = cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
        for range_dict in color_ranges[1:]:
            mask |= cv2.inRange(hsv_img, range_dict['lower'], range_dict['upper'])
        return mask

    def process_image(self, image):
        """
        处理图像的函数，可以根据需要修改
        """
        filter_image = cv2.medianBlur(image, 5)

        result_image = image  # 创建图像副本以便绘制

        # 获取图像尺寸
        img_height, img_width = image.shape[:2]
        img_center_x = img_width / 2
        img_center_y = img_height / 2

        hsv = cv2.cvtColor(filter_image, cv2.COLOR_BGR2HSV)

        result_json = {}

        # 仅遍历可能存在的颜色
        for color_name, color_config in CONFIG['colors'].items():
            # 创建颜色掩码
            mask = self.create_color_mask(hsv, color_config['ranges'])
            
            # 形态学操作改善掩码质量 - 使用预先计算的内核
            kernel = CONFIG['morph_kernel']
            # 将两个操作合并，减少临时数组的创建
            mask = cv2.morphologyEx(
                cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
                cv2.MORPH_CLOSE, kernel
            )
            
            # 查找轮廓 - 使用CHAIN_APPROX_SIMPLE降低内存占用
            # 兼容不同版本的OpenCV
            if cv2.__version__.startswith('3.'):
                # OpenCV 3.x
                _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                # OpenCV 4.x及以上
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 处理轮廓
            contour_result = self.process_contours(contours, color_name, result_image, img_center_x, img_center_y)
            
            # 仅在检测到轮廓时构建结果
            if contour_result:

                cx, cy, area, area_percentage = contour_result

                result_json = {
                    "color": color_name,
                    "position": {"x": cx, "y": cy},
                    "area": float(area),
                    "area_percentage": float(area_percentage)
                }
                # 找到符合条件的色块后退出循环
                break

        return result_json
    
    def process_contours(self, contours, color_name, result_image, img_center_x, img_center_y):
        if not contours:  # 快速检查是否有轮廓
            return None

        total_area = CONFIG['image_total_area']
        color_config = CONFIG['colors'][color_name]
        bgr_color = color_config['bgr']

        max_area = 0
        max_area_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > CONFIG['min_area'] and area > max_area:
                # 仅对超过阈值的大轮廓进行多边形逼近
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                
                # 只处理矩形（顶点数为4的形状）
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)

                    # 计算以图像中心为原点的坐标
                    cx = x + w / 2 - img_center_x
                    cy = img_center_y - (y + h / 2)  # 注意y轴方向翻转
                    
                    # 计算面积百分比
                    area_percentage = (area / total_area) * 100

                    # 更新最大面积的轮廓
                    max_area = area
                    max_area_contour = {
                        'contour': contour,
                        'area': area,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'cx': cx,
                        'cy': cy,
                        'area_percentage': area_percentage
                    }

        # 如果找到合适的轮廓
        if max_area_contour:
            # 绘制轮廓
            cv2.drawContours(result_image, [max_area_contour['contour']], -1, bgr_color, 2)

            # 绘制中心点
            screen_cx = max_area_contour['x'] + max_area_contour['w'] / 2
            screen_cy = max_area_contour['y'] + max_area_contour['h'] / 2
            cv2.circle(result_image, (int(screen_cx), int(screen_cy)), 5, bgr_color, -1)

            # 返回中心点x、y、面积和面积百分比
            return (
                max_area_contour['cx'], 
                max_area_contour['cy'], 
                max_area_contour['area'], 
                max_area_contour['area_percentage']
            )
        
        return None

    def handle_request(self, req):
        """
        处理服务请求的函数
        """
        topic_name = req.camera_topic
        rospy.loginfo("收到处理图像请求，话题名: %s", topic_name)
        
        # 重置图像接收状态
        self.received_image = False
        self.current_image = None
        
        # 创建订阅者
        self.image_subscriber = rospy.Subscriber(
            topic_name, Image, self.image_callback, queue_size=1)
        
        # 等待接收图像，设置超时
        timeout = rospy.Duration(5.0)  # 5秒超时
        start_time = rospy.Time.now()
        
        while not self.received_image and (rospy.Time.now() - start_time) < timeout:
            rospy.sleep(0.1)
        
        # 取消订阅
        if self.image_subscriber:
            self.image_subscriber.unregister()
            self.image_subscriber = None
        
        # 检查是否收到图像
        if not self.received_image:
            rospy.logwarn("未能在超时时间内接收到图像")
            return None
        
        # 处理图像
        try:
            rospy.loginfo("start process image")
            result_json = self.process_image(self.current_image)

            if result_json:
                rospy.loginfo("图像处理完成")
                # 返回检测到的颜色块信息
                return ColorDetectorResponse(
                    color = result_json['color'],
                    x=result_json['position']['x'], 
                    y=result_json['position']['y'], 
                    area=result_json['area'], 
                    area_percentage=result_json['area_percentage']
                )
            else:
                return None
            
        except Exception as e:
            rospy.logerr("图像处理错误: %s", str(e))
            return None

if __name__ == '__main__':
    print("aaaaa")
    server = ImageProcessingServer()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("服务器已关闭")