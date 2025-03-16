import cv2
import numpy as np
import time
from collections import deque
import math

class VideoColorRectangleDetector:
    def __init__(self, video_source=0, target_fps=30, min_continuous_frames=3):
        """
        初始化视频流颜色矩形检测器
        
        Args:
            video_source (int/str): 视频源 
            0代表默认摄像头，也可以是视频文件路径
            target_fps (int): 目标帧率
            min_continuous_frames (int): 最小连续检测帧数
        """
        # 打开视频流
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源 {video_source}")
        
        # 设置帧率相关参数
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps  # 每帧时间间隔
        
        # 设置连续检测参数
        self.min_continuous_frames = min_continuous_frames
        self.detection_history = {
            'green': deque(maxlen=min_continuous_frames),  # 绿色检测历史
            'red': deque(maxlen=min_continuous_frames)     # 红色检测历史
        }
        
        # 创建窗口
        cv2.namedWindow('rectangle')
        cv2.namedWindow('HSV')
        self.debug_mode = True
        # 初始化HSV调节滑动条
        self._create_trackbars()
        
    def _create_trackbars(self):
        """
        创建HSV颜色范围调节滑动条
        """
        # 绿色HSV滑动条
        cv2.createTrackbar('green H lower', 'HSV', 40, 179, self._on_trackbar)
        cv2.createTrackbar('green H upper', 'HSV', 80, 179, self._on_trackbar)
        cv2.createTrackbar('green S lower', 'HSV', 50, 255, self._on_trackbar)
        cv2.createTrackbar('green S upper', 'HSV', 255, 255, self._on_trackbar)
        cv2.createTrackbar('green V lower', 'HSV', 50, 255, self._on_trackbar)
        cv2.createTrackbar('green V upper', 'HSV', 255, 255, self._on_trackbar)
        
        # 红色HSV滑动条（考虑HSV色环特性）
        cv2.createTrackbar('red H lower1', 'HSV', 0, 179, self._on_trackbar)
        cv2.createTrackbar('red H upper1', 'HSV', 10, 179, self._on_trackbar)
        cv2.createTrackbar('red H lower2', 'HSV', 170, 179, self._on_trackbar)
        cv2.createTrackbar('red H upper2', 'HSV', 179, 179, self._on_trackbar)
        cv2.createTrackbar('red S lower', 'HSV', 50, 255, self._on_trackbar)
        cv2.createTrackbar('red S upper', 'HSV', 255, 255, self._on_trackbar)
        cv2.createTrackbar('red V lower', 'HSV', 50, 255, self._on_trackbar)
        cv2.createTrackbar('red V upper', 'HSV', 255, 255, self._on_trackbar)
        
    def _on_trackbar(self, x):
        """
        滑动条回调函数（实际上不执行任何操作，仅保持接口一致）
        """
        pass
    
    def _get_color_mask(self, hsv_frame):
        """
        获取绿色和红色的HSV掩码
        
        Args:
            hsv_frame (numpy.ndarray): HSV颜色空间的帧
        
        Returns:
            tuple: (绿色掩码, 红色掩码)
        """
        # 获取当前绿色HSV参数
        green_h_low = cv2.getTrackbarPos('green H lower', 'HSV')
        green_h_high = cv2.getTrackbarPos('green H upper', 'HSV')
        green_s_low = cv2.getTrackbarPos('green S lower', 'HSV')
        green_s_high = cv2.getTrackbarPos('green S upper', 'HSV')
        green_v_low = cv2.getTrackbarPos('green V lower', 'HSV')
        green_v_high = cv2.getTrackbarPos('green V upper', 'HSV')
        
        # 获取当前红色HSV参数
        red_h_low1 = cv2.getTrackbarPos('red H lower1', 'HSV')
        red_h_high1 = cv2.getTrackbarPos('red H upper1', 'HSV')
        red_h_low2 = cv2.getTrackbarPos('red H lower2', 'HSV')
        red_h_high2 = cv2.getTrackbarPos('red H upper2', 'HSV')
        red_s_low = cv2.getTrackbarPos('red S lower', 'HSV')
        red_s_high = cv2.getTrackbarPos('red S upper', 'HSV')
        red_v_low = cv2.getTrackbarPos('red V lower', 'HSV')
        red_v_high = cv2.getTrackbarPos('red V upper', 'HSV')
        
        # 创建绿色掩码
        green_mask = cv2.inRange(hsv_frame, 
            (green_h_low, green_s_low, green_v_low), 
            (green_h_high, green_s_high, green_v_high)
        )
        
        # 创建红色掩码（考虑色环特性）
        red_mask1 = cv2.inRange(hsv_frame, 
            (red_h_low1, red_s_low, red_v_low), 
            (red_h_high1, red_s_high, red_v_high)
        )
        red_mask2 = cv2.inRange(hsv_frame, 
            (red_h_low2, red_s_low, red_v_low), 
            (red_h_high2, red_s_high, red_v_high)
        )
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        return green_mask, red_mask
    
    def _process_contours(self, frame, contours, color):
        """
        处理并绘制检测到的轮廓
        
        Args:
            frame (numpy.ndarray): 原始帧
            contours (list): 轮廓列表
            color (tuple): 绘制矩形的颜色 (B,G,R)
        
        Returns:
            list: 检测到的矩形中心点坐标
        """
        centers = []
        
        # 首先检测圆形
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        circles = self._detect_circles(mask)
        
        # 在调试模式下绘制检测到的圆形
        if self.debug_mode:
            for cx, cy, r in circles:
                # 绘制圆形
                cv2.circle(frame, (cx, cy), r, (255, 255, 0), 2)  # 黄色显示圆形
                # 绘制圆心
                cv2.circle(frame, (cx, cy), 2, (255, 255, 0), 3)
                # 显示半径
                cv2.putText(frame, f"R:{r}", (cx-20, cy-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 相机参数 - 这些应该通过相机标定获取
        focal_length = 1000.0  # 焦距（像素单位）
        known_width = 0.05  # 已知物体宽度（米）
        known_area = 0.0025  # 已知物体面积（平方米）
        
        # 面积-距离乘积常数（通过实验标定）
        # 这个值应该通过实际测量获得：在已知距离处测量物体的像素面积
        area_distance_product = 1000000  # 示例值
        
        for contour in contours:
            # 过滤小轮廓以减少噪声
            area = cv2.contourArea(contour)
            if area > 4000:
                # 获取最小外接矩形
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # 计算矩形的宽和高
                width = rect[1][0]
                height = rect[1][1]
                
                # 计算长宽比
                aspect_ratio = width / height
                
                # 计算矩形度（轮廓面积与最小外接矩形面积之比）
                rect_area = width * height
                rectangularity = area / rect_area if rect_area > 0 else 0
                
                # 计算轮廓的圆形度
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # 计算中心点
                center_x = int(rect[0][0])
                center_y = int(rect[0][1])
                
                # 检查中心点是否在任何检测到的圆形内
                if not self._is_point_in_circles((center_x, center_y), circles):
                    # 估计距离 - 使用面积反比定律
                    # 距离 = sqrt(面积-距离乘积常数 / 像素面积)
                    estimated_distance = math.sqrt(area_distance_product / area)
                    
                    # 或者使用已知宽度估计距离
                    # 距离 = 焦距 * 已知宽度 / 像素宽度
                    distance_by_width = focal_length * known_width / width
                    
                    # 在调试模式下绘制轮廓和特征
                    if self.debug_mode:
                        # 绘制轮廓
                        cv2.drawContours(frame, [box], 0, color, 2)
                        # 绘制中心点
                        cv2.circle(frame, (center_x, center_y), 5, color, -1)
                        # 显示特征值
                        cv2.putText(frame, f"R:{rectangularity:.2f}", (center_x-40, center_y-40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.putText(frame, f"C:{circularity:.2f}", (center_x-40, center_y-20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        # 显示宽和高
                        cv2.putText(frame, f"W:{int(width)}", (center_x-40, center_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.putText(frame, f"H:{int(height)}", (center_x-40, center_y+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        # 显示长宽比
                        cv2.putText(frame, f"AR:{aspect_ratio:.2f}", (center_x-40, center_y+40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        # 显示估计距离
                        cv2.putText(frame, f"D:{estimated_distance:.2f}m", (center_x-40, center_y+60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # 将距离信息添加到中心点数据中
                    centers.append((center_x, center_y, width, height, estimated_distance))
        
        return centers
    
    def _is_continuous_detection(self, color):
        """
        检查是否连续检测到目标
        
        Args:
            color (str): 颜色名称 ('green' 或 'red')
        
        Returns:
            bool: 是否连续检测到目标
        """
        history = self.detection_history[color]
        return len(history) == self.min_continuous_frames and all(history)
    
    def _draw_detection(self, frame, centers, color):
        """
        绘制检测结果
        
        Args:
            frame (numpy.ndarray): 要绘制的帧
            centers (list): 中心点列表
            color (tuple): 绘制颜色 (B,G,R)
        """
        for center_x, center_y in centers:
            # 绘制矩形边界
            x = center_x - 20  # 假设矩形大小为40x40
            y = center_y - 20
            cv2.rectangle(frame, (x, y), (x+40, y+40), color, 2)
            # 绘制中心点
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
    
    def run(self):
        """
        运行实时视频流处理，控制帧率为30帧
        """
        # 记录处理开始时间
        prev_frame_time = 0
        
        while True:
            # 记录当前时间
            current_time = time.time()
            
            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取视频帧")
                break
            
            # 帧率控制：确保每帧间隔符合目标帧率
            if current_time - prev_frame_time < self.frame_time:
                time.sleep(max(0, self.frame_time - (current_time - prev_frame_time)))
            
            # 更新上一帧处理时间
            prev_frame_time = time.time()
            
            # 创建帧副本用于绘制
            display_frame = frame.copy()
            
            # 转换到HSV颜色空间
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 获取颜色掩码
            green_mask, red_mask = self._get_color_mask(hsv_frame)
            
            # 预处理：降噪
            kernel = np.ones((5,5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # 找到轮廓
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 处理绿色轮廓
            green_centers = self._process_contours(display_frame, green_contours, (0, 255, 0))
            # 更新绿色检测历史
            self.detection_history['green'].append(len(green_centers) > 0)
            
            # 处理红色轮廓
            red_centers = self._process_contours(display_frame, red_contours, (0, 0, 255))
            # 更新红色检测历史
            self.detection_history['red'].append(len(red_centers) > 0)
            
            # 只在连续检测到时绘制
            if self._is_continuous_detection('green'):
                self._draw_detection(display_frame, green_centers, (0, 255, 0))
            
            if self._is_continuous_detection('red'):
                self._draw_detection(display_frame, red_centers, (0, 0, 255))
            

            
            # 显示帧率和检测状态
            # fps = 1 / (time.time() - prev_frame_time)
            # cv2.putText(display_frame, f'FPS: {fps:.2f}', (10, 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示连续检测状态
            green_status = "Green: Detected" if self._is_continuous_detection('green') else "Green: Searching"
            red_status = "Red: Detected" if self._is_continuous_detection('red') else "Red: Searching"
            cv2.putText(display_frame, green_status, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, red_status, (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 展示结果
            cv2.imshow('rectangle', display_frame)
            cv2.imshow('green_mask', green_mask)
            cv2.imshow('red_mask', red_mask)
            # 检查是否退出
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC键
                break
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

# 使用示例
if __name__ == '__main__':
    try:
        # 0 表示默认摄像头，也可以是视频文件路径
        detector = VideoColorRectangleDetector("D:/Code/arm_ik-a1pass/src/data/aim_data/aim_2.avi", 
                                             target_fps=30,
                                             min_continuous_frames=5)  # 需要连续3帧检测到才显示
        detector.run()
    except Exception as e:
        print(f"发生错误: {e}")