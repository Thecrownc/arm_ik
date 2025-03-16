#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
彩色方块检测程序
用于检测视频中的红色和绿色矩形方块，并标记其位置和属性
"""

import cv2
import numpy as np
import time
import os

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
    }
}

# 预先创建常用字体
FONT = cv2.FONT_HERSHEY_SIMPLEX

class ColorDetector:
    """
    颜色检测器类：用于检测视频中特定颜色和形状的物体
    """
    def __init__(self, video_path, output_path=None):
        """
        初始化颜色检测器
        
        Args:
            video_path (str): 视频文件路径
            output_path (str, optional): 输出视频路径，如果为None则不保存
        """
        # 打开视频源
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {video_path}")
        
        # 获取视频属性
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # 配置输出视频 (如果需要)
        self.output = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output = cv2.VideoWriter(
                output_path, fourcc, self.fps, 
                (self.frame_width, self.frame_height)
            )
    
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
    
    def process_frame(self, frame):
        """
        处理单帧图像，检测并标记特定颜色和形状的目标
        
        Args:
            frame (ndarray): 输入图像帧
            
        Returns:
            ndarray: 处理后的图像，包含标记和信息
        """
        # 创建图像副本以便绘制
        result_image = frame.copy()
        
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
                    if len(approx) ==4:
                        # 获取边界矩形
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # 检查是否符合矩形方块的尺寸要求
                        is_width_valid = width_range[0] <= w <= width_range[1]
                        is_height_valid = height_range[0] <= h <= height_range[1]
                        is_height_greater_than_width = h >= w
                        
                        # 只处理符合条件的形状：宽度合适、高度合适、高度大于宽度
                        if is_width_valid and is_height_valid and is_height_greater_than_width:
                            # 计算中心点
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # 计算相对于图像中心的坐标 (采用常见的计算机视觉坐标系)
                            cx = center_x - img_center_x
                            cy = img_center_y - center_y  # 注意y轴方向翻转
                            
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
                            
                            # 显示尺寸
                            size_text = f"Size: {w}x{h}"
                            cv2.putText(result_image, size_text,
                                    (center_x + 10, center_y + 40),
                                    FONT, 0.5, bgr_color, 2)
        
        # 显示结果图像
        cv2.imshow("Color Detection", result_image)
        return result_image
    
    def run(self):
        """
        运行视频处理主循环
        """
        print("开始处理视频...")
        
        # 设置目标帧率
        target_fps = 30
        frame_time = 1.0 / target_fps  # 每帧的理想时间间隔（秒）
        
        frame_count = 0
        processing_time = 0
        
        try:
            while True:
                # 记录开始处理时间
                start_time = time.time()
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("视频处理完成或无法读取帧")
                    break
                
                # 处理帧
                result_frame = self.process_frame(frame)
                
                # 如果需要保存视频
                if self.output is not None:
                    self.output.write(result_frame)
                
                # 计算处理时间
                elapsed = time.time() - start_time
                processing_time += elapsed
                frame_count += 1
                
                # 动态调整等待时间以保持稳定帧率
                wait_time = max(1, int((frame_time - elapsed) * 1000))
                
                # 检查是否退出
                key = cv2.waitKey(wait_time) & 0xFF
                if key == 27:  # ESC键
                    print("用户终止处理")
                    break
                
        except Exception as e:
            print(f"处理视频时发生错误: {e}")
        finally:
            # 输出性能统计
            if frame_count > 0:
                avg_time = processing_time / frame_count
                actual_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"处理了 {frame_count} 帧")
                print(f"平均每帧处理时间: {avg_time:.4f} 秒")
                print(f"实际帧率: {actual_fps:.2f} FPS")
            
            # 释放资源
            self.cap.release()
            if self.output is not None:
                self.output.release()
            cv2.destroyAllWindows()
            print("资源已释放，程序结束")


def main():
    """主函数"""
    # 在这里直接指定视频路径
    video_path = "D:/Code/arm_ik-a1pass/src/data/aim_data/aim_3.avi"  # 修改为您的视频文件路径
    
    # 可选：指定输出视频路径
    output_path = None  # 如果需要保存结果视频，可以设置为 "output.avi"
    
    try:
        # 创建颜色检测器
        detector = ColorDetector(
            video_path=video_path,
            output_path=output_path
        )
        
        # 运行检测
        detector.run()
        
    except KeyboardInterrupt:
        print("\n程序已被用户终止")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == '__main__':
    main()