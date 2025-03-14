#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class VideoPublisher:
    def __init__(self, video_path):
        # 初始化 ROS 节点
        rospy.init_node('video_publisher', anonymous=True)

        # 创建发布者，话题名为 /camera_f/color/image_raw
        self.image_pub = rospy.Publisher('/camera_f/color/image_raw', Image, queue_size=10)

        # 创建 OpenCV 视频读取对象
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            rospy.logerr("无法打开视频文件: " + video_path)
            raise RuntimeError("无法打开视频文件")

        # 创建 OpenCV 和 ROS 之间的转换工具
        self.bridge = CvBridge()

        # 读取视频的帧率，计算帧间隔
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps > 0:
            self.frame_delay = 1.0 / self.fps  # 计算每帧间隔
        else:
            self.frame_delay = 0.03  # 如果无法获取帧率，默认 30 FPS

        rospy.loginfo(f"视频 {video_path} 读取成功，FPS: {self.fps:.2f}")

    def publish_frames(self):
        """读取视频帧并发布到 ROS 话题"""
        rate = rospy.Rate(30)  # 设定发布频率

        while not rospy.is_shutdown() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                rospy.loginfo("视频播放完毕")
                break

            # 将 OpenCV 图像转换为 ROS Image 消息
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            # 发布消息
            self.image_pub.publish(ros_image)
            rospy.loginfo("已发布一帧图像")

            # 控制发布频率
            time.sleep(self.frame_delay)
            rate.sleep()

        # 释放视频资源
        self.cap.release()
        rospy.loginfo("视频发布器关闭")

if __name__ == "__main__":
    try:
        video_path = "/home/jhw/code/arm_ik-main/src/data/aim_data/aim_3.avi"  # 替换为你的 AVI 文件路径
        publisher = VideoPublisher(video_path)
        publisher.publish_frames()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"程序出错: {str(e)}")
