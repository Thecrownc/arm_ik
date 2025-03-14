#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

class CameraSaver:
    def __init__(self):
        rospy.init_node('camera_saver', anonymous=True)

        # 设置图像和点云保存路径
        self.image_save_path = "./image.jpg"
        self.pointcloud_save_path = "./pointcloud.pcd"

        # 初始化 CvBridge
        self.bridge = CvBridge()

        # 订阅图像话题
        self.image_sub = rospy.Subscriber("/camera_f/color/image_raw", Image, self.image_callback)

        # 订阅点云话题
        self.pointcloud_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pointcloud_rgb_callback)

        # 标志变量，用于防止重复保存
        self.image_saved = False
        self.pointcloud_saved = False

    def image_callback(self, msg):
        if not self.image_saved:  # 确保只保存一次
            try:
                # 将 ROS 图像消息转换为 OpenCV 图像
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # 保存图像到磁盘
                cv2.imwrite(self.image_save_path, cv_image)
                rospy.loginfo(f"Image saved to {self.image_save_path}")
                self.image_saved = True
            except Exception as e:
                rospy.logerr(f"Failed to save image: {e}")

    def pointcloud_callback(self, msg):
        if not self.pointcloud_saved:  # 确保只保存一次
            try:
                # 将 PointCloud2 消息转换为点云数据
                points = []
                for point in pc2.read_points(msg, skip_nans=True):
                    points.append([point[0], point[1], point[2]])

                # 创建 Open3D 点云对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(points))

                # 保存点云到 PCD 文件
                o3d.io.write_point_cloud(self.pointcloud_save_path, pcd)
                rospy.loginfo(f"PointCloud saved to {self.pointcloud_save_path}")
                self.pointcloud_saved = True
            except Exception as e:
                rospy.logerr(f"Failed to save pointcloud: {e}")
              
    def pointcloud_rgb_callback(self, msg):
        if not self.pointcloud_saved:  # 确保只保存一次
            try:
                # 提取点云数据（包括坐标和颜色）
                points = []
                colors = []

                # 读取点云数据并处理颜色信息
                for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
                    # 坐标
                    points.append([point[0], point[1], point[2]])

                    # 颜色（提取 RGB 信息）
                    rgb = int(point[3])  # `rgb` 是一个 32 位的整数（0xRRGGBB）
                    r = ((rgb >> 16) & 0xFF) / 255.0  # 提取红色通道并归一化
                    g = ((rgb >> 8) & 0xFF) / 255.0   # 提取绿色通道并归一化
                    b = (rgb & 0xFF) / 255.0          # 提取蓝色通道并归一化
                    colors.append([r, g, b])          # 将颜色添加到列表


                # 创建 Open3D 点云对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))  # 添加颜色信息

                # 保存带颜色的点云到 PCD 文件
                o3d.io.write_point_cloud(self.pointcloud_save_path, pcd)
                rospy.loginfo(f"PointCloud with color saved to {self.pointcloud_save_path}")
                self.pointcloud_saved = True
            except Exception as e:
                rospy.logerr(f"Failed to save pointcloud: {e}")
    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        camera_saver = CameraSaver()
        camera_saver.spin()
    except rospy.ROSInterruptException:
        pass

