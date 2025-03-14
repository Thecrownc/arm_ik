import sys
import cv2
import numpy as np
import os
from collections import deque
import open3d as o3d
from scipy.spatial import KDTree
from datetime import datetime
from .detect_rgb import yolo_detect,pre_load_model
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import rospy
from sensor_msgs.msg import Image,PointCloud2
from cv_bridge import CvBridge
import ros_numpy
from threading import Thread,Lock,Event
import time
import signal
import tf
from tf.transformations import quaternion_from_matrix
from geometry_msgs.msg import TransformStamped,Point
import tf2_ros
from visualization_msgs.msg import Marker,MarkerArray

sys.path.append("/home/abc/arm_ik/src/arm_ik/scripts")

get_kps_flag = False

class runtime():
    def __init__(self):
        return
    
    def ros_thread(self):
        return

def save_data(rgb_image,pcd_data,kps,kps_3d,plane_center,normal):
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # 在当前路径生成文件夹
    folder_name = f"./{timestamp}_data"
    os.makedirs(folder_name, exist_ok=True)  # 如果文件夹已存在，则不会报错

    # 定义文件路径
    pcd_file_path = os.path.join(folder_name, "point_cloud.pcd")
    rgb_image_path = os.path.join(folder_name, "rgb_image.jpg")
    txt_path = os.path.join(folder_name, "kps_3d.txt")

    # 保存点云数据到 PCD 文件
    o3d.io.write_point_cloud(pcd_file_path, pcd_data)

    # 保存 RGB 图像
    cv2.imwrite(rgb_image_path, rgb_image)

    # 使用 "a" 模式打开文件，追加内容
    with open(txt_path, "a") as file:
        file.write("kps:" + "\n")  # 添加新行并换行
    with open(txt_path, "a") as file:
        file.write(", ".join(map(str, kps)) + "\n")  # 每次追加一行新内容
    with open(txt_path, "a") as file:
        file.write("kps_3d:" + "\n")  # 添加新行并换行
    with open(txt_path, "a") as file:
        file.write(", ".join(map(str, kps_3d)) + "\n")  # 每次追加一行新内容
    with open(txt_path, "a") as file:
        file.write("plane_center:" + "\n")  # 添加新行并换行
    with open(txt_path, "a") as file:
        file.write(", ".join(map(str,plane_center)) + "\n")  # 每次追加一行新内容
    with open(txt_path, "a") as file:
        file.write("normal:" + "\n")  # 添加新行并换行
    with open(txt_path, "a") as file:
        file.write(", ".join(map(str,normal)) + "\n")  # 每次追加一行新内容

    print(f"Data saved in folder: {folder_name}")


def ros_to_open3d(pcl_msg):
    """
    将 ROS PointCloud2 消息转换为 Open3D 格式点云。
    """
    # 使用 ros_numpy 提取点云数据
    pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl_msg, remove_nans=True)
    
    # 转换为 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)
    #print(pcd)
    
    return pcd
    

def point_cloud_to_depth_image(points, fx, fy, cx, cy, width, height):
    """
    将点云映射到深度图像。

    :param pcd: open3d.geometry.PointCloud, 点云对象
    :param fx, fy, cx, cy: float, 相机内参
    :param width, height: int, 深度图像的宽和高
    :return: depth_image, 深度图像 (H, W)
    """

    # 初始化深度图像
    depth_image = np.zeros((height, width), dtype=np.float32)

    # 投影到像素坐标
    u = (points[:, 0] * fx / points[:, 2] + cx).astype(int)
    v = (points[:, 1] * fy / points[:, 2] + cy).astype(int)

    # 过滤无效点（像素坐标超出范围或深度 <= 0）
    valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (points[:, 2] > 0)
    u, v, z = u[valid_mask], v[valid_mask], points[:, 2][valid_mask]

    # 填充深度图像（存储最小深度以避免遮挡问题）
    for x, y, depth in zip(u, v, z):
        if depth_image[y, x] == 0 or depth < depth_image[y, x]:
            depth_image[y, x] = depth*1000

    return depth_image

def draw_keypoints_on_image(image, kps, kps_3d):
    """
    在RGB图像上绘制关键点和3D坐标。

    :param image: np.ndarray, 原始图像
    :param kps: list, 2D关键点 [(x1, y1), ...]
    :param kps_3d: list, 对应的3D空间坐标 [(x, y, z), ...]
    """
    for kp, kp_3d in zip(kps, kps_3d):
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        text = f"({kp_3d[0]:.2f}, {kp_3d[1]:.2f}, {kp_3d[2]:.2f})"
        cv2.putText(image, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)


# def map_2d_to_3d(kps, depth_image, fx, fy, cx, cy):
#     height, width = depth_image.shape
#     depth_image = depth_image / 1000.0
#     points = []
#
#     coo = []
#     for kp in kps:
#         coo.append((int(kp[0]), int(kp[1])))
#
#
#     kps_3d = []
#     for u in range(0, width):
#         for v in range(0, height):
#             for coo_x, coo_y in coo:
#                 if coo_x==u and coo_y==v:
#                     # print(u,v)
#                     x = (u - cx) * depth_image[v][u] / fx
#                     y = (v - cy) * depth_image[v][u] / fy
#                     kps_3d.append((x, y, depth_image[v][u]))
#
#
#     return kps_3d

def map_2d_to_3d_kdtree(kps, pcd, fx, fy, cx, cy, k=1):
    """
    使用 KDTree 将 2D 关键点映射到 3D 点云坐标。

    :param kps: list, 2D关键点 [(x1, y1), (x2, y2), ...]
    :param pcd: open3d.geometry.PointCloud, 点云对象
    :param fx: float, 相机内参 fx
    :param fy: float, 相机内参 fy
    :param cx: float, 相机内参 cx
    :param cy: float, 相机内参 cy
    :param k: int, 最近邻数量，用于加权平均
    :return: list, 每个关键点对应的 3D 坐标 [(x, y, z), ...]
    """
    # 提取点云的 3D 坐标
    points = np.asarray(pcd.points)

    # 过滤无效点
    z_values = points[:, 2]
    valid_indices = z_values > 0
    points = points[valid_indices]
    z_values = z_values[valid_indices]

    # 投影到像素平面
    u = (points[:, 0] * fx / z_values) + cx
    v = (points[:, 1] * fy / z_values) + cy
    pixel_coords = np.column_stack((u, v))

    # 检查 pixel_coords 是否有效
    if np.any(np.isnan(pixel_coords)) or np.any(np.isinf(pixel_coords)):
        raise ValueError("pixel_coords 包含 NaN 或 inf 值，请检查点云数据和深度值是否正确。")

    # 构建 KDTree
    tree = KDTree(pixel_coords)

    # 映射 2D 关键点到 3D
    kps_3d = []
    for kp in kps:
        dist, indices = tree.query(kp, k=k)  # 找到 k 个最近邻
        if k == 1:
            indices = [indices]  # 保持一致性
        neighbors = points[indices]

        # 加权平均计算3D坐标
        if k > 1:
            weights = 1 / (dist + 1e-6)  # 距离越近权重越大
            weights /= weights.sum()
            kp_3d = np.average(neighbors, axis=0, weights=weights)
        else:
            kp_3d = neighbors[0]

        kps_3d.append(kp_3d)

    return kps_3d


def visualize_kps_on_depth(depth_image, kps):
    """
    可视化2D关键点在深度图像上的位置。

    :param depth_image: np.ndarray, 深度图像 (H, W)
    :param kps: list, 2D关键点 [(x1, y1), (x2, y2), ...]
    """
    # 将深度图归一化到0-255范围以进行可视化
    normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 转换为三通道图像
    depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

    # 在深度图上绘制关键点
    for kp in kps:
        x, y = int(kp[0]), int(kp[1])
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            cv2.circle(depth_colormap, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    # 使用matplotlib显示图像
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    plt.title("2D Keypoints on Depth Image")
    plt.axis("off")
    plt.show()

def visualize_kps_and_pcd(kps_3d, pcd):
    # 将关键点转换为 Open3D 点云对象
    kps_pcd = o3d.geometry.PointCloud()
    kps_pcd.points = o3d.utility.Vector3dVector(kps_3d)

    # 给关键点设置不同的颜色（例如红色）
    kps_pcd.paint_uniform_color([1, 0, 0])  # 红色

    # 给点云设置颜色（例如灰色）
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

    # 可视化点云和关键点
    o3d.visualization.draw_geometries([pcd, kps_pcd],
                                      window_name="3D Keypoints and Point Cloud",
                                      width=800,
                                      height=600,
                                      point_show_normal=False)
# def expand_kps(kps,  range_size=10):
#     """
#     将每个关键点周围指定范围的像素点加入结果列表。
#
#     :param kps: np.ndarray, 关键点数组，形状为 (N, 2)，每行表示 (x, y)。
#     :param depth_image: np.ndarray, 深度图像。
#     :param range_size: int, 扩展范围，以像素为单位（默认30像素）。
#     :return: list, 每个关键点周围的像素点坐标 [(x, y), ...]。
#     """
#     height, width = 480,640
#     kps_new = []
#
#     for kp in kps:
#         x, y = int(kp[0]), int(kp[1])  # 当前关键点的整数坐标
#
#         # 确保扩展范围不超出图像边界
#         x_min = max(0, x - range_size)
#         x_max = min(width - 1, x + range_size)
#         y_min = max(0, y - range_size)
#         y_max = min(height - 1, y + range_size)
#
#         # 遍历扩展区域的所有像素点
#         for nx in range(x_min, x_max + 1):
#             for ny in range(y_min, y_max + 1):
#                 kps_new.append((nx, ny))
#
#     return kps_new


def fit_plane_to_points(kps_3d):
    """
    求最佳平面，使关键点到平面的距离最短。

    :param kps_3d: list of (x, y, z), 关键点的 3D 坐标
    :return: (a, b, c, d), 平面方程的系数
    """
    points = np.array(kps_3d)  # 转换为 NumPy 数组
    assert points.shape[0] >= 3, "至少需要 3 个点来定义平面。"

    # 初始法向量和 d
    centroid = points.mean(axis=0)  # 关键点的质心
    initial_guess = np.array([0, 0, 1, -centroid[2]])  # 假设初始法向量接近于 z 轴方向

    # 定义目标函数：所有点到平面的平方距离和
    def distance_sum(params):
        a, b, c, d = params
        norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm
        return np.sum(distances ** 2)

    # 优化求解平面参数
    result = minimize(distance_sum, initial_guess, method='L-BFGS-B')

    # 检查优化结果
    if result.success:
        # 确保法向量归一化
        a, b, c, d = result.x
        norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        return a / norm, b / norm, c / norm, d / norm
    else:
        raise RuntimeError("平面拟合优化失败")


def point_to_plane_distance(point, plane_params):
    """
    计算点到平面的距离。

    :param point: (x, y, z), 点的 3D 坐标
    :param plane_params: (a, b, c, d), 平面方程的系数
    :return: 距离值
    """
    a, b, c, d = plane_params
    x, y, z = point
    norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)
    distance = np.abs(a * x + b * y + c * z + d) / norm
    return distance

def get_plane_normal(plane_params):
    """
    从平面参数中提取法向量，并归一化。

    :param plane_params: (a, b, c, d), 平面方程的系数
    :return: 法向量 (nx, ny, nz)
    """
    a, b, c, _ = plane_params
    norm = np.sqrt(a**2 + b**2 + c**2)
    return a / norm, b / norm, c / norm
    

def shrink_kps(kps, r):
    global get_kps_flag
    """
    求关键点中心，并将关键点以中心为基准缩小 r 倍。

    :param kps: list of tuples, 包含四个关键点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param r: float, 缩小倍数，必须大于0
    :return: list of tuples, 缩小后的关键点 [(x1', y1'), (x2', y2'), (x3', y3'), (x4', y4')]
    """
    if len(kps) != 4:
        get_kps_flag = False
        rospy.logerr("关键点列表必须包含四个点。")
        return None

    if r <= 0:
        rospy.logerr("缩小倍数 r 必须大于 0。")

    # 计算关键点的中心
    kps_array = np.array(kps)
    center = kps_array.mean(axis=0)

    # 缩小关键点
    shrunk_kps = []
    for x, y in kps:
        x_new = center[0] + (x - center[0]) / r
        y_new = center[1] + (y - center[1]) / r
        shrunk_kps.append((x_new, y_new))

    return shrunk_kps

def move_point_along_normal(center_point, plane_normal, d):
    """
    沿着法向量的反方向，移动点 `center_point` 距离 `d`。

    :param center_point: np.ndarray, 点的坐标 (x, y, z)
    :param plane_normal: np.ndarray, 法向量 (nx, ny, nz)
    :param d: float, 移动的距离
    :return: np.ndarray, 新的点的坐标
    """
    # 归一化法向量
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # 沿着法向量反方向移动点
    new_point = center_point - plane_normal * d

    return new_point

def mark_kps_on_image(image, kps, output_path='None'):
    """
    在RGB图像上标记关键点，并保存到指定路径。

    :param rgb_image_path: str, 输入的 RGB 图像路径
    :param kps: list of tuples, 包含关键点 [(x1, y1), (x2, y2), ...]
    :param output_path: str, 输出标记后图像的保存路径
    """
    # 加载图像

    # 遍历关键点并标记
    for idx, (x, y) in enumerate(kps):
        # 绘制关键点
        cv2.circle(image, (int(x), int(y)), radius=8, color=(0, 255, 0), thickness=-1)
        # 绘制编号标签
        #cv2.putText(image, f"{idx+1}", (int(x) + 10, int(y) - 10),
                    #cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 255), thickness=2)

    #
    cv2.imwrite(output_path, image)
    #cv2.imshow('Image', image)
    print(f"标记后的图像已保存到: {output_path}")
  
def are_points_coplanar(points, threshold: float = 1e-6) -> bool:
    points = np.array(points)
    """
    检查四个3D点是否共面。

    参数:
        points (np.ndarray): 形状为 (4, 3) 的数组，表示四个点的坐标。
        threshold (float): 判断共面的阈值，默认为1e-6。

    返回:
        bool: 如果共面返回 True，否则返回 False。
    """
    if points.shape != (4, 3):
        raise ValueError("points 数组必须是形状 (4, 3)")

    # 计算法向量
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)

    # 检查法向量是否为零向量
    if np.linalg.norm(normal) == 0:
        raise ValueError("前三个点共线，无法确定平面")

    # 计算第四个点到平面的距离
    v3 = points[3] - points[0]
    distance = abs(np.dot(v3, normal)) / np.linalg.norm(normal)

    return distance < threshold


bridge = CvBridge()
rgb_img = None
rgb_img_deque = deque(maxlen=5)
point_img = None
point_img_deque = deque(maxlen=5)

def image_callback(msg):
    global rgb_img
    try:
        frame = bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
        rgb_img = frame
        rgb_img_deque.append(rgb_img)
    except Exception as e:
        print("cvbridge convert msg to cv2 error: ",e)
    
def point_cloud_callback(msg):
    global point_img
    """
    回调函数，处理点云消息并转换为 Open3D 格式。
    """
    #rospy.loginfo("Received point cloud data")
    
    # 转换点云
    point_img = ros_to_open3d(msg)
    point_img_deque.append(point_img)
    #print("point_img: ",point_img)
    #print(point_img)
    
    # 可视化或保存点云
    #o3d.visualization.draw_geometries([point_img])
    #o3d.io.write_point_cloud("output.pcd", pcd)
    #rospy.loginfo("Saved point cloud to output.pcd")

def depth_callback(msg):
    global depth_img
    try:
        frame = bridge.imgmsg_to_cv2(msg,desired_encoding="32FC1")
        depth_img = frame
    except Exception as e:
        print("cvbridge convert msg to cv2 error: ",e)

def compute_quaternion_from_normal(nx, ny, nz):
    """
    根据法向量 (nx, ny, nz) 计算四元数
    """
    # 假设法向量为 z 轴，x 轴可以任意选取，但要和 z 垂直
    z_axis = np.array([nx, ny, nz])
    z_axis = z_axis / np.linalg.norm(z_axis)  # 归一化

    # 构造 x 轴：选取一个与 z 轴不平行的向量
    temp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(temp, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # 构造 y 轴
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 组合成旋转矩阵
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = x_axis
    rotation_matrix[:3, 1] = y_axis
    rotation_matrix[:3, 2] = z_axis

    # 将旋转矩阵转换为四元数
    quaternion = quaternion_from_matrix(rotation_matrix)
    return quaternion

ros_event = Event()
transfrom_global = None
transform_grab_globa = None
marker_global = None

def start_caculate():
    global get_kps_flag,rgb_img,point_img
    global transfrom_global,transform_grab_globa,marker_global
    rgb_img = None
    point_img = None
    rospy.loginfo("waiting for rgb img")
    while not rospy.is_shutdown() and (rgb_img is None or point_img is None):  
        time.sleep(1)
        print("rgb_img type: ",type(rgb_img)," point type: ",type(point_img))
        if(ros_event.is_set()):
            rospy.loginfo("shuting down")
            return False

    rospy.loginfo("getimg!")
    # realsense相机内参
    #fx, fy = 428.725, 428.725
    #cx, cy = 423.512, 241.212
    
    # deeya相机内参
    #fx, fy = 453.39,  453.39
    #cx, cy = 315.65,  240.48
    
    # dabai相机内参
    #fx, fy = 453.354,  453.354
    #cx, cy = 326.444,  238.468
    fx_color, fy_color = 453.354, 453.354
    cx_color, cy_color = 319.444, 214.468
    
    fx, fy = 477.174, 477.174
    cx, cy = 319.029, 196.973

    # 加载RGB图像
    frame = rgb_img

    start_time = time.time()

    center_list = []
    normal_list = []
    while not rospy.is_shutdown() and len(center_list) < 12:
        # YOLO检测关键点 (2D坐标)
        #cv2.imshow("raw pic",frame)
        frame = rgb_img_deque.pop()
        kps = yolo_detect(frame, './yolo_detect/weights/110.pt')  # 示例返回 [(x1, y1), ...]
        # visualize
        visualize_frame = np.copy(frame)
        mark_kps_on_image(visualize_frame,kps,'./result.jpg')
        print("kps: ",kps)
        kps = shrink_kps(kps,1.5)
        if kps==None:
            error_count=error_count+1
            if error_count>20:
                rospy.loginfo("error count > 10,shuting down")
                return False
            continue
        kps_plane = shrink_kps(kps,0.25)

        #/camera_f/depth/color/points
        # 加载点云
        pcd = point_img_deque.pop()
        #pcd = o3d.io.read_point_cloud(pcd_path,
        #                          remove_nan_points=True,
        #                            remove_infinite_points=True)

        points = np.asarray(point_img.points)

        # 映射点云到深度图像
        #depth_image = point_cloud_to_depth_image(points, fx, fy, cx, cy, 640, 480)
        #visualize_kps_on_depth(depth_image, kps)

        # 将2D关键点映射到3D
        kps_3d = map_2d_to_3d_kdtree(kps, pcd, fx, fy, cx, cy, k=3)
        kps_plane_3d = map_2d_to_3d_kdtree(kps_plane, pcd, fx, fy, cx, cy, k=3)
        # is_coplanar = are_points_coplanar(kps_3d, threshold=0.003)
        is_coplanar = are_points_coplanar(kps_plane_3d, threshold=0.003)

        if is_coplanar == False:
            rospy.logwarn("**********is coplanar False!!!!************")
            continue

        # visualize_kps_and_pcd(kps_3d, pcd)

        #print("---kps 3d---: ",kps_3d)
        plane_params = fit_plane_to_points(kps_plane_3d)
        plane_normal = get_plane_normal(plane_params)
        kps_3d  = np.array(kps_3d)
    
        plane_center = kps_3d.mean(axis=0)  # 平面中心设置为关键点质心
        center_list.append(plane_center)
        normal_list.append(plane_normal)

    plane_center_array = np.array(center_list)
    normal_array = np.array(normal_list)
    plane_center_mean = plane_center_array.mean(axis=0)
    normal_mean = normal_array.mean(axis=0)
    robot_point = move_point_along_normal(plane_center_mean, normal_mean, 0.02)

    # test wrong param
    # kps_3d_color = map_2d_to_3d_kdtree(kps, pcd, fx_color, fy_color, cx_color, cy_color, k=3)
    # kps_3d_color  = np.array(kps_3d_color)
    # plane_center_color = kps_3d_color.mean(axis=0)
    # robot_point = plane_center_color   #robot_point = plane_center
    end_time = time.time()
    print("法向量 (nx, ny, nz):", plane_normal)
    print("x,y,z: ",robot_point)
    print(f"caculate cost time: {end_time-start_time:.6f} sec")
    qua = compute_quaternion_from_normal(plane_normal[0],plane_normal[1],plane_normal[2])
    print("*******qua********: ",qua)

    save_data(rgb_img,point_img,kps,kps_3d,plane_center,plane_normal)

    # 设置Transform消息
    transform = TransformStamped()

    # 填充变换信息
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "camera_link"  
    transform.child_frame_id = "yaban_link"  
    transform.transform.translation.x = robot_point[0]
    transform.transform.translation.y = robot_point[1]
    transform.transform.translation.z = robot_point[2]
    transform.transform.rotation.x = qua[0]
    transform.transform.rotation.y = qua[1]
    transform.transform.rotation.z = qua[2]
    transform.transform.rotation.w = qua[3]
    
    # 设置Transform消息
    transform_grab = TransformStamped()

    # 填充变换信息
    transform_grab.header.stamp = rospy.Time.now()
    transform_grab.header.frame_id = "yaban_link"  
    transform_grab.child_frame_id = "grab_link"  
    transform_grab.transform.translation.x = 0
    transform_grab.transform.translation.y = 0

    # transform_grab.transform.translation.z = -0.13
    transform_grab.transform.translation.z = -0.12

    transform_grab.transform.rotation.x = 0
    transform_grab.transform.rotation.y = 0
    transform_grab.transform.rotation.z = 0
    transform_grab.transform.rotation.w = 1

    # 创建 MarkerArray 消息
    marker_array = MarkerArray()

    # 定义多个球体的位置、大小、颜色
    sphere_positions = [
        kps_3d[0],  # 球体1位置 (x, y, z)
        kps_3d[1],  # 球体2位置 (x, y, z)
        kps_3d[2],  # 球体3位置 (x, y, z)
        kps_3d[3]
    ]

    for i, pos in enumerate(sphere_positions):
        # 创建每个 Marker
        marker = Marker()
        marker.header.frame_id = "camera_link"  # 使用的参考坐标系
        marker.header.stamp = rospy.Time.now()
        marker.ns = "spheres_namespace"
        marker.id = i  # 每个 Marker 的 id 必须唯一
        marker.type = Marker.SPHERE  # SPHERE 类型
        marker.action = Marker.ADD

        # 设置位置
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # 设置大小 (缩放)
        marker.scale.x = 0.01  # 半径
        marker.scale.y = 0.01
        marker.scale.z = 0.01

        # 设置颜色
        marker.color.r = 0.0
        marker.color.g = 1.0 / (i + 1)  # 不同球体设置不同颜色
        marker.color.b = 1.0
        marker.color.a = 1.0  # 不透明

        # 生命周期 (-1 表示永久，>0 表示持续时间)
        marker.lifetime = rospy.Duration(0.1)

        # 添加到 MarkerArray 中
        marker_array.markers.append(marker)
        # 添加箭头
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "camera_link"
        arrow_marker.header.stamp = rospy.Time.now()

        arrow_marker.ns = "arrow_namespace"
        arrow_marker.id = 100  # 确保 ID 不与其他 Marker 冲突
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD

        # 设置箭头的起点和终点
        arrow_marker.points = []
        start_point = Point(robot_point[0], robot_point[1], robot_point[2])  # 箭头起点
        end_point = Point(robot_point[0]+0.2*plane_normal[0], robot_point[1]+0.2*plane_normal[1], robot_point[2]+0.2*plane_normal[2])    # 箭头终点
        arrow_marker.points.append(start_point)
        arrow_marker.points.append(end_point)

        # 设置箭头的颜色
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 1.0

        # 设置箭头的大小
        arrow_marker.scale.x = 0.01  # 箭头杆的宽度
        arrow_marker.scale.y = 0.02  # 箭头头的宽度
        arrow_marker.scale.z = 0.02  # 箭头头的长度

        arrow_marker.lifetime = rospy.Duration(0.1)
        # 添加到 MarkerArray 中
        marker_array.markers.append(arrow_marker)

        transfrom_global = transform
        transform_grab_globa = transform_grab
        marker_global = marker_array

        # wait for tf to be published
        time.sleep(0.2)
        return True


def pub_tf_thread():

    global get_kps_flag
    global transfrom_global,transform_grab_globa,marker_global

    br = tf2_ros.TransformBroadcaster()
    while not rospy.is_shutdown():
        if transfrom_global is None or transform_grab_globa is None or marker_global is None:
            time.sleep(0.1)
            continue

    	# 广播变换
        transfrom_global.header.stamp = rospy.Time.now()
        transform_grab_globa.header.stamp = rospy.Time.now()
        br.sendTransform(transfrom_global)
        br.sendTransform(transform_grab_globa)
        # 发布 MarkerArray
        for marker in marker_global.markers:
            marker.header.stamp = rospy.Time.now()
        marker_pub.publish(marker_global)
        time.sleep(0.1)

        if get_kps_flag == False:
            get_kps_flag = True
            

ros_thread = Thread(target=pub_tf_thread)

def start_rgb_detector():
    global rgb_sub,point_sub,marker_pub
    rospy.loginfo("Start yolo detect node")
    rgb_sub = rospy.Subscriber("/camera_f/color/image_raw",Image,image_callback)
    point_sub = rospy.Subscriber("/camera_f/depth/points", PointCloud2, point_cloud_callback)
    marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
    ros_thread.start()
    pre_load_model('./yolo_detect/weights/110.pt')
    
def signal_handler(signal,frame):
    ros_event.set()
    rospy.signal_shutdown("shuting down")

# 主流程
if __name__ == "__main__":
    signal.signal(signal.SIGINT,signal_handler)
    rospy.init_node("yolo_detect")
    rospy.loginfo("Start yolo detect node")
    rgb_sub = rospy.Subscriber("/camera_f/color/image_raw",Image,image_callback)
    point_sub = rospy.Subscriber("/camera_f/depth/points", PointCloud2, point_cloud_callback)
    marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
    #depth_sub = rospy.Subscriber("/camera_f/depth/image_rect_raw",Image,depth_callback)
    ros_thread.start()
    rospy.spin()