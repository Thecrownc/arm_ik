import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

# 定义回调函数
def image_callback(msg):
    global video_writer, bridge
   # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 将ROS图像消息转换为OpenCV图像
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")
        return
    
    # 将OpenCV图像写入视频文件
    video_writer.write(cv_image)

# 初始化 ROS 节点
rospy.init_node('save_camera_video', anonymous=True)

# 设置视频写入器（保存为视频文件）
frame_width = 640  # 根据摄像头分辨率调整
frame_height = 480
frame_rate = 30  # 每秒30帧

# 创建一个 VideoWriter 对象，指定输出文件名、编码器、帧率和分辨率
video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))

# 创建 CvBridge 对象，用于转换 ROS 图像消息
bridge = CvBridge()

# 订阅摄像头话题
camera_topic = "/camera_f/color/image_raw"  # 根据你的摄像头话题来调整
rospy.Subscriber(camera_topic, Image, image_callback)

# 保持 ROS 节点运行
rospy.spin()

# 关闭视频写入器
video_writer.release()

