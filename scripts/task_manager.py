#!/bin/python3
import sys
import argparse
import rospy
import time
from threading import Thread,Lock,Event
import signal

from std_msgs.msg import String
from ik import ik_caculator
from arm_hw import arm_hw
from yolo_detect import map1

class task_manager:
    def __init__(self,args):
        self.args = args
        # 初始化机械臂硬件控制器
        self.arm_hw = arm_hw()
        # 初始化逆解求解器
        self.ik_caculator = ik_caculator()

        map1.start_rgb_detector()

        self.cmd_sub = rospy.Subscriber("/task_cmd",String,self.task_cmd_callback)
        self.state_pub = rospy.Publisher('/task_states', String, queue_size=10)
        self.cmd_type = None
        
        self.task_thread = Thread(target=self.run_task)
        self.task_states_thread = Thread(target=self.pub_state)
        self.update_thread_stop_event = Event()
        self.shutdown = False
        self.task_state = "idle"
    
    def pub_state(self):

        while not rospy.is_shutdown():
            msg = String()
            msg.data = self.task_state
            self.state_pub.publish(msg)
            time.sleep(0.2)

    def run_task(self):

        while not rospy.is_shutdown():
            log_flag = False
            # 等待新任务进来
            while not rospy.is_shutdown() and self.cmd_type is None:
                if log_flag == False:
                    rospy.logwarn("waiting for cmd topic to start task")
                    self.task_state = "idle"
                    log_flag = True
                time.sleep(0.1)
            
            if rospy.is_shutdown():
                return
            
            rospy.loginfo(f"cmd_type: {self.cmd_type}")

            if self.cmd_type == "init_left":
                # 机械臂归位
                rospy.loginfo("Start init task")
                self.task_state = "run_init_left"
                self.ik_caculator.init_arm(puppet="left")
                self.cmd_type = None
                rospy.loginfo("******* end *******")
                continue
            elif self.cmd_type == "init_right":
                # 机械臂归位
                rospy.loginfo("Start init task")
                self.task_state = "run_init_right"
                self.ik_caculator.init_arm(puppet="right")
                self.cmd_type = None
                rospy.loginfo("******* end *******")
                continue
            elif self.cmd_type == "right_open_gripper":
                self.task_state = "run_right_open_gripper"
                self.arm_hw.gripper_control(gripper="open",puppet="right")
                self.cmd_type = None
                continue
            elif self.cmd_type == "right_close_gripper":
                self.task_state = "run_right_close_gripper"
                self.arm_hw.gripper_control(gripper="close",puppet="right")
                self.cmd_type = None
                continue
            elif self.cmd_type == "left_open_gripper":
                self.task_state = "run_left_open_gripper"
                self.arm_hw.gripper_control(gripper="open",puppet="left")
                self.cmd_type = None
                continue
            elif self.cmd_type == "left_close_gripper":
                self.task_state = "run_left_close_gripper"
                self.arm_hw.gripper_control(gripper="close",puppet="left")
                self.cmd_type = None
                continue
            
            if self.cmd_type == "right_grab_green" or self.cmd_type == "right_grab_red" \
            or self.cmd_type == "left_grab_green" or self.cmd_type == "left_grab_red" or self.cmd_type == "test":
                result = map1.start_caculate()

            counter = 0
            while not rospy.is_shutdown() and map1.get_kps_flag is False:
                counter+=1
                if counter>5:
                    break
                time.sleep(1)
                rospy.loginfo("wait rgb detector")
            
            rospy.loginfo("-------start--------")
            
            # if args.init:
            #     # 机械臂归位
            #     self.ik_caculator.init_arm()
            
            # right arm 
            if self.cmd_type == "right_grab_green":
                self.task_state = "run_grab_green"
                self.grab(color="green",puppet="right")
            elif self.cmd_type == "right_grab_red":
                self.task_state = "run_grab_red"
                self.grab(color="red",puppet="right")
            elif self.cmd_type == "right_put_green":
                self.task_state = "run_put_green"
                self.put(color="green",puppet="right")
            elif self.cmd_type == "right_put_red":
                self.task_state = "run_put_red"
                self.put(color="red",puppet="right")
            
            # left arm
            if self.cmd_type == "left_grab_green":
                self.task_state = "run_grab_green"
                self.grab(color="green",puppet="left")
            elif self.cmd_type == "left_grab_red":
                self.task_state = "run_grab_red"
                self.grab(color="red",puppet="left")
            elif self.cmd_type == "left_put_green":
                self.task_state = "run_put_green"
                self.put(color="green",puppet="left")
            elif self.cmd_type == "left_put_red":
                self.task_state = "run_put_red"
                self.put(color="red",puppet="left")
            elif self.cmd_type == "test":
                self.test()
            
            # 机械臂归零
            self.ik_caculator.init_arm()

            # clear cmd_type
            self.cmd_type = None
            rospy.loginfo("******* end *******") 

    def grab(self,color,puppet):
        # 运动到压板预瞄点
        self.arm_hw.fold_arm(puppet=puppet)
        self.ik_caculator.run(color=color,puppet=puppet)
        time.sleep(2.2)
        
        # 打开夹爪
        self.arm_hw.gripper_control(gripper="open",puppet=puppet)

        # 机械臂前伸
        self.ik_caculator.run(step_list=[0.085,0,0],color=color,puppet=puppet)
        time.sleep(2.1)

        # 闭合夹爪
        self.arm_hw.gripper_control(gripper="close",puppet=puppet)

        # 机械臂后伸
        self.ik_caculator.run(step_list=[0.055,0,0],color=color,puppet=puppet)
        time.sleep(2.1)

        # 旋转压板
        self.arm_hw.motor_add_control(joint=5,angle=-1.57,puppet=puppet)

        # 打开夹爪
        self.arm_hw.gripper_control(gripper="open",puppet=puppet)
        
        # 回到预瞄点
        self.ik_caculator.run(color=color,puppet=puppet)
        time.sleep(2.1)
        self.arm_hw.fold_arm(puppet=puppet)

    def put(self,color,puppet):
        # 运动到压板预瞄点
        self.arm_hw.fold_arm(puppet=puppet)
        self.ik_caculator.run(color=color,puppet=puppet)
        time.sleep(2.2)
        
        # 打开夹爪
        self.arm_hw.gripper_control(gripper="open",puppet=puppet)

        # 旋转压板
        self.arm_hw.motor_add_control(joint=5,angle=-1.57,puppet=puppet)

        # 机械臂前伸
        self.arm_hw.lock_rotation_flag=True
        self.ik_caculator.run(step_list=[0.08,0,0],color=color,puppet=puppet)
        time.sleep(2.1)

        # 闭合夹爪
        self.arm_hw.gripper_control(gripper="close",puppet=puppet)

        # 机械臂后伸
        self.arm_hw.lock_rotation_flag=True
        self.ik_caculator.run(step_list=[0.045,0,0],color=color,puppet=puppet)
        time.sleep(2.1)

        # 旋转压板
        self.arm_hw.motor_add_control(joint=5,angle=1.57,puppet=puppet)

        # 打开夹爪
        self.arm_hw.gripper_control(gripper="open",puppet=puppet)
        
        # 回到预瞄点
        self.ik_caculator.run(color=color,puppet=puppet)
        time.sleep(2.1)
        self.arm_hw.fold_arm(puppet=puppet)
    
    def test(self):
        # 运动到压板预瞄点
        self.ik_caculator.run(color="green",puppet="left")
        #self.ik_caculator.run(color="green",puppet="right")
        time.sleep(3.2)


    def task_cmd_callback(self,msg):
        rospy.loginfo("Got cmd ros msg")
        if msg.data == "right_grab_green":
            self.cmd_type = "right_grab_green"
        
        elif msg.data == "right_put_green":
            self.cmd_type = "right_put_green"
        
        elif msg.data == "right_grab_red":
            self.cmd_type = "right_grab_red"
        
        elif msg.data == "right_put_red":
            self.cmd_type = "right_put_red"

        elif msg.data == "left_grab_green":
            self.cmd_type = "left_grab_green"
        
        elif msg.data == "left_put_green":
            self.cmd_type = "left_put_green"
        
        elif msg.data == "left_grab_red":
            self.cmd_type = "left_grab_red"
        
        elif msg.data == "left_put_red":
            self.cmd_type = "left_put_red"
        
        elif msg.data == "init_left":
            self.cmd_type = "init_left"
        
        elif msg.data == "init_right":
            self.cmd_type = "init_right"

        elif msg.data == "right_open_gripper":
            self.cmd_type = "right_open_gripper"
        elif msg.data == "right_close_gripper":
            self.cmd_type = "right_close_gripper"
        elif msg.data == "left_open_gripper":
            self.cmd_type = "left_open_gripper"
        elif msg.data == "left_close_gripper":
            self.cmd_type = "left_close_gripper"

        elif msg.data == "test":
            self.cmd_type = "test" 
        elif msg.data == "stop":
            self.cmd_type = "stop"

        
def signal_handler(signal,frame):
    task_manager_node.update_thread_stop_event.set()
    print('You pressed Ctrl + C!')
    rospy.signal_shutdown('Shutting down...')

if __name__=='__main__' :
    rospy.init_node("arm_task_manager")

    parser = argparse.ArgumentParser(description="示例：解析命令行参数")

    # 可选参数
    parser.add_argument('--init', action='store_true', help='init arm before excute action')
    parser.add_argument('--skip_detect', action='store_true', help='skip rgb detect')

    args = parser.parse_args()
    task_manager_node = task_manager(args)
    task_manager_node.task_thread.start()
    task_manager_node.task_states_thread.start()
    rospy.spin()


    
