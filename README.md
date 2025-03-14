# arm_ik 
## 代码架构
### arm_hw.py
- 核心类：arm_hw
- 使用方法：直接实例化类对象
- 作用：负责和机械臂硬件接口打交道，直接和机械臂驱动进行通信并控制机械臂，同时将一些简单的控制逻辑进行打包是机械臂驱动在上层的抽象表达类
- 数据流：该节点是单向节点，只有一个数据输入源和一个输出对象
  - 数据输入：
    - /target_pose/right：右臂目标位置（关节空间）
    - /target_pose/left：左臂目标位置（关节空间）
  - 数据处理：
    - 在回调函数target_pose_callback和target_pose_left_callback中，将目标关节空间和当前关节空间做差值，并生成连续的关节空间控制指令，让机械臂平稳运动到目标关节空间
  - 数据输出：
    - /master/joint_right：右臂关节空间控制话题
    - /master/joint_left：左臂关节空间控制话题
- arm_hw类中的其他method：
  - 控制类
    - gripper_control：控制末端夹爪的开合
    - motor_add_control：设置对应关节旋转对应角度
    - fold_arm：控制机械臂到折叠状态
  - 反馈类
    - arm_state_callback：订阅并存储右臂当前关节角度
    - arm_left_state_callback：订阅并存储左臂当前关节角度

### ik.py

- 核心类：ik_caculator
- 使用方法：直接实例化类对象
- 作用：从tf坐标系统读取笛卡尔空间下的机械臂末端姿态，运用逆解解出关节空间，并转发给arm_hw类进行后续处理
- 数据流：该节点中类的方法被调用，从tf坐标读取机械臂目标姿态，最终发送目标姿态的关节空间
  - 数据输出：
    - /target_pose/right：右臂目标位置（关节空间）
    - /target_pose/left：左臂目标位置（关节空间）

### task_manager.py

-  核心类：task_manager
- 使用方法：直接实例化类对象
- 作用：作为核心类，实例化其他功能模块的子类，调用其他模块的方法，完成抓取任务流程的所有逻辑