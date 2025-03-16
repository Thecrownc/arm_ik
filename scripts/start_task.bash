#!/bin/bash
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate yolo
source ~/arm_ik/devel/setup.bash
cd ~/arm_ik/src/arm_ik/scripts
roslaunch arm_ik start_task.launch &
python3 task_manager.py "$@"


