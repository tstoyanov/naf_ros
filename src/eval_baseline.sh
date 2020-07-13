#!/bin/bash
source /home/aass/pandarl_ws/devel/setup.bash
echo $ROS_PACKAGE_PATH
#for seed in 1234 2341 3412; do
for kd in 0 10; do
  for a_scale in 100; do # 100 ; do
    for u_step in 10 50; do #1 5 10 ; do
       /home/aass/anaconda3/envs/naf_ros/bin/python main.py --env 2DProblem --seed 2341 --kd ${kd} --num_episodes 300 --action_scale ${a_scale} --logdir \
/home/aass/naf_logs/ --updates_per_step ${u_step} --exploration-end 200
    done
  done
done
#done