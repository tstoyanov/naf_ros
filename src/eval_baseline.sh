#!/bin/bash
source /home/aass/pandarl_ws/devel/setup.bash
echo $ROS_PACKAGE_PATH
for seed in 54123; do # 123 231 312; do
for kd in 0; do
  for a_scale in 1; do # 100 ; do
    for u_step in 10 20; do #1 5 10 ; do
       /home/aass/anaconda3/envs/naf_ros/bin/python main.py --env 2DProblem --seed ${seed} --kd ${kd} --num_episodes 50 --action_scale ${a_scale} --logdir \
/home/aass/naf_logs/action_project/ --updates_per_step ${u_step} --exploration_end 45 --project_actions=True
#       /home/aass/anaconda3/envs/naf_ros/bin/python main.py --env 2DProblem --seed ${seed} --kd ${kd} --num_episodes 50 --action_scale ${a_scale} --logdir \
#/home/aass/naf_logs/action_n_objective/ --updates_per_step ${u_step} --exploration_end 45 --project_actions=True --optimize_actions=True
#       /home/aass/anaconda3/envs/naf_ros/bin/python main.py --env 2DProblem --seed ${seed} --kd ${kd} --num_episodes 50 --action_scale ${a_scale} --logdir \
#/home/aass/naf_logs/baseline/ --updates_per_step ${u_step} --exploration_end 45
    done
  done
done
done