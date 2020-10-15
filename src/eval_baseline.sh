#!/bin/bash
source /home/quantao/Workspaces/catkin_ws/devel/setup.bash
echo $ROS_PACKAGE_PATH
for seed in 54123; do # 123 231 312; do
  for u_step in 10 20; do
    for noise in 0.4 0.6 0.8; do
      for runid in 0 1 2 3 4; do
        /home/quantao/anaconda3/envs/py37/bin/python main.py --env 2DProblem --seed ${seed} --num_episodes 50 --run_id ${runid} --noise_scale ${noise} --logdir \
/home/quantao/naf_logs/action_project/ --updates_per_step ${u_step} --exploration_end 45 --project_actions=True
#       /home/aass/anaconda3/envs/naf_ros/bin/python main.py --env 2DProblem --seed ${seed} --num_episodes 50  --run_id ${runid} --noise_scale ${noise} --logdir \
#/home/aass/naf_logs/action_n_objective/ --updates_per_step ${u_step} --exploration_end 45 --project_actions=True --optimize_actions=True
#        /home/aass/anaconda3/envs/naf_ros/bin/python main.py --env 2DProblem --seed ${seed} --num_episodes 50  --run_id ${runid} --noise_scale ${noise} --logdir \
#/home/aass/naf_logs/baseline/ --updates_per_step ${u_step} --exploration_end 45
      done
    done
  done
done