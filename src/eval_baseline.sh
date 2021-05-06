#!/bin/bash
source /home/quantao/Workspaces/catkin_ws/devel/setup.bash
echo $ROS_PACKAGE_PATH
for seed in 54123; do # 123 231 312; do
  for u_step in 10; do
    for noise in 1.2; do
      for runid in 0 1 2; do
        /home/quantao/anaconda3/envs/py37/bin/python main.py --env 2DProblem --algo NAF --seed ${seed} --num_episodes 200 --run_id ${runid} --noise_scale ${noise} --logdir \
/home/quantao/hiqp_logs/2DProblem/NAF/ --updates_per_step ${u_step} --exploration_end 50
        /home/quantao/anaconda3/envs/py37/bin/python main.py --env 2DProblem --algo DDPG --seed ${seed} --num_episodes 200  --run_id ${runid} --noise_scale ${noise} --logdir \
/home/quantao/hiqp_logs/2DProblem/DDPG/ --updates_per_step ${u_step} --exploration_end 50
      done
    done
  done
done

