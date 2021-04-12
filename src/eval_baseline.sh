#!/bin/bash
source /home/qoyg/Workspaces/catkin_ws/devel/setup.bash
echo $ROS_PACKAGE_PATH
for seed in 54123; do # 123 231 312; do
  for u_step in 10; do
    for noise in 1.2; do
      for runid in 0 1 2 3 4; do
#        /home/qoyg/anaconda3/envs/py37/bin/python main.py --env 2DProblem --seed ${seed} --num_episodes 100 --run_id ${runid} --noise_scale ${noise} --logdir \
#/home/qoyg/naf_logs/action_project/ --updates_per_step ${u_step} --exploration_end 90 --project_actions=True
#       /home/qoyg/anaconda3/envs/py37/bin/python main.py --env 2DProblem --seed ${seed} --num_episodes 50  --run_id ${runid} --noise_scale ${noise} --logdir \
#/home/qoyg/naf_logs/action_n_objective/ --updates_per_step ${u_step} --exploration_end 45 --project_actions=True --optimize_actions=True
        /home/qoyg/anaconda3/envs/py37/bin/python main.py --env 2DProblem --algo NAF --seed ${seed} --num_episodes 100  --run_id ${runid} --noise_scale ${noise} --logdir \
/home/qoyg/naf_logs/baseline/ --updates_per_step ${u_step} --exploration_end 90
      done
    done
  done
done

