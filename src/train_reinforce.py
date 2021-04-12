#!/usr/bin/env python
import argparse
import numpy as np
import torch
import time
import pickle
import matplotlib.colors as colors
import matplotlib.cm as cmx
import gym
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from pathlib import Path

from reinforce import REINFORCE
from environment import ManipulateEnv


def main():
    parser = argparse.ArgumentParser(description='Nullspace RL learner')
    parser.add_argument('--algo', default='REINFORCE',
                        help='algorithm to use: NAF | DDPG | REINFORCE')
    parser.add_argument('--env_name', default="2DProblem",
                        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--project_actions', type=bool, default=False,
                        help='project to feasible actions only during training')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='batch size (default: 512)')
    parser.add_argument('--num_steps', type=int, default=300, metavar='N',
                        help='max episode length (default: 300)')
    parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                        help='number of episodes (default: 2000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='hidden size (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=10, metavar='N',
                    help='model updates per simulator step (default: 50)')
    parser.add_argument('--run_id', type=int, default=0, metavar='N',
                        help='increment this externally to re-run same parameters multiple times')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--save_agent', type=bool, default=False,
                        help='save model to file')
    parser.add_argument('--train_model', type=bool, default=True,
                        help='Training or run')
    parser.add_argument('--load_agent', type=bool, default=False,
                        help='load model from file')
    parser.add_argument('--load_exp', type=bool, default=False,
                        help='load saved experience')
    parser.add_argument('--logdir', default="/home/quantao/hiqp_logs",
                        help='directory where to dump log files')
    parser.add_argument('--action_scale', type=float, default=1.0, metavar='N',
                        help='scale applied to the normalized actions (default: 1.0)')
    parser.add_argument('--kd', type=float, default=0.0, metavar='N',
                        help='derivative gain for ee_rl (default: 0.0)')

    args = parser.parse_args()

    if args.env_name == '2DProblem':
        env = ManipulateEnv(bEffort=False)
        print(args.action_scale)

        env.set_scale(args.action_scale)
        env.set_kd(args.kd)
    else:
        env = gym.make(args.env_name)

    basename = 'REINFORCE_sd{}'.format(args.seed)
    writer = SummaryWriter(args.logdir+'/runs/'+basename)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- initialize agent --
    agent = REINFORCE(args.hidden_size, env.observation_space.shape[0], env.action_space)
    
    env.stop()

    for i_episode in range(args.num_episodes):
        # -- reset environment for every episode --
        print('++++++++++++++++++++++++++i_episode+++++++++++++++++++++++++++++:', i_episode)
        state = torch.Tensor([env.start()])
        
        episode_numsteps = 0
        entropies = []
        log_probs = []
        rewards = []
        while True:
            # -- action selection, observation and store transition --   
            action, log_prob, entropy = agent.select_action(state)               
            action = action.cpu()
            
            next_state, reward, done, _, _ = env.step(action)

            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = torch.Tensor([next_state])
            
            episode_numsteps += 1
            if done or episode_numsteps==args.num_steps:
                print("break:", episode_numsteps)
                break

        env.stop()
        time.sleep(4)

        #Training models
        agent.update_parameters(rewards, log_probs, entropies, args.gamma)
        
        writer.add_scalar('reward/train', np.sum(rewards), i_episode)       
        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

if __name__ == '__main__':
    main()

