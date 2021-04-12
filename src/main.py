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
#get rid of that horrible plotter
from tensorboardX import SummaryWriter

from pathlib import Path
import csv

#import files...
from naf import NAF
from ddpg import DDPG
from reinforce import REINFORCE
from ounoise import OUNoise
from replay_memory import Transition, ReplayMemory
from prioritized_replay_memory import PrioritizedReplayMemory
from environment import ManipulateEnv

import quad

def main():
    parser = argparse.ArgumentParser(description='NAF nullspace learner')
    parser.add_argument('--algo', default='NAF',
                        help='algorithm to use: NAF | DDPG | REINFORCE')
    parser.add_argument('--env_name', default="2DProblem",
                        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--ou_noise', type=bool, default=True,
                        help='Should we use an OU process to sample noise?')
    parser.add_argument('--constr_gauss_sample', type=bool, default=False,
                        help='Should we use constrained Gaussian sampling?')
    parser.add_argument('--noise_scale', type=float, default=0.5, metavar='G',
                        help='initial noise scale (default: 0.5)')
    parser.add_argument('--final_noise_scale', type=float, default=0.2, metavar='G',
                        help='final noise scale (default: 0.2)')
    parser.add_argument('--project_actions', type=bool, default=False,
                        help='project to feasible actions only during training')
    parser.add_argument('--optimize_actions', type=bool, default=False,
                        help='add loss to objective')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='batch size (default: 512)')
    parser.add_argument('--num_steps', type=int, default=300, metavar='N',
                        help='max episode length (default: 300)')
    parser.add_argument('--num_episodes', type=int, default=5000, metavar='N',
                        help='number of episodes (default: 5000)')
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
    parser.add_argument('--logdir', default="/home/qoyg/hiqp_logs",
                        help='directory where to dump log files')
    parser.add_argument('--action_scale', type=float, default=1.0, metavar='N',
                        help='scale applied to the normalized actions (default: 1.0)')
    parser.add_argument('--kd', type=float, default=0.0, metavar='N',
                        help='derivative gain for ee_rl (default: 0.0)')
    parser.add_argument('--prioritized_replay_memory', type=bool, default=False,
                        help='prioritized experience replay')

    args = parser.parse_args()

    print("++++++seed:{} algo:{} updates_per_step:{} project_actions:{} optimize_actions:{}++++++".format(args.seed, args.algo, args.updates_per_step, args.project_actions, args.optimize_actions))

    if args.env_name == '2DProblem':
        env = ManipulateEnv(bEffort=False)
        print(args.action_scale)

        env.set_scale(args.action_scale)
        env.set_kd(args.kd)
    else:
        env = gym.make(args.env_name)

    basename = 'sd{}_us_{}_ns_{}_run_{}'.format(args.seed,args.updates_per_step,args.noise_scale,args.run_id)
    writer = SummaryWriter(args.logdir+'/runs/'+basename)

    path = Path(args.logdir)
    if not path.exists():
        raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

    csv_train = open(args.logdir+'/'+basename+'_train.csv', 'w',
              newline='')
    train_writer = csv.writer(csv_train, delimiter=' ')

    csv_test = open(args.logdir+'/'+basename+'_test.csv', 'w',
                  newline='')
    test_writer = csv.writer(csv_test, delimiter=' ')


    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- initialize agent --
    if args.algo == "NAF":
        agent = NAF(args.gamma, args.tau, args.hidden_size, env.observation_space.shape[0], env.action_space)
    elif args.algo == "DDPG":
        agent = DDPG(args.gamma, args.tau, args.hidden_size, env.observation_space.shape[0], env.action_space)
    else:
        agent = REINFORCE(args.hidden_size, env.observation_space.shape[0], env.action_space)    

    # -- declare memory buffer and random process N
    if args.prioritized_replay_memory:
        memory = PrioritizedReplayMemory(args.replay_size, 0.8)
    else:
        memory = ReplayMemory(args.replay_size)
    ounoise = OUNoise(env.action_space.shape[0])

    # -- load existing model --
    if args.load_agent:
        agent.load_model(args.env_name, args.batch_size, args.num_episodes, basename+'.pth', model_path=args.logdir)
        print('loaded agent '+basename+' from '+args.logdir)
    # -- load experience buffer --
    if args.load_exp:
        with open(args.logdir+'/'+basename+'.pk', 'rb') as input:
            memory.memory = pickle.load(input)
            memory.position = len(memory)
    
    reaches = []
    violations = []
    rewards = []
    total_numsteps = 0
    updates = 0
    
    env.stop()
    t_start = time.time()

    for i_episode in range(args.num_episodes+1):
        # -- reset environment for every episode --
        print('++++++++++++++++++++++++++i_episode+++++++++++++++++++++++++++++:', i_episode)
        t_st = time.time()
        state = torch.Tensor([env.start()])
        print("reset took {}".format(time.time() - t_st))

        scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end - i_episode) / args.exploration_end + args.final_noise_scale
        #scale = [min(scale,0.4), min(scale,0.2)]

        # -- initialize noise (random process N) --
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(
            0, args.exploration_end - i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()
        print("noise scale is {}".format(ounoise.scale))

        episode_reached = 0
        episode_violation = 0
        episode_reward = 0
        visits = []
        Ax_prev = np.identity(env.action_space.shape[0])
        bx_prev = env.action_space.high

        t_project = 0
        t_act = 0
        step = 0
        while True:
            # -- action selection, observation and store transition --
            if args.ou_noise:
                action = agent.select_action(state, ounoise) if args.train_model else agent.select_action(state)
                if args.project_actions:
                    #project, add noise, project again
                    action = agent.select_proj_action(state,Ax_prev,bx_prev,ounoise)
                #else clause: just plain OU noise on top (or no-noise)
            else:
                if args.constr_gauss_sample:
                    #Gaussian noise sample
                    action = agent.select_proj_action(state,Ax_prev,bx_prev,simple_noise=scale)
                else:
                    #noise-free run
                    action = agent.select_proj_action(state,Ax_prev,bx_prev)
            
            if args.algo == "REINFORCE":
                action, log_prob, entropy = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
            # env step
            t_st0 = time.time()
            next_state, reward, done, Ax, bx = env.step(action)
            t_act += time.time() - t_st0
            #print("act took {}".format(time.time() - t_st0))

            visits = np.concatenate((visits,state.numpy(),args.action_scale*action,[reward]),axis=None)
            #env.render()
            total_numsteps += 1
            episode_reward += reward

            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            reward = torch.Tensor([reward])
            next_state = torch.Tensor([next_state])
            Ax_trace = torch.Tensor(Ax_prev)
            bx_trace = torch.Tensor([bx_prev])
            
            
            #step += 1
            #agent.plot_state_action_pair([state], [action], [action_naf], Ax, bx, i_episode, step)

            Ax_prev = Ax
            bx_prev = bx[0]

            #print('reward:', reward)
            memory.push(state, action, mask, next_state, reward, Ax_trace, bx_trace)

            state = next_state

            if done or total_numsteps % args.num_steps == 0 or env.bConstraint:
                if env.bConstraint:
                    episode_violation += 1
                elif done:
                    episode_reached += 1
                break

        print("====>Train Episode: {}, total numsteps: {}, reward: {}, time: {} act: {} project: {}".format(i_episode, total_numsteps,
                                                                         episode_reward,time.time()-t_st,t_act,t_project))
        print("Percentage of actions in constraint violation was {}".format(np.sum([env.episode_trace[i][2]>0 for i in range(len(env.episode_trace))])))

        train_writer.writerow(np.concatenate(([episode_reward],[episode_violation],[episode_reached],visits),axis=None))
        rewards.append(episode_reward)
        violations.append(episode_violation)
        reaches.append(episode_reached)
        #trying out this?
        env.stop()
        t_st = time.time()

        #Training models
        if len(memory) >= args.batch_size and args.train_model:
            #env.reset()
            print("Training model")
            #env.step(torch.Tensor([[0,0]]))

            t_st = time.time()
            for _ in range(args.updates_per_step*args.num_steps):
                if args.prioritized_replay_memory:
                    transitions = memory.sample(args.batch_size, 0.5)
                else:
                    transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))
                if args.algo == "NAF":
                    value_loss, reg_loss = agent.update_parameters(batch,optimize_feasible_mu=args.optimize_actions)
                else:
                    value_loss, reg_loss = agent.update_parameters(batch)

                writer.add_scalar('loss/full', value_loss, updates)
                if args.optimize_actions:
                    writer.add_scalar('loss/value', value_loss-reg_loss, updates)
                    writer.add_scalar('loss/regularizer', reg_loss, updates)

                updates += 1
            print("train took {}".format(time.time() - t_st))

        #agent.save_value_funct(
        #    args.logdir + '/kd{}_sd{}_as{}_us_{}'.format(args.kd, args.seed, args.action_scale, args.updates_per_step),
        #    i_episode,
        #    ([-1.0, -1.0], [1.0, 1.0], [300, 300]))
        
        #runing evaluation episode
        greedy_numsteps = 0
        if i_episode > 90:
            #state = env.reset()
            if time.time() - t_st < 4:
                print("waiting for reset...")
                time.sleep(4.0)
                print("DONE")
            state = torch.Tensor([env.start()])
            Ax_prev = np.identity(env.action_space.shape[0])
            bx_prev = env.action_space.high

            episode_violation = 0
            episode_reward = 0
            visits = []
            while True:
                action = agent.select_action(state)
                if args.project_actions:
                    action = agent.select_proj_action(state, Ax_prev, bx_prev)

                #action = torch.Tensor(
                #    [quad.project_action(env.action_scale * action.numpy()[0], Ax_prev, bx_prev) / env.action_scale])

                next_state, reward, done, Ax, bx = env.step(action)
                visits = np.concatenate((visits, state.numpy(), action, [reward]), axis=None)
                episode_reward += reward
                greedy_numsteps += 1
                Ax_prev = Ax
                bx_prev = bx[0]

                state = torch.Tensor([next_state])

                if done or greedy_numsteps % args.num_steps == 0 or env.bConstraint:
                    if env.bConstraint:
                        episode_violation += 1
                    break
                
            writer.add_scalar('reward/test', episode_reward, i_episode)
            test_writer.writerow(np.concatenate(([episode_reward], [episode_violation], visits), axis=None))

            rewards.append(episode_reward)
            violations.append(episode_violation)

            print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))
            print('Time per episode: {} s'.format((time.time() - t_start) / (i_episode+1)))
            print("Percentage of actions in constraint violation was {}".format(np.sum([env.episode_trace[i][2]>0 for i in range(len(env.episode_trace))])))

            env.stop()
            time.sleep(4.0) #wait for reset


    #-- saves model --
    if args.save_agent:
        agent.save_model(args.env_name, args.batch_size, args.num_episodes, basename+'.pth')
        with open(args.logdir+'/'+basename+'.pk', 'wb') as output:
            pickle.dump(memory.memory, output, pickle.HIGHEST_PROTOCOL)

    print('Training ended after {} minutes'.format((time.time() - t_start)/60))
    print('Time per episode: {} s'.format((time.time() - t_start) / args.num_episodes))
    print('Mean reward: {}'.format(np.mean(rewards)))
    print('Max reward: {}'.format(np.max(rewards)))
    print('Min reward: {}'.format(np.min(rewards)))
    print('Total constraint violations: {} out of {} episodes'.format(np.sum(violations), args.num_episodes))
    

if __name__ == '__main__':
    main()

