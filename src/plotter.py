import argparse

from numpy import genfromtxt
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc

import matplotlib.colors as colors
import matplotlib.cm as cmx

import csv

def plot_mean_and_CI(axis, mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(axis, ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(axis, mean, color_mean)

def generate_plot(axis_dict, mean_dict, std_dict, colors, title):
    fig = plt.figure(1, figsize=(7, 2.5))
    legend_series = []
    for series in mean_dict:
        col = colors.pop()
        plot_mean_and_CI(axis_dict[series], mean_dict[series], mean_dict[series] - std_dict[series],
                         mean_dict[series] + std_dict[series], col, col)
        legend_series.append(series)

    plt.title(title)
    plt.legend(legend_series)
    plt.tight_layout()
    plt.grid()
    plt.show()

def plot_means(mean_dict, title="Eval episode return"):
    fig = plt.figure(1, figsize=(7, 9))
    legend_series = []
    for series in mean_dict:
        plt.plot(mean_dict[series])
        legend_series.append(series)
    plt.title(title)
    plt.legend(legend_series)
    plt.tight_layout()
    plt.grid()
    plt.show()

def parse_csvs(logdir, seeds, kds, action_scales, update_steps):

    mean_dict = {}
    for seed in seeds:
        for kd in kds:
            for action_scale in action_scales:
                for update_step in update_steps:
                    visited_states_training=numpy.array([])
                    rewards_training_step = numpy.array([])
                    rewards_training = []
                    rewards_testing = []
                    eval_traces=[]
                    with open(logdir+'/kd{}_sd{}_as{}_us_{}_train.csv'.format(kd,seed,action_scale,update_step), newline='') as train_csv:
                        csv_reader = csv.reader(train_csv,delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
                        for row in csv_reader:
                            reward = row.pop(0)
                            rewards_training.append(reward)
                            visit = numpy.reshape(row,[len(row)//5,5])
                            visited_states_training = numpy.append(visited_states_training,visit[:,0:2])
                            rewards_training_step = numpy.append(rewards_training_step,visit[:,4])

                    with open(logdir+'/kd{}_sd{}_as{}_us_{}_test.csv'.format(kd,seed,action_scale,update_step), newline='') as test_csv:
                        csv_reader = csv.reader(test_csv,delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
                        for row in csv_reader:
                            reward = row.pop(0)
                            rewards_testing.append(reward)
                            visit = numpy.reshape(row,[len(row)//5,5])
                            eval_traces.append(visit[:,0:2])

                    visited_states_training = numpy.reshape(visited_states_training,[len(visited_states_training)//2,2])
                    #test_series = genfromtxt(logdir+'/kd{}_sd{}_as{}_us_{}_test.csv'.format(kd,seed,action_scale,update_step),usecols=(0))
                    my_name="kd={},as={},us={},s={}".format(kd,action_scale,update_step,seed)
                    mean_dict[my_name] = rewards_testing

                    #plot visited states
                    fig = plt.figure()
                    #plt.hist2d(visited_states[:,0], visited_states[:,1],bins=20,range=[[-1,1],[-1,1]])
                    plt.scatter(visited_states_training[:,0], visited_states_training[:,1], c=rewards_training_step)
                    plt.title("Visited States (training) "+my_name)
                    plt.tight_layout()
                    plt.xlim((-1,1))
                    plt.ylim((-1,1))
#                    plt.show()
                    figname= logdir+"/"+my_name+'_train_visits.png'
                    plt.savefig(figname)
                    plt.close()

                    #plot evaluation trajectories
                    fig2 = plt.figure()
                    cmap = plt.cm.viridis
                    cNorm = colors.Normalize(vmin=-2000, vmax=500)
                    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
                    for i in range(len(eval_traces)):
                        plt.plot(eval_traces[i][:,0],eval_traces[i][:,1],color=scalarMap.to_rgba(rewards_testing[i]))
                    plt.title("Evaluation traces "+my_name)
                    plt.tight_layout()
                    plt.xlim((-1,1))
                    plt.ylim((-1, 1))
                    #plt.show()
                    figname = logdir + "/" + my_name + '_eval_visits.png'
                    plt.savefig(figname)
                    plt.close()

    return mean_dict


parser = argparse.ArgumentParser(description='Plot learning curves')
parser.add_argument('--logdir', default="",
                    help='directory where to dump log files')
args = parser.parse_args()

seeds = [2341]
kds = [0.0] #,1.0,10.0,100.0]
ascales = [100.0]
uscales = [10,50]

means = parse_csvs(args.logdir,seeds,kds,ascales,uscales)
plot_means(means)

#colors = ['darkgreen', 'maroon', 'darkorange','indigo',  'olive', 'navy', 'magenta', 'slategrey']
#colors = ['slategrey', 'lightsteelblue', 'cornflowerblue', 'royalblue', 'rosybrown', 'lightcoral', 'indianred', 'brown']
#colors = ['darkorange', 'navy']

