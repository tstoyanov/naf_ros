import argparse

from numpy import genfromtxt
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc

import matplotlib.colors as colors
import matplotlib.cm as cmx

import csv

def plot_violations(violation_dict):
    fig = plt.figure(1, figsize=(12, 9))
    #colors = ['darkorange', 'navy']
    #subdirs = ["action_project", "baseline"]

    violation_dict["action_project"] = violation_dict["action_project"][:50]
    violation_dict["baseline"] = violation_dict["baseline"][:50]
    axis_x = list(range(len(violation_dict["action_project"]+1)))
    #axis_x = list(range(50))
    violations_actionproj = violation_dict["action_project"]
    violations_baseline = violation_dict["baseline"]
    # creating the bar plot 
    plt.bar(axis_x, violations_actionproj, color = 'darkorange', width = 1.0, label='action_proj')
    plt.bar(axis_x, violations_baseline, color = 'navy', width = 0.5, label='baseline')
        
    plt.xticks(range(0, len(violation_dict["action_project"]+1), 10))
        
    subdirs = ["null space NAF with action projection", "null space NAF"]
    plt.xlabel("iter") 
    plt.ylabel("violations")
    plt.yticks(numpy.arange(0, 5, 1))
    plt.title("Constraint violations of null space NAF methods") 
    plt.grid()
    plt.legend(subdirs)
    
    plt.show() 
    
    # save this figure
    figname = "Constraint violations.png"
    fig.savefig(figname)
    
def plot_reaches(reach_dict):
    fig = plt.figure(1, figsize=(12, 9))
    #colors = ['darkorange', 'navy']
    #subdirs = ["action_project", "baseline"]

    axis_x = list(range(len(reach_dict["action_project"]+1)))
    reaches_actionproj = reach_dict["action_project"]
    reaches_baseline = reach_dict["baseline"]
    # creating the bar plot 
    plt.bar(axis_x, reaches_actionproj, color = 'darkorange', width = 1.0, label='action_proj')
    plt.bar(axis_x, reaches_baseline, color = 'navy', width = 0.5, label='baseline')
        
    plt.xticks(range(0, len(reach_dict["action_project"]+1), 10))
        
    plt.xlabel("iter") 
    plt.ylabel("reaches") 
    plt.title("Reaches of null-space RL and baseline NAF") 
    plt.grid()
    plt.legend()
    
    plt.show() 
    
    # save this figure
    figname = "reaches.png"
    fig.savefig(figname)
        

def plot_mean_and_CI(axis, mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(axis, ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(axis, mean, color_mean)

def generate_plot(mean_dict, std_dict, seeds, update_steps, noises):
    subdirs = ["action_project", "baseline"]
    
    iter_num = len(mean_dict["action_project,sd=54123,us=10,ns=1.2"])
    x = numpy.linspace(0, iter_num-1, iter_num)
    
    for seed in seeds:
        for update_step in update_steps:
            for noise in noises:
                #fig = plt.figure(1, figsize=(7, 2.5))
                colors = ['darkorange', 'navy']
                title="sd={},us={},ns={}".format(seed,update_step,noise)
                title="Cumulative reward"
                # plot one figure
                fig = plt.figure(1, figsize=(12, 9))
                plt.title(title)
                
                for subdir in subdirs:
                    col = colors.pop(0)
                    my_name="{},sd={},us={},ns={}".format(subdir,seed,update_step,noise)
                    plot_mean_and_CI(x, mean_dict[my_name], mean_dict[my_name] - std_dict[my_name],
                         mean_dict[my_name] + std_dict[my_name], col, col)
                
                subdirs = ["null space NAF with action projection", "null space NAF"]
                plt.ylim(-4000, 1000)
                plt.legend(subdirs)
                #plt.tight_layout()
                plt.xlabel("iter")
                plt.ylabel("eval rewards")
                plt.grid()
                plt.show()
                               
                # save this figure
                figname = title+".png"
                fig.savefig(figname)

def plot_means(figname, mean_dict, title="Eval episode return"):
    fig = plt.figure(1, figsize=(7, 9))
    legend_series = []
    for series in mean_dict:
        plt.plot(mean_dict[series])
        legend_series.append(series)
    plt.title(title)
    plt.legend(legend_series)
    plt.tight_layout()
    plt.grid()
    plt.savefig(figname)
   # plt.show()
    plt.close()

def parse_mean_and_std(logdir, seeds, update_steps, noises, runs):
    rootdir = logdir
    subdirs = ["action_project", "baseline"]
    
    mean_dict = {}
    std_dict = {}
    violation_dict = {}
    reach_dict = {}

    for seed in seeds:
        for update_step in update_steps:
            for noise in noises:
                for subdir in subdirs:
                    logdir = rootdir+subdir
                    rewards_list = []
                    violations_list = []
                    reaches_list = []
                    for run in runs:
                        #if subdir == "baseline":
                        #    run = 1
                        rewards_testing = []
                        violations_testing = []
                        reaches_testing = []
                        with open(logdir+'/sd{}_us_{}_ns_{}_run_{}_train.csv'.format(seed,update_step,noise,run), newline='') as test_csv:
                            csv_reader = csv.reader(test_csv,delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
                            for row in csv_reader:
                                reward = row.pop(0)
                                rewards_testing.append(reward)
                                violation = row.pop(0)
                                violations_testing.append(violation)
                                reach = row.pop(0)
                                reaches_testing.append(reach)
                                
                        rewards_list.append(rewards_testing)
                        violations_list.append(violations_testing)
                        reaches_list.append(reaches_testing)
                    
                    rewards_allruns = numpy.array(rewards_list)
                    violations_allruns = numpy.array(violations_list)
                    reaches_allruns = numpy.array(reaches_list)

                    # rewards mean and std                     
                    means_allruns = rewards_allruns.mean(axis=0)
                    std_allruns = rewards_allruns.std(axis=0)
                    my_name="{},sd={},us={},ns={}".format(subdir,seed,update_step,noise)                    
                    mean_dict[my_name] = means_allruns
                    std_dict[my_name] = std_allruns
                    
                    # violation count
                    violation_dict[subdir] = violations_allruns.sum(axis=0)
                    # reach count
                    reach_dict[subdir] = reaches_allruns.sum(axis=0)

    return mean_dict, std_dict, violation_dict, reach_dict
    
    
def parse_csvs(logdir, seeds, update_steps, noises, runs):
    rootdir = logdir
    subdirs = ["action_project", "baseline"]
    
    mean_dict = {}
    
    for seed in seeds:
        for update_step in update_steps:
            for noise in noises:
                for subdir in subdirs:
                    logdir = rootdir+subdir
                    for run in runs:
                        visited_states_training=numpy.array([])
                        actions_training=numpy.array([])
                        rewards_training_step = numpy.array([])
                        rewards_training = []
                        rewards_testing = []
                        eval_traces=[]
                        with open(logdir+'/sd{}_us_{}_ns_{}_run_{}_train.csv'.format(seed,update_step,noise,run), newline='') as train_csv:
                            csv_reader = csv.reader(train_csv,delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
                            for row in csv_reader:
                                reward = row.pop(0)
                                rewards_training.append(reward)
                                visit = numpy.reshape(row[:len(row)//5*5],[len(row)//5,5])
                                visited_states_training = numpy.append(visited_states_training,visit[:,0:2])
                                actions_training = numpy.append(actions_training,visit[:,2:4])
                                rewards_training_step = numpy.append(rewards_training_step,visit[:,4])
    
                        with open(logdir+'/sd{}_us_{}_ns_{}_run_{}_test.csv'.format(seed,update_step,noise,run), newline='') as test_csv:
                            csv_reader = csv.reader(test_csv,delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
                            for row in csv_reader:
                                reward = row.pop(0)
                                rewards_testing.append(reward)
                                visit = numpy.reshape(row[:len(row)//5*5],[len(row)//5,5])
                                eval_traces.append(visit[:,0:2])
    
                        visited_states_training = numpy.reshape(visited_states_training,[len(visited_states_training)//2,2])
                        actions_training = numpy.reshape(actions_training,[len(actions_training)//2,2])
                        #test_series = genfromtxt(logdir+'/kd{}_sd{}_as{}_us_{}_test.csv'.format(kd,seed,action_scale,update_step),usecols=(0))
                        my_name="{},sd={},us={},ns={},run={}".format(subdir,seed,update_step,noise,run)
                        mean_dict[my_name] = rewards_testing
    
                        cmap = plt.cm.viridis
                        cNorm = colors.Normalize(vmin=numpy.min(rewards_training_step), vmax=numpy.max(rewards_training_step))
                        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    
                        #plot visited states
                        fig = plt.figure()
                        #plt.hist2d(visited_states[:,0], visited_states[:,1],bins=20,range=[[-1,1],[-1,1]])
                        plt.scatter(visited_states_training[:,0], visited_states_training[:,1], c=rewards_training_step)
                        plt.title("Visited States (training) "+my_name)
                        plt.tight_layout()
                        plt.xlim((-1,1))
                        plt.ylim((-1,1))
                        #plt.show()
                        figname= logdir+"/"+my_name+'_train_visits.png'
                        plt.savefig(figname)
                        plt.close()
    
                        fig3 = plt.figure()
                        for i in range(numpy.shape(actions_training)[0]):
                            colorVal = scalarMap.to_rgba(rewards_training_step[i])
                            plt.arrow(visited_states_training[i,0], visited_states_training[i,1], actions_training[i,0],
                                      actions_training[i,1], fc=colorVal, ec=colorVal, head_width=0.004,head_length=0.008)
    
                        plt.title("Visited States (training) " + my_name)
                        plt.tight_layout()
                        plt.xlim((-1, 1))
                        plt.ylim((-1, 1))
                        figname = logdir + "/" + my_name + '_train_actions.png'
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

    #myList[:] = [x / myInt for x in myList]
    return mean_dict


parser = argparse.ArgumentParser(description='Plot learning curves')
parser.add_argument('--logdir', default="/home/qoyg/naf_logs/evaluation_27_10_2020/",
                    help='directory where to dump log files')
args = parser.parse_args()

seeds = [54123]
uscales = [10]
noises = [1.2]
runs = [0,1,2,3,4]

#means = parse_csvs(args.logdir,seeds,uscales,noises,runs)
mean_dict, std_dict, violation_dict, reach_dict = parse_mean_and_std(args.logdir,seeds,uscales,noises,runs)


generate_plot(mean_dict, std_dict, seeds, uscales, noises)

plot_violations(violation_dict)

plot_reaches(reach_dict)




#figname = args.logdir + "/evaluation_curves.png"
#plot_means(figname,means)

#colors = ['darkgreen', 'maroon', 'darkorange','indigo',  'olive', 'navy', 'magenta', 'slategrey']
#colors = ['slategrey', 'lightsteelblue', 'cornflowerblue', 'royalblue', 'rosybrown', 'lightcoral', 'indianred', 'brown']
#colors = ['darkorange', 'navy']

