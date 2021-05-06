import rospy
from rl_task_plugins.msg import DesiredErrorDynamicsMsg
from rl_task_plugins.msg import StateMsg
import subprocess
import math
import torch
import gym
from gym import spaces
import numpy as np
import time
import csv

from controller_manager_msgs.srv import *
from std_msgs.msg import *
from hiqp_msgs.srv import *
from hiqp_msgs.msg import *
from trajectory_msgs.msg import *

from halfspaces import qhull

class ManipulateEnv(gym.Env):
    """Manipulation Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self,bEffort=True):
        super(ManipulateEnv, self).__init__()

        self.goal = np.array([-0.2, -0.5])
        self.bEffort = bEffort
        self.bConstraint = False

        #These seem to be here for the enjoyment of the reader only, what are theyused for?
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        obs_low = np.array([-1.89, -1.89, -1.89, -2.5, -2.5, -2.5, -1, -1])
        self.observation_space = spaces.Box(low=obs_low, high=-obs_low, dtype=np.float32)

        self.action_scale = 10
        self.kd = 10

        rospy.init_node('DRL_node', anonymous=True)
        #queue_size = None forces synchronous publishing
        self.pub = rospy.Publisher('/ee_rl/act', DesiredErrorDynamicsMsg, queue_size=None)
        self.effort_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        self.velocity_pub = rospy.Publisher('/velocity_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        self.rate = rospy.Rate(10) #10Hz
        self.set_primitives()
        self.set_tasks()

        #queue size = 1 only keeps most recent message
        self.sub = rospy.Subscriber("/ee_rl/state", StateMsg, self._next_observation, queue_size=1)
        
        #monitor constraints      
        self.sub_monitor = rospy.Subscriber("/hiqp_joint_velocity_controller/task_measures", TaskMeasures, self._constraint_monitor, queue_size=1)

        self.rate.sleep()
        time.sleep(1) #wait for ros to start up
        #HOW ugly can you be?
        self.fresh=False

        csv_train = open("/home/quantao/hiqp_logs/constraints.csv", 'w', newline='')

        self.twriter = csv.writer(csv_train, delimiter=' ')
        self.episode_trace = [(np.identity(self.action_space.shape[0]),self.action_space.high,0)]

    def set_scale(self,action_scale):
        self.action_scale = action_scale

    def set_kd(self,kd):
        self.kd = kd

    def set_primitives(self):
        #print("setting primitves")
        #set all primitives into hiqp
        if(self.bEffort):
            hiqp_primitve_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_primitives', SetPrimitives)
        else:
            hiqp_primitve_srv = rospy.ServiceProxy('/hiqp_joint_velocity_controller/set_primitives', SetPrimitives)

        ee_prim = Primitive(name='ee_point',type='point',frame_id='three_dof_planar_eef',visible=True,color=[1,0,0,1],parameters=[0,0,0])
        goal_prim = Primitive(name='goal',type='sphere',frame_id='world',visible=True,color=[0,1,0,1],parameters=[self.goal[0],self.goal[1],0,0.02])       
        back_plane = Primitive(name='back_plane',type='plane',frame_id='world',visible=True,color=[0,1,0,0.5],parameters=[0,1,0,-0.6])
        front_plane = Primitive(name='front_plane',type='plane',frame_id='world',visible=True,color=[0,1,0,0.5],parameters=[0,1,0,0.6])
        left_plane = Primitive(name='left_plane',type='plane',frame_id='world',visible=True,color=[0,1,0,0.5],parameters=[1,0,0,-0.6])
        right_plane = Primitive(name='right_plane',type='plane',frame_id='world',visible=True,color=[0,1,0,0.5],parameters=[1,0,0,0.6])
        # four corners      
        corner1 = Primitive(name='corner1',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[0.6,0.6,0,0.02])
        corner2 = Primitive(name='corner2',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[0.6,-0.6,0,0.02])
        corner3 = Primitive(name='corner3',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[-0.6,-0.6,0,0.02])
        corner4 = Primitive(name='corner4',type='sphere',frame_id='world',visible=True,color=[0,0,1,1],parameters=[-0.6,0.6,0,0.02])
        # obstacle
        obs_cylinder1 = Primitive(name='obs_cylinder1',type='cylinder',frame_id='world',visible=True,color=[1.0,0.0,0.0,0.5],parameters=[0,0,1,0.3,0.2,0,0.1,0.1])
        obs_cylinder2 = Primitive(name='obs_cylinder2',type='cylinder',frame_id='world',visible=True,color=[1.0,0.0,0.0,0.5],parameters=[0,0,1,0.2,-0.1,0,0.1,0.1])
        obs_cylinder3 = Primitive(name='obs_cylinder3',type='cylinder',frame_id='world',visible=True,color=[1.0,0.0,0.0,0.5],parameters=[0,0,1,0.4,-0.4,0,0.1,0.1])

        hiqp_primitve_srv([ee_prim, back_plane, front_plane, left_plane, right_plane, goal_prim, corner1, corner2, corner3, corner4, obs_cylinder1, obs_cylinder2, obs_cylinder3])

    def set_tasks(self):
        #set the tasks to hiqp
        #print("setting tasks")
        if self.bEffort:
            hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_tasks', SetTasks)
        else:
            hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_velocity_controller/set_tasks', SetTasks)

        cage_front = Task(name='ee_cage_front',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'plane', 'ee_point < front_plane'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        cage_back = Task(name='ee_cage_back',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'plane', 'ee_point > back_plane'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        cage_left = Task(name='ee_cage_left',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'plane', 'ee_point > left_plane'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        cage_right = Task(name='ee_cage_right',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'plane', 'ee_point < right_plane'],
                          dyn_params=['TDynPD', '16.0', '9.0'])
        cylinder_avoidance1 = Task(name='cylinder_avoidance1',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'cylinder', 'ee_point > obs_cylinder1'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        cylinder_avoidance2 = Task(name='cylinder_avoidance2',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'cylinder', 'ee_point > obs_cylinder2'],
                          dyn_params=['TDynPD', '1.0', '2.0']) 
        cylinder_avoidance3 = Task(name='cylinder_avoidance3',priority=0,visible=True,active=True,monitored=True,
                          def_params=['TDefGeomProj','point', 'cylinder', 'ee_point > obs_cylinder3'],
                          dyn_params=['TDynPD', '1.0', '2.0']) 
        rl_task = Task(name='ee_rl',priority=1,visible=True,active=True,monitored=True,
                          def_params=['TDefRL2DSpace','1','0','0','0','1','0','ee_point'],
                          dyn_params=['TDynAsyncPolicy', '{}'.format(self.kd), 'ee_rl/act', 'ee_rl/state']) #, '/home/aass/hiqp_logs/'
        redundancy = Task(name='full_pose',priority=2,visible=True,active=True,monitored=True,
                          def_params=['TDefFullPose', '0.3', '-0.8', '-1.3'],
                          dyn_params=['TDynPD', '1.0', '2.0'])
        #hiqp_task_srv([cage_front,cage_back,cage_left,cage_right,cylinder_avoidance1,cylinder_avoidance2,cylinder_avoidance3,rl_task,redundancy])
        hiqp_task_srv([cage_front,cage_back,cage_left,cage_right,rl_task,redundancy])

    def _next_observation(self, data):
        self.e = np.array(data.e)
        self.de = np.array(data.de)
        self.J = np.transpose(np.reshape(np.array(data.J_lower),[data.n_joints,data.n_constraints_lower]))
        self.A = np.transpose(np.reshape(np.array(data.J_upper),[data.n_joints,data.n_constraints_upper]))
        self.b = -np.reshape(np.array(data.b_upper),[data.n_constraints_upper,1])
        self.rhs = -np.reshape(np.array(data.rhs_fixed_term),[data.n_constraints_lower,1])
        self.q = np.reshape(np.array(data.q),[data.n_joints,1])
        self.dq = np.reshape(np.array(data.dq),[data.n_joints,1])
        self.ddq_star = np.reshape(np.array(data.ddq_star),[data.n_joints,1])
        self.observation = np.concatenate([np.squeeze(self.q), np.squeeze(self.dq), self.e-self.goal])

        self.fresh = True
        
    def _constraint_monitor(self, data):
        violate_thre = 0.01
        penalty_scale = 1.0
                
        for task in data.task_measures:
            if task.task_name == "ee_cage_back" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_back violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
            
            if task.task_name == "ee_cage_front" and task.e[0] > 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_front violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
       
            if task.task_name == "ee_cage_left" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_left violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
            
            if task.task_name == "ee_cage_right" and task.e[0] > 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************ee_cage_right violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                
            if task.task_name == "jnt1_limits":
                if task.e[0] < 0 or task.e[1] > 0 or task.e[2] < 0 or task.e[3] > 0 or task.e[4] < 0 or task.e[5] > 0:
                    print("*************jnt1_limits violated!***", task.e[0], task.e[1],task.e[2],task.e[3],task.e[4],task.e[5])
                    self.bConstraint = True
                    
            if task.task_name == "jnt2_limits":
                if task.e[0] < 0 or task.e[1] > 0 or task.e[2] < 0 or task.e[3] > 0 or task.e[4] < 0 or task.e[5] > 0:
                    print("*************jnt2_limits violated!***", task.e[0], task.e[1],task.e[2],task.e[3],task.e[4],task.e[5])
                    self.bConstraint = True
                    
            if task.task_name == "jnt3_limits":
                if task.e[0] < 0 or task.e[1] > 0 or task.e[2] < 0 or task.e[3] > 0 or task.e[4] < 0 or task.e[5] > 0:
                    print("*************jnt3_limits violated!***", task.e[0], task.e[1],task.e[2],task.e[3],task.e[4],task.e[5])
                    self.bConstraint = True    
                
            if task.task_name == "self_collide" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************self_collide violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    
            if task.task_name == "cylinder_avoidance1" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************cylinder_avoidance1 violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    
            if task.task_name == "cylinder_avoidance2" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************cylinder_avoidance2 violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
                    
            if task.task_name == "cylinder_avoidance3" and task.e[0] < 0:
                if np.abs(task.e[0]) > violate_thre:
                    print("*************cylinder_avoidance3 violated!******", task.e[0])
                    self.reward -= penalty_scale*np.abs(task.e[0])
                    self.bConstraint = True
               
    def step(self, action):
        # Execute one time step within the environment
        a = action.numpy()[0] * self.action_scale
        #act_pub = [a[0], a[1]]
        self.pub.publish(a)
        self.fresh = False
        while not self.fresh:
            self.rate.sleep()

        success, Ax, bx = qhull(self.A,self.J,self.b)
        Ax = -Ax
        if(success) :
            self.twriter.writerow(self.episode_trace[-1][0])
            self.twriter.writerow(self.episode_trace[-1][1])
            self.twriter.writerow(self.A)
            self.twriter.writerow(self.b)
            self.twriter.writerow(self.J)
            self.twriter.writerow(action.numpy()[0])
            self.twriter.writerow(self.observation)
            self.twriter.writerow(self.ddq_star)
            self.twriter.writerow(self.rhs)
            bx = bx - Ax.dot(self.rhs).transpose() 
            #we should be checking the actiuons were feasible according to previous set of constraints
            feasible = self.episode_trace[-1][0].dot(action.numpy()[0] * self.action_scale) - self.episode_trace[-1][1]
            n_infeasible = np.sum(feasible>0.001)
            self.episode_trace.append((Ax,bx,n_infeasible))

        if self.bConstraint:
            done = True
        else:
            self.reward, done = self.calc_shaped_reward()

        return self.observation, self.reward, done, Ax, bx

    def stop(self):
        self.bConstraint = False
        self.episode_trace.clear()
        self.episode_trace = [(np.identity(self.action_space.shape[0]),self.action_space.high,0)]
        joints = ['three_dof_planar_joint1', 'three_dof_planar_joint2', 'three_dof_planar_joint3']
        if self.bEffort:
            remove_tasks = rospy.ServiceProxy('/hiqp_joint_effort_controller/remove_tasks', RemoveTasks)
            #remove_tasks = rospy.ServiceProxy('/hiqp_joint_effort_controller/remove_all_tasks', RemoveAllTasks)
        else:
            remove_tasks = rospy.ServiceProxy('/hiqp_joint_velocity_controller/remove_tasks', RemoveTasks)
            #remove_tasks = rospy.ServiceProxy('/hiqp_joint_velocity_controller/remove_all_tasks', RemoveAllTasks)
        remove_tasks(['ee_rl'])
        #remove_tasks()
        if self.sub is not None:
            self.sub.unregister()
        cs = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        if self.bEffort:
            resp = cs({'position_joint_trajectory_controller'},{'hiqp_joint_effort_controller'},2,True,0.1)
            self.effort_pub.publish(JointTrajectory(joint_names=joints, points=[
                JointTrajectoryPoint(positions=[0.3, -0.8,-1.3], time_from_start=rospy.Duration(4.0))]))
        else:
            resp = cs({'velocity_joint_trajectory_controller'},{'hiqp_joint_velocity_controller'},2,True,0.1)
            self.velocity_pub.publish(JointTrajectory(joint_names=joints,points=[
                JointTrajectoryPoint(positions=[0.3,-0.8,-1.3],time_from_start=rospy.Duration(4.0))]))

    def start(self):
        cs = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        if self.bEffort:
            resp = cs({'hiqp_joint_effort_controller'},{'position_joint_trajectory_controller'},2,True,0.1)
            #hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_effort_controller/set_tasks', SetTasks)
        else:
            resp = cs({'hiqp_joint_velocity_controller'},{'velocity_joint_trajectory_controller'},2,True,0.1)
            #hiqp_task_srv = rospy.ServiceProxy('/hiqp_joint_velocity_controller/set_tasks', SetTasks)
        #rl_task = Task(name='ee_rl',priority=2,visible=True,active=True,monitored=True,
        #               def_params=['TDefRL2DSpace','1','0','0','0','1','0','ee_point'],
        #               dyn_params=['TDynAsyncPolicy', '{}'.format(self.kd), 'ee_rl/act', 'ee_rl/state'])
        #hiqp_task_srv([rl_task])
        self.set_tasks()
        #wait for fresh state
        self.fresh = False
        #queue size = 1 only keeps most recent message
        self.sub = rospy.Subscriber("/ee_rl/state", StateMsg, self._next_observation, queue_size=1)

        while not self.fresh:
            self.rate.sleep()
        return self.observation  # reward, done, info can't be included

    def reset_vel(self):
        self.episode_trace.clear()
        self.episode_trace = [(np.identity(self.action_space.shape[0]),self.action_space.high,0)]

        cs = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        cs_unload = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
        cs_load = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        remove_tasks = rospy.ServiceProxy('/hiqp_joint_velocity_controller/remove_all_tasks', RemoveAllTasks)
        #print('removing tasks')
        remove_tasks()
        resp = cs({'velocity_joint_trajectory_controller'},{'hiqp_joint_velocity_controller'},2,True,0.1)
        cs_unload('hiqp_joint_velocity_controller')
        print('setting to home pose')
        joints = ['three_dof_planar_joint1','three_dof_planar_joint2','three_dof_planar_joint3']
        self.velocity_pub.publish(JointTrajectory(joint_names=joints,points=[JointTrajectoryPoint(positions=[0.3,-0.3,-0.25],time_from_start=rospy.Duration(4.0))]))
        time.sleep(4.5)
        #restart hiqp
        cs_load('hiqp_joint_velocity_controller')
        #print("restarting controller")
        resp = cs({'hiqp_joint_velocity_controller'},{'velocity_joint_trajectory_controller'},2,True,0.1)
        #set tasks to controller
        self.set_primitives()
        self.set_tasks()

        #wait for fresh state
        self.fresh = False
        while not self.fresh:
            self.rate.sleep()
        return self.observation  # reward, done, info can't be included

    def reset(self):

        if not self.bEffort:
            return self.reset_vel()

        self.episode_trace.clear()
        self.episode_trace = [(np.identity(self.action_space.shape[0]),self.action_space.high,0)]
        # Reset the state of the environment to an initial state

        #print("Resetting environment")
        cs = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        cs_unload = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
        cs_load = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        remove_tasks = rospy.ServiceProxy('/hiqp_joint_effort_controller/remove_all_tasks', RemoveAllTasks)
        #print('removing tasks')
        remove_tasks()
        #time.sleep(1)

        #stop hiqp
        #print('switching controller')
        resp = cs({'position_joint_trajectory_controller'},{'hiqp_joint_effort_controller'},2,True,0.1)
        #time.sleep(0.2)
        cs_unload('hiqp_joint_effort_controller')

        #print('setting to home pose')
        joints = ['three_dof_planar_joint1','three_dof_planar_joint2','three_dof_planar_joint3']
        self.effort_pub.publish(JointTrajectory(joint_names=joints,points=[JointTrajectoryPoint(positions=[0.3,-0.3,-0.25],time_from_start=rospy.Duration(4.0))]))
        time.sleep(4.5)
        #restart hiqp
        cs_load('hiqp_joint_effort_controller')

        #print("restarting controller")
        resp = cs({'hiqp_joint_effort_controller'},{'position_joint_trajectory_controller'},2,True,0.1)
        #set tasks to controller
        self.set_primitives()
        self.set_tasks()
        #self.pub.publish([0,0])
        #wait for fresh state
        self.fresh = False
        while not self.fresh:
            self.rate.sleep()

        #print("Now acting")

        return self.observation  # reward, done, info can't be included
         
    def render(self, mode='human'):
        pass
        #self.twriter.writerow(self.A.tolist())
        #self.twriter.writerow(self.J.tolist())
        #self.twriter.writerow(self.b.tolist())

    def close (self):
        pass

    def calc_dist(self):
        dist = np.linalg.norm(self.e-self.goal) #   math.sqrt((self.observation[0]-self.goal[0]) ** 2 + (self.observation[1]-self.goal[1])  ** 2)
        return dist

    def calc_shaped_reward(self):
        reward = 0
        done = False

        dist = self.calc_dist()

        if dist < 0.02:
            reward += 500
            print("--- Goal reached!! ---")
            done = True
        else:
            reward += -10*dist

        return reward, done
        
