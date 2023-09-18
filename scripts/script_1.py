"""script_1 should implement the functions below

The script is meant to test the straight line functionality. Feel free to add any utilities that you need here either from your utils or from our utils
"""
import medra_robotics.arm_control.xarm_constants as consts

import time
from medra_robotics.scripts.script_utils import *
from medra_robotics.utils import transformations


"""
This is rrt star code for 3D
@author: yue qi
"""
import numpy as np
from numpy.matlib import repmat
from collections import defaultdict
import time
import matplotlib.pyplot as plt

import os
import sys


class rrt():
    def __init__(self):
        self.Parent = {}
        self.V = []
        self.i = 0
        self.maxiter = 1000
        self.stepsize = 50
        self.Path = []
        self.done = False
        self.xt = tuple( np.array([0.0,0.0,0.0]))
        self.x0 = tuple( np.array([0.0,0.0,0.0]))
        self.block = [[0, 6, 0, 50, 50, 50]]
        self.blocks = getblocks(self.block)
        self.AABB = getAABB2(self.blocks)
        self.AABB_pyrr = getAABB(self.blocks)
        # resolution=1
        self.boundary = np.array([-800, -800, 0, 800, 800, 800]) 
        self.t = 0 

        self.ind = 0
        # self.fig = plt.figure(figsize=(10, 8))

    def wireup(self, x, y):
        # self.E.add_edge([s, y])  # add edge
        self.Parent[x] = y

    def run(self,fp,sp):
        self.xt = tuple(fp)
        self.x0 = tuple(sp)
        self.V.append(self.x0)
        while self.ind < self.maxiter:
            xrand = sampleFree(self)
            xnearest = nearest(self, xrand)
            xnew, dist = steer(self, xnearest, xrand)
            #print(xnew,dist)
            collide, _ = isCollide(self, xnearest, xnew, dist=dist)
            if not collide:
                self.V.append(xnew)  # add point
                self.wireup(xnew, xnearest)

                if getDist(xnew, self.xt) <= self.stepsize:
                    self.wireup(self.xt, xnew)
                    self.Path, D = path(self)
                    print('Total distance = ' + str(D))
                    break
                self.i += 1
            self.ind += 1
            # if the goal is really reached
            
        self.done = True
        return self.Path


def move_around_obstacle(arm, **kwargs):

    # start = (370, -67, 481)
    # goal = (370, 300, 481)
    # obstacles = get_cube_vertices([0,1.2,0],500)
    p = rrt()
    starttime = time.time()
    first_pose  = kwargs['first_pose']
    second_pose = kwargs['second_pose']
    path = p.run(first_pose[:3],second_pose[:3])
    quat3 = np.zeros(shape=(len(path), 3))
    print('time used = ' + str(time.time() - starttime))
    # print(quat3)
    # Run the A* algorithm to find the path
    quat1 = transformations.quat_from_euler(first_pose[3:])
    quat2 =  transformations.quat_from_euler(second_pose[3:])
    for i in range(len(path)):
        k = transformations.quaternion_slerp(quat1,quat2,i*(1/(len(path)-1)))
        quat3[i] = transformations.euler_from_quat(k)

    if path is None:
        print("No path found")
    else:
        for i in range(len(path)):

                path[i] =np.hstack([path[i], quat3[i]])
                path[i] = arm.get_inverse_kinematics(path[i])
    return path
