# -*- coding: utf-8 -*-
# Author: Hsin-Yu
# environment: Ubuntu 16.04 + python 2.7 + Tensorflow 0.12.1(GPU enabled)
# implement '2013-Playing Atari with Deep Reinforcement Learning'

from DQN_Network import *
import sys
import gym
from gym import wrappers
import cv2
import numpy as np
import tensorflow as tf


# Hyper Parameters
MAXSTEP = 10000         # Step limitation in an episode


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    return np.reshape(observation, (84, 84, 1))


def main():
    if len(sys.argv) != 2:
        print "Usage:", sys.argv[0], "<Game Name>"
        sys.exit(1)
    Game_Name = sys.argv[1]
    EXDIR = "experiment" + "/" + Game_Name  # experiment directory
    env = gym.make(Game_Name)
    env = wrappers.Monitor(env,EXDIR,force=True)    # log performance
    agent = DQN(env,Game_Name)                

    for episode in xrange(EPISODE):
        print "=== Episode %d ===============" % episode
        s1 = preprocess(env.reset())            # initial state
        for step in xrange(MAXSTEP):
            action = agent.eGreedy(s1)
            s2,r,term,_ = env.step(action)
            s2 = preprocess(s2)
            if term:                            # adjust rewards
                reward = -1.0
            else:
                reward = +1.0
            agent.perceive(s1,action,reward,s2,term)
            s1 = s2
            if term:
                break


    print "=== Process Finished ==="


if __name__ == '__main__':
    main()
