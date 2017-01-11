# -*- coding: utf-8 -*-
# Author: Hsin-Yu
# environment: Ubuntu 16.04 + python 2.7 + Tensorflow 0.12.1(GPU enabled)
# implement '2013-Playing Atari with Deep Reinforcement Learning'

import tensorflow as tf
import numpy as np
import random
from collections import deque


# Hyper Parameters
discount = 0.99             # discount factor
EPISODE = 10000             # Episode limitation
Memory_Size = 1000000       # Replay memory max size
minibatch_size = 32         # max mini batch size
SaverStep = 200	            # how many timestep to save weight


class DQN():
    def __init__(self, env , Game_Name):
        self.Time_Step = 0                                      # init global time step
        self.action_dim = env.action_space.n                    # action number
        self.replay_memory = deque([], maxlen=Memory_Size)      # replay memory
        self.gamename = Game_Name                               # Game Name
        # epsilon annealing
        self.ep_start = 1.0
        self.ep_end = 0.1
        self.epsilon = self.ep_start
        # learning rate
        self.lr = 0.01
        # init deep network
        self.createQNetwork()
        self.createTrainingMethod()
        # init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # init savr
        self.SaverDir = "checkpoint/" + Game_Name + "/" # checkpoint directory
        self.saver = tf.train.Saver()
        chkpt = tf.train.get_checkpoint_state(self.SaverDir)
        if chkpt and chkpt.model_checkpoint_path:
            print "Loading weight:" , chkpt.model_checkpoint_path
            self.saver.restore(self.sess , chkpt.model_checkpoint_path)
        else:
            print "First running"


    def createQNetwork(self):
        with tf.device('/gpu:0'):
            self.stateInput = tf.placeholder("float",[None,84,84,1])
            # layer 1 (convolution 1)
            W_conv1 = self.weight_variable([8,8,1,32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(self.stateInput ,W_conv1, 4) + b_conv1)
            # layer 2 (convolution 2)
            W_conv2 = self.weight_variable([4,4,32,64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)
            # layer 3 (convolution 3)
            W_conv3 = self.weight_variable([3,3,64,64])
            b_conv3 = self.bias_variable([64])
            h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
            h_conv3_flat = tf.reshape(h_conv3,[-1,7*7*64])
            # layer 4 (fully-connection 1)
            W_fc1 = self.weight_variable([7*7*64,512])
            b_fc1 = self.bias_variable([512])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
            # layer 5 (fully-connection 2)
            W_fc2 = self.weight_variable([512, self.action_dim]) 	
            b_fc2 = self.bias_variable([self.action_dim])
            self.QValue = tf.matmul(h_fc1,W_fc2) + b_fc2


    def createTrainingMethod(self):
        with tf.device('/gpu:0'):
            self.y_input = tf.placeholder("float",[None])
            self.action_input = tf.placeholder("float",[None,self.action_dim])
            q_sa = tf.reduce_sum(tf.mul(self.QValue,self.action_input),reduction_indices = 1)   # Q(s,a)
            self.cost = tf.reduce_mean(tf.square(self.y_input - q_sa))                          # cost = (y-Q(s,a))^2
            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.cost)             # RMSProp
            

    def getQUpdate(self):
        # obtain random minibatch
        size = min(len(self.replay_memory) , minibatch_size)
        batch = random.sample(self.replay_memory,size)
        s1_batch = [data[0] for data in batch]          # put all s1 in list s1_batch
        a_batch = [data[1] for data in batch]
        r_batch = [data[2] for data in batch]
        s2_batch = [data[3] for data in batch]
        # compute y , stored in y_batch
        y_batch = []
        s2_Qvalue_batch = self.QValue.eval( feed_dict={self.stateInput: s2_batch} , session = self.sess )   # Q(s',a)
        for i in xrange(size):
            term = batch[i][4]
            y_batch.append(r_batch[i] + (1-term)*discount*np.max(s2_Qvalue_batch[i]))                       # y = r + (1-term)*gamma* max Q(s',a)
        # run optimizer
        self.optimizer.run(feed_dict = { self.y_input       : y_batch,
                                         self.stateInput    : s1_batch,
                                         self.action_input  : a_batch},
                           session=self.sess)  
        # save checkpoint
        if self.Time_Step % SaverStep == 0:
            self.saver.save(self.sess , self.SaverDir + self.gamename , global_step = self.Time_Step)



    def eGreedy(self,state):
        # choice action
        if random.random() <= self.epsilon:
            action = random.randint(0,self.action_dim - 1)
        else:
            action = self.greedy(state)
        return action


    def greedy(self,state):
        Qvector = self.QValue.eval( feed_dict={self.stateInput:[state]} , session = self.sess )[0]
        max_Q_index = np.argmax(Qvector)
        return max_Q_index


    def perceive(self, s1, a, r, s2, term):
        self.Time_Step += 1                                     # global time step +1
        # epsilon annealing
        if term == True:
            self.epsilon -= (self.ep_start - self.ep_end) / EPISODE
            print "Save weight ; step=%d , epsilon=%.4f , lr=%.4f" % (self.Time_Step, self.epsilon, self.lr)

        action = np.zeros(self.action_dim)                      # convert acton into vector (for easy training)
        action[a] = 1
        self.replay_memory.append((s1, action, r, s2, term))    # store (s,a,r,s', is terminal) to replay memory
        self.getQUpdate()                                       # call training step


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")


