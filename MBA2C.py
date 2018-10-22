
# coding: utf-8

# # Meta Reinforcement Learning with A3C
# 
# This iPython notebook includes an implementation of the [A3C algorithm capable of Meta-RL](https://arxiv.org/pdf/1611.05763.pdf).
# 
# For more information, see the accompanying [Medium post](https://medium.com/p/b15b592a2ddf)

# In[ ]:


import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal

from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont

from helper import *
import argparse
from random import choice
from random import *
from time import sleep
from time import time


# ### Helper Functions

# In[ ]:


class Daw():
    def __init__(self):
        self.num_actions = 2
        self.episode_count = 0
        self.last_reverse = 0
        self.reverse_interval = 200
        self.rng = np.random
        self.rng.seed(9300)
        self.STP_mat = self.rng.uniform(0.8,0.8,(1000,2))
        self.STP_idx = 0
        self.reset()
        
    def reset(self):
        #initialize
        self.timestep = 0

        #probability setting
        self.state_trans_prob = [[0.9,0.1],[0.9,0.1]]
        self.state_trans_prob[0] = [self.STP_mat[self.STP_idx][0], 1-self.STP_mat[self.STP_idx][0]] 
        self.state_trans_prob[1] = [self.STP_mat[self.STP_idx][1], 1-self.STP_mat[self.STP_idx][1]] 
        #np.random.shuffle(self.state_trans_prob)

        #reward setting
        self.reward_dist = [[4,0], [0,4]]
        self.init_state = ((self.episode_count - self.last_reverse)/ self.reverse_interval ) % 2
        #self.init_state = np.random.randint(2)
        #np.random.shuffle(self.reward_dist)
        if(self.init_state == 1):
            self.reward_dist = self.reward_dist[::-1]
            self.last_reverse = self.episode_count
            self.reverse_interval = 200
            self.STP_idx += 1
        self.record_env_change = open('record_env_change','a')
        record_env_change_str = "%d\t%d\t%d\t%s\n"%(self.episode_count, self.last_reverse, self.reverse_interval, str(self.STP_mat[self.STP_idx]))
        print record_env_change_str
        self.record_env_change.write(record_env_change_str)
        self.record_env_change.close()

        
    def DoDecide(self,action):
        self.timestep +=1
        possible_prob = self.state_trans_prob[action]
        possible_reward = self.reward_dist[action]

        #sampling reward
        threshold = np.random.uniform()

        if threshold < possible_prob[0]:
            reward = possible_reward[0]
            state = action * 2
        else:
            reward = possible_reward[1]
            state = action * 2 + 1

        #check episode ends
        if self.timestep > 99:
            episode_done = True
            self.episode_count += 1
        else:
            episode_done = False
        
        return reward, episode_done, self.timestep, state

# ### Actor-Critic Network

# In[ ]:

class AC_Network():
    def __init__(self,a_size,scope,trainer):
        with tf.variable_scope(scope):
            with tf.variable_scope('A3C_part'):
                #Input and visual encoding layers
                self.prev_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)
                self.prev_actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.timestep = tf.placeholder(shape=[None,1],dtype=tf.float32)
                self.prev_actions_onehot = tf.one_hot(self.prev_actions,a_size,dtype=tf.float32)

                hidden = tf.concat([self.prev_rewards,self.prev_actions_onehot,self.timestep],1)
            
                #Recurrent network for temporal dependencies
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(48,state_is_tuple=True)
                c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
                h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
                self.state_init = [c_init, h_init]
                c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
                h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
                self.state_in = (c_in, h_in)
                rnn_in = tf.expand_dims(hidden, [0])
                step_size = tf.shape(self.prev_rewards)[:1]
                state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                    time_major=False)
                lstm_c, lstm_h = lstm_state
                self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
                rnn_out = tf.reshape(lstm_outputs, [-1, 48])
                
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                            
                #Output layers for policy and value estimations
                self.policy = slim.fully_connected(rnn_out,a_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None)
                self.value = slim.fully_connected(rnn_out,1,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None)

            with tf.variable_scope('state_pred_part'):
                ### State_Predictor_Network
                #input layer addition
                self.init_states = tf.placeholder(shape=[None,1], dtype = tf.float32)
                self.states = tf.placeholder(shape=[None],dtype=tf.int32)
                self.states_onehot = tf.one_hot(self.states,4,dtype=tf.float32)
            
                hidden_sp = tf.concat([self.prev_actions_onehot, self.init_states],1)

                #Recurrent Network
                lstm_cell_sp = tf.contrib.rnn.BasicLSTMCell(48,state_is_tuple=True)
                c_init_sp = np.zeros((1,lstm_cell.state_size.c), np.float32)
                h_init_sp = np.zeros((1,lstm_cell.state_size.h), np.float32)
                self.state_init_sp = [c_init_sp, h_init_sp]
                c_in_sp = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
                h_in_sp = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
                self.state_in_sp = (c_in_sp, h_in_sp)
                rnn_in_sp = tf.expand_dims(hidden_sp, [0])
                step_size_sp = tf.shape(self.prev_rewards)[:1]
                state_in_sp = tf.contrib.rnn.LSTMStateTuple(c_in_sp, h_in_sp)
                lstm_outputs_sp, lstm_state_sp = tf.nn.dynamic_rnn(
                    lstm_cell_sp, rnn_in_sp, initial_state=state_in_sp, sequence_length=step_size_sp,
                    time_major = False)
                lstm_c_sp, lstm_h_sp = lstm_state_sp
                self.state_out_sp = (lstm_c_sp[:1,:], lstm_h_sp[:1, :])
                rnn_out_sp = tf.reshape(lstm_outputs_sp, [-1, 48])

                #output layer for predict next state
                self.state_pred = slim.fully_connected(rnn_out, 4,
                    activation_fn = tf.nn.softmax,
                    weights_initializer = normalized_columns_initializer(0.01),
                    biases_initializer = None)
                #self.state_pred_max_idx = tf.argmax(self.state_pred,0)
                self.state_pred_max_idx = self.state_pred

            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7)*self.advantages)
                self.state_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.states_onehot, logits=self.state_pred))
                self.loss = 0.5 *self.value_loss + self.policy_loss - self.entropy * 0.05 + 10*self.state_loss
                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,50.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


# ### Worker Agent

# In[ ]:


class Worker():
    def __init__(self,game,name,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number),graph=tf.get_default_graph())

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        self.env = game
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        actions = rollout[:,0]
        rewards = rollout[:,1]
        timesteps = rollout[:,2]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:,4]
        states = rollout[:,5]
        init_states = rollout[:,6]
        
        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.prev_rewards:np.vstack(prev_rewards),
            self.local_AC.prev_actions:prev_actions,
            self.local_AC.actions:actions,
            self.local_AC.timestep:np.vstack(timesteps),
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1],
            self.local_AC.state_in_sp[0]:rnn_state[0],
            self.local_AC.state_in_sp[1]:rnn_state[1],
            self.local_AC.states:states,
            self.local_AC.init_states:np.vstack(init_states)}
        v_l,p_l,e_l,s_l, g_n,v_n,s_pred,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.state_loss,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_pred_max_idx,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        #f_s_pred = open("s_pred_log.txt","a")
        #s_pred_data = str(self.env.episode_count) + "\t" + str(s_pred[-1]) + "\n"
        #print s_pred_data
        #f_s_pred.write(s_pred_data)
        #f_s_pred.close()
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), s_l / len(rollout), g_n,v_n, s_pred / len(rollout)
        
    def work(self,gamma,sess,coord,saver,train):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print "Starting worker " + str(self.number)
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                s_pred_argmax_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = [0,0,0,0]
                episode_state_cnt = [0,0,0,0]
                episode_step_count = 0
                d = False
                r = 0
                a = 0
                t = 0
                s = 0
                self.env.reset()
                rnn_state = self.local_AC.state_init
                
                while d == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state_new = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={
                        self.local_AC.prev_rewards:[[r]],
                        self.local_AC.timestep:[[t]],
                        self.local_AC.prev_actions:[a],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    rnn_state = rnn_state_new
                    r,d,t,s = self.env.DoDecide(a)                        

                    self.record_ETRA = open('record_ETRA','a') 
                    record_ETRA_str = "%d\t%d\t%d\t%d\n"%(episode_count,t,r,a)  
                    self.record_ETRA.write(record_ETRA_str)
                    self.record_ETRA.close()

                    episode_buffer.append([a,r,t,d,v[0,0],s,self.env.init_state])
                    episode_values.append(v[0,0])
                    episode_frames.append(set_image_daw(episode_reward,self.env.state_trans_prob,s,t,episode_state_cnt,self.env.reward_dist))
                    episode_reward[a] += r
                    episode_state_cnt[s] += 1
                    total_steps += 1
                    episode_step_count += 1
                                            
                self.episode_rewards.append(np.sum(episode_reward))
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train == True:
                    v_l,p_l,e_l,s_l,g_n,v_n,s_pred = self.train(episode_buffer,sess,gamma,0.0)
                    s_pred_argmax = np.argmax(s_pred[-1])
                    s_pred_argmax_buffer.append(s_pred_argmax)
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 4000 == 0 and self.name == 'worker_0' and train == True:
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print "Saved Model"

                    #if episode_count % 800 == 0 and self.name == 'worker_0':
                    #    self.images = np.array(episode_frames)
                    #    make_gif(self.images,'./frames/image'+str(episode_count)+'.gif',
                    #        duration=len(self.images)*0.1,true_image=True)

                    mean_reward = np.mean(self.episode_rewards[-50:])
                    mean_length = np.mean(self.episode_lengths[-50:])
                    mean_value = np.mean(self.episode_mean_values[-50:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if train == True:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/State', simple_value=float(s_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        s_pred_argmax_freq = np.argmax(np.bincount(s_pred_argmax_buffer))
                        summary.value.add(tag='State_pred/best_action_pred', simple_value=float(s_pred_argmax_freq))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                    s_pred_argmax_buffer = []
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


# In[ ]:


gamma = .8 # discount rate for advantage estimation and reward discounting
a_size = 2
load_model = True
train = True
model_path = './model_meta'


# In[ ]:


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
if not os.path.exists('./frames'):
    os.makedirs('./frames')

    
with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    master_network = AC_Network(a_size,'global',None) # Generate global network
    #num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    num_workers = 1
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(Daw(),i,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,sess,coord,saver,train)
        thread = threading.Thread(target=(worker_work))
        thread.start()
        worker_threads.append(thread)
    coord.join(worker_threads)

