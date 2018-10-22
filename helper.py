import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


#This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy
  
  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)
  
  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration,verbose=False)

def set_image_daw(values,probs,state,trial,state_cnt,reward_dist):
    daw_image = Image.open('./resources/daw.png').convert('RGB')
    draw = ImageDraw.Draw(daw_image)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)

    draw.text((0, 0),'Trial: ' + str(trial),(0,0,0),font=font)  
    total_reward = sum(values)
    draw.text((0,30),'Total Reward: ' + str(total_reward), (0,0,255), font=font)

    #plot state transition probability
    draw.text((114, 250),'Prob: ',(255,0,0),font=font)
    draw.text((340, 250),str(float("{0:.2f}".format(probs[0][0]))),(255,0,0),font=font)
    draw.text((500, 250),str(float("{0:.2f}".format(probs[0][1]))),(255,0,0),font=font)
    draw.text((660, 250),str(float("{0:.2f}".format(probs[1][0]))),(255,0,0),font=font)
    draw.text((820, 250),str(float("{0:.2f}".format(probs[1][1]))),(255,0,0),font=font)

    #plot reward distribution
    draw.text((114, 350),'Reward: ',(0,0,0),font=font)
    draw.text((340, 350),str("%d" % reward_dist[0][0]),(0,0,0),font=font) 
    draw.text((500, 350),str("%d" % reward_dist[0][1]),(0,0,0),font=font) 
    draw.text((660, 350),str("%d" % reward_dist[1][0]),(0,0,0),font=font) 
    draw.text((820, 350),str("%d" % reward_dist[1][1]),(0,0,0),font=font)

    #plot # of state arrive
    draw.text((114, 450), 'num(#): ', (0,0,0), font=font)
    draw.text((340, 450),str("%d" % state_cnt[0]),(0,0,0),font=font) 
    draw.text((500, 450),str("%d" % state_cnt[1]),(0,0,0),font=font)  
    draw.text((660, 450),str("%d" % state_cnt[2]),(0,0,0),font=font)  
    draw.text((820, 450),str("%d" % state_cnt[3]),(0,0,0),font=font)

    #plot bar graph - total rewards in episode
    sign_list = []
    sign_list += (np.sign(reward_dist[0])).tolist()
    sign_list += (np.sign(reward_dist[1])).tolist()
    color_list = []
    for i in range(len(sign_list)):
        if(sign_list[i] == 1):
            color_list.append([0,255.0,0])
        elif(sign_list[i] == -1):
            color_list.append([255.0,0,0])
        else:
            color_list.append([0,0,0])

    daw_image = np.array(daw_image)
    daw_image[500:500+int(np.floor(abs(values[0]))),310:370,:] = color_list[0]
    daw_image[500:500+int(np.floor(abs(values[1]))),470:530,:] = color_list[1]
    daw_image[500:500+int(np.floor(abs(values[2]))),630:690,:] = color_list[2]
    daw_image[500:500+int(np.floor(abs(values[3]))),790:850,:] = color_list[3]
    daw_image[480:490,310+(state*160):310+(state*160)+60,:] = [80.0,80.0,225.0]

    return daw_image
