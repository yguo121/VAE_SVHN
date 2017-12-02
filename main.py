#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:37:56 2017

@author: yuejin
"""

from __future__ import division
from __future__ import print_function
import os.path

import tensorflow as tf
import numpy as np

import scipy.io as sio
import matplotlib
from dataset import *
from sys import exit

dataset = SVHNDataset()

input_dim = 1024
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

x = tf.placeholder("float", shape=[None, input_dim])
l2_loss = tf.constant(0.0)

W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

# Hidden layer encoder
hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

'''
Gaussian MLP?
'''
# Mu encoder
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.multiply(std_encoder, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

# KL divergence

KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction

'''
Computes sigmoid cross entropy given `logits`.
What is logits?
'''
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, targets=x), reduction_indices=1)

loss = tf.reduce_mean(BCE + KLD)

# regularized_loss = 0 + loss?
regularized_loss = loss + lam * l2_loss

'''
Outputs a `Summary` protocol buffer containing a single scalar value.
The generated Summary has a Tensor.proto containing the input Tensor.
tf.summary.scalar(name, tensor, collections=None)
name = generated node, will also serve as the series name in TensorBoard
tensor = real numeric Tensor containing a single value
'''
loss_summ = tf.summary.scalar("lowerbound", loss)

# Adam: a combined gradient descent strategy

# 0.01 - learning rate
'''
tf.train.AdamOptimizer(learning rate)
tf.train.AdamOptimizer.minimize(self,loss)
loss = a tensor containing the value to minimize
'''
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()

n_steps = int(2e3)
batch_size = 100
      
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(1, n_steps):
    batch = dataset.next_batch(batch_size)
    feed_dict = {x: np.transpose(batch[0].reshape((1024,3)),(1,0))}
    _, cur_loss=sess.run([train_step, loss], feed_dict=feed_dict)
#    summary_writer.add_summary(summary_str, step)

    if step % 50 == 0:
    #  save_path = saver.save(sess, "save/model.ckpt")
      print("Step {0} | Loss: {1}".format(step, cur_loss))      


