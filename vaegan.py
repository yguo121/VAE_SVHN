"""
Project Name: Variational Autoencoder (Improved Model)
Author: Yue Jin, Dinghui Li
Date: Dec 18, 2017

Reference: 
    https://github.com/jlindsey15/VAEGAN
    

"""


# Import Packages
from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt

from prettytensor import pretty_tensor_image_methods
import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
from matplotlib import pyplot as plt
from deconv import deconv2d
#from progressbar import ETA, Bar, Percentage, ProgressBar
from dataset import *




#%%
flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 50, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 1000, "max epoch")
flags.DEFINE_float("g_learning_rate", 1e-2, "learning rate")
flags.DEFINE_float("d_learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_float("hidden_size", 10, "hidden size")
flags.DEFINE_float("gamma", 1, "gamma")

FLAGS = flags.FLAGS

#%%
def encoder(input_tensor):
    '''Create encoder network.
        
        Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]
        
        Returns:
        A tensor that expresses the encoder network
        '''
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 32, 32, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor


def discriminator(input_features):
    '''Create a network that discriminates between images from a dataset and
    generated ones.

    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''
    return  input_features.fully_connected(1, activation_fn=None).tensor


def discriminator_features(input_tensor):
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 32, 32, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten())



def get_discrinator_loss(D1, D2):
    '''Loss for the discriminator network

    Args:
        D1: logits computed with a discriminator networks from real images
        D2: logits computed with a discriminator networks from generated images

    Returns:
        Cross entropy loss, positive samples have implicit labels 1, negative 0s
    '''
    return tf.reduce_mean(tf.nn.relu(D1) - D1 + tf.log(1.0 + tf.exp(-tf.abs(D1)))) + \
        tf.reduce_mean(tf.nn.relu(D2) + tf.log(1.0 + tf.exp(-tf.abs(D2))))



def generator(input_tensor):
    '''Create a network that generates images
    TODO: Add fixed initialization, so we can draw interpolated images

    Returns:
        A deconvolutional (not true deconv, transposed conv2d) network that
        generated images.
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
    mean = input_tensor[:, :FLAGS.hidden_size]
    stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:]))
    input_sample = mean + epsilon * stddev
    input_sample = tf.reshape(input_sample, [FLAGS.batch_size, 1, 1, -1])
    return (pt.wrap(input_sample).
            deconv2d(8, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=1,edges='VALID').
            deconv2d(4, 1, stride=2, activation_fn=tf.nn.sigmoid)).tensor


#%%
def binary_crossentropy(t,o):
    return -(t*tf.log(o+1e-9) + (1.0-t)*tf.log(1.0-o+1e-9))

def get_generator_loss(D2):
    '''Loss for the genetor. Maximize probability of generating images that
    discrimator cannot differentiate.

    Returns:
        see the paper
    '''
    return tf.reduce_mean(tf.nn.relu(D2) - D2 + tf.log(1.0 + tf.exp(-tf.abs(D2))))


#%%
    
# Load Data
dataset = SVHNDataset('.')


input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 32 * 32])


with pt.defaults_scope(activation_fn=tf.nn.elu,
                   batch_normalize=True,
                   learned_moments_update_rate=0.0003,
                   variance_epsilon=0.001,
                   scale_after_normalization=True):
    with tf.variable_scope("encoder"):
        encoding = encoder(input_tensor)
    E_params_num = len(tf.trainable_variables())
    with tf.variable_scope("model"):
        input_features = discriminator_features(input_tensor)  # positive examples
        D1 = discriminator(input_features)
        input_features = input_features.tensor
        D_params_num = len(tf.trainable_variables())
        G = generator(encoding)


    with tf.variable_scope("model", reuse=True):
        gen_features = discriminator_features(G)  # positive examples
        D2 = discriminator(gen_features)
        gen_features = gen_features.tensor
    

#%%            
reconstruction_loss = binary_crossentropy(tf.sigmoid(input_features), tf.sigmoid(gen_features))
reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, 1))
D_loss = get_discrinator_loss(D1, D2)
G_loss = FLAGS.gamma * reconstruction_loss + get_generator_loss(D2)

learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0)
params = tf.trainable_variables()
E_params = params[:E_params_num]
D_params = params[E_params_num:D_params_num]
G_params = params[D_params_num:]
train_encoder = pt.apply_optimizer(optimizer, losses=[reconstruction_loss], regularize=True, include_marked=True, var_list=E_params)
train_discrimator = pt.apply_optimizer(optimizer, losses=[D_loss], regularize=True, include_marked=True, var_list=D_params)
train_generator = pt.apply_optimizer(optimizer, losses=[G_loss], regularize=True, include_marked=True, var_list=G_params)
#%%
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(FLAGS.max_epoch):
    
        discriminator_loss = 0.0
        generator_loss = 0.0
        encoder_loss = 0.0
    
        #widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
        #pbar = ProgressBar(FLAGS.updates_per_epoch, widgets=widgets)
        #pbar.start()
        for i in range(FLAGS.updates_per_epoch):
        #    pbar.update(i)
            x = dataset.next_batch(FLAGS.batch_size).reshape(FLAGS.batch_size,32*32)
            
            _, loss_value = sess.run([train_encoder, reconstruction_loss], {input_tensor: x, learning_rate: FLAGS.g_learning_rate})
            
            encoder_loss += loss_value
            
            _, loss_value = sess.run([train_discrimator, D_loss], {input_tensor: x, learning_rate: FLAGS.d_learning_rate})
            discriminator_loss += loss_value
    
            # We still need input for moving averages.
            # Need to find how to fix it.
            _, loss_value, imgs = sess.run([train_generator, G_loss, G], {input_tensor: x, learning_rate: FLAGS.g_learning_rate})
            generator_loss += loss_value
    
        discriminator_loss = discriminator_loss / FLAGS.updates_per_epoch
        generator_loss = generator_loss / FLAGS.updates_per_epoch
        encoder_loss = encoder_loss / FLAGS.updates_per_epoch
        print("Epoc %d: Enc. loss %4.2f, Gen. loss %4.2f, Disc. loss %4.2f, Total %4.2f" % (epoch, encoder_loss, generator_loss,
                                                discriminator_loss, encoder_loss + generator_loss + discriminator_loss))
        # Plot
        imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)
        img = plt.figure(figsize=(8, 12))
        for i in range(5):
            plt.subplot(5, 2, 2*i + 1)
            plt.imshow(x[i].reshape(32, 32), vmin=0, vmax=1, cmap="gray")
            plt.title("Test input")
            plt.colorbar()
            plt.subplot(5, 2, 2*i + 2)
            plt.imshow(imgs[i].reshape(32, 32), vmin=0, vmax=1, cmap="gray")
            plt.title("Reconstruction")
            plt.colorbar()
        plt.suptitle('Epoc %d' %epoch)
        img.savefig(os.path.join(imgs_folder, '%d.png')  %epoch, dpi = img.dpi)   