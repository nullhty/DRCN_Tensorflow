# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def weight_variable(shape, name):
    weight = tf.get_variable(name, shape, initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)
    return weight
    
    
def bias_variable(shape, name):
    bias = tf.get_variable(name, shape, initializer=tf.zeros_initializer(), dtype=tf.float32)
    return bias
    
    
def relu(x):
    return tf.nn.relu(x)
    
    
def lrelu(x, alpha=0.05):
    return tf.nn.leaky_relu(x, alpha)
    
    
def conv2d(x, shape, name, stride=[1, 1, 1, 1], pad='SAME', act='lrelu', alpha=0.05, use_bias=True):
    w_name = name + '_w'
    b_name = name + '_b'
    weight = weight_variable(shape, w_name)
    
    y = tf.nn.conv2d(x, weight, strides=stride, padding=pad)
    if use_bias is True:
        bias = bias_variable(shape[3], b_name)
        y = y + bias
    
    if act == 'relu':
        y = relu(y)
    elif act == 'lrelu':
        y = lrelu(y, alpha)

    return y, weight

    
def DRCN(input_img, recusive_time):
    ksize = 256#原文使用了256
    
    temp, weight_conv1 = conv2d(input_img, [3, 3, 1, ksize], 'conv1', stride=[1, 1, 1, 1], pad='SAME', act='relu', use_bias=True)
    temp, weight_conv2 = conv2d(temp, [3, 3, ksize, ksize], 'conv2', stride=[1, 1, 1, 1], pad='SAME', act='relu', use_bias=True)
    
    weight_recursive = weight_variable(shape=[3, 3, ksize, ksize], name='weight_recursive')
    bias_recursive = bias_variable(shape=[ksize], name='bias_recursive')
    H = recusive_time * [None]

    for i in range(recusive_time):
        temp = tf.nn.conv2d(temp, weight_recursive, strides=[1, 1, 1, 1], padding='SAME')
        temp = temp + bias_recursive
        temp = tf.nn.relu(temp)
        H[i] = temp

    weight_reconstruction1 = weight_variable(shape=[3, 3, ksize, ksize], name='weight_reconstruction1')
    bias_reconstruction1 = bias_variable(shape=[ksize], name='bias_reconstruction1')
    weight_reconstruction2 = weight_variable(shape=[3, 3, ksize, 1], name='weight_reconstruction2')
    bias_reconstruction2 = bias_variable(shape=[1], name='bias_reconstruction2')

    W = tf.Variable(np.full(fill_value=1.0 / recusive_time, shape=[recusive_time], dtype=np.float32), name="LayerWeights")
    W_sum = tf.reduce_sum(W)
    output_list = recusive_time * [None]
    for i in range(recusive_time):
        temp = tf.nn.conv2d(H[i], weight_reconstruction1, strides=[1, 1, 1, 1], padding='SAME')
        temp = temp + bias_reconstruction1
        #temp = tf.nn.relu(temp)
        #temp = tf.concat([temp, input_img], 3)
        temp = tf.nn.conv2d(temp, weight_reconstruction2, strides=[1, 1, 1, 1], padding='SAME')
        temp = temp + bias_reconstruction2
        #temp = tf.nn.relu(temp)
        H[i] = temp# + input_img
        #这里怀疑原文的方法或者表述存在问题，
        # 1.如果这里用了残差，效果很差 2.如果重建阶段使用relu也会很差（可以去除下面两个+input_img）
        #由于原文没有公布训练程序，所以难以考证
        output_list[i] = H[i]*W[i]/W_sum
        H[i] = H[i] + input_img

    output = tf.add_n(output_list)
    output = output + input_img
    l2_norm = tf.nn.l2_loss(weight_conv1) + tf.nn.l2_loss(weight_conv2) + tf.nn.l2_loss(weight_recursive) + \
              tf.nn.l2_loss(weight_reconstruction1) + tf.nn.l2_loss(weight_reconstruction2)
    return output, H, l2_norm