# -*- coding: utf-8 -*-
from utility import *
import sys
import time
import shutil
import os
import h5py
import numpy as np
import math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        print(train_data.shape)
        print(train_label.shape)
    return train_data, train_label


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 255.0
    return 10.0 * tf_log10((max_pixel ** 2) / (tf.reduce_mean(tf.square(y_pred - y_true))))

        
def DRCN_train(train_data_file,test_data_file,model_save_path):
    train_data, train_label = read_data(train_data_file)
    test_data, test_label = read_data(test_data_file)
    
    batch_size = 64
    iterations = train_data.shape[0]//batch_size//2

    #lr = 0.01
    momentum_rate = 0.9
    image_size = 41
    label_size = 41
    recusive = 16
    is_load = True#是否加载现有模型进行再训练
    per_epoch_save = 1
    start_epoch = 0
    alpha_init = 1.0
    alpha_decay = 25
    total_epoch = 55 + alpha_decay#这里是指还需要继续训练多少个epoch
    beta = 0.0001
    images = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='images')
    labels = tf.placeholder(tf.float32, [None, label_size, label_size, 1], name='labels')
    learning_rate = tf.placeholder(tf.float32)
    alpha = tf.placeholder(tf.float32, shape=[], name="alpha")

    pred, recusive_maps, l2_norm = DRCN(images, recusive)
    loss1 = tf.reduce_mean(tf.square(labels - pred))
    
    rec_loss = recusive * [None]
    for i in range(0, recusive):
        rec_loss[i] = tf.reduce_mean(tf.square(labels - recusive_maps[i]))
    loss2 = tf.add_n(rec_loss)*(1.0/recusive)

    if alpha == 0.0:
        beta = 0.0
    #这里参考了github上开源项目的设置，原文这里也没有交代清楚

    loss = loss1*(1-alpha) + loss2*alpha + l2_norm*beta
    
    psnr = PSNR(labels, pred)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.1, beta2=0.1)#.minimize(loss)#beta1=0.1, beta2=0.1
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate,use_nesterov=True).minimize(loss)

    #前30没有梯度裁剪
    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads, global_norm = tf.clip_by_global_norm(grads, 1.0 / learning_rate)
    train_step = optimizer.apply_gradients(zip(grads, variables))
    saver = tf.train.Saver()
        
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())     
        if is_load:
            start_epoch = 60
            check_point_path = model_save_path + '/' + str(start_epoch) + '/'# 保存好模型的文件路径
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        bar_length = 30
        for ep in range(1+start_epoch, total_epoch+1):
            input_alpha = alpha_init - (ep-1) * (1.0/alpha_decay)
            input_alpha = max(0.0, input_alpha)
            if ep <= 15 + alpha_decay:
                lr = 0.001
            elif ep <= 35 + alpha_decay:
                lr = 0.0001
            else:
                lr = 0.00001
            start_time = time.time()
            pre_index = 0
            train_loss = 0.0
            train_psnr = 0.0
            print("\nepoch %d/%d, lr = %2.5f, alpha = %2.5f:" % (ep, total_epoch, lr, input_alpha))
            indices = np.random.permutation(len(train_data))#每次随机打乱数据
            train_data = train_data[indices]
            train_label = train_label[indices]
            for it in range(1, iterations+1):
                batch_x = train_data[pre_index:pre_index+batch_size]
                batch_y = train_label[pre_index:pre_index+batch_size]
                _, batch_loss, batch_psnr = sess.run(
                    [train_step, loss, psnr], feed_dict={images: batch_x, labels: batch_y,
                                                         learning_rate: lr, alpha: input_alpha})
                
                train_loss += batch_loss
                train_psnr += batch_psnr
                pre_index += batch_size
                
                if it == iterations:
                    train_loss /= iterations
                    train_psnr /= iterations
                    test_loss, test_psnr = sess.run([loss, psnr], feed_dict={images: test_data, labels: test_label,
                                                                              alpha: input_alpha})
                    
                    s1 = "\r%d/%d [%s%s] - batch_time = %.2fs - train_loss = %.5f - train_psnr = %.2f" % \
                         (it, iterations, ">"*(bar_length*it//iterations), "-"*(bar_length-bar_length*it//iterations),
                          (time.time()-start_time)/it, train_loss, train_psnr)#run_test()
                    sys.stdout.write(s1)
                    sys.stdout.flush()
                    
                    print("\ncost_time: %ds, test_loss: %.5f, test_psnr: %.2f" %(int(time.time()-start_time), test_loss, test_psnr))
                    '''
                    这里输出的test_psnr并不是最终Set5的psnr，而是图像块的平均值
                    '''
                else:
                    s1 = "\r%d/%d [%s%s] - batch_time = %.2fs - train_loss = %.5f - train_psnr = %.2f" % \
                         (it, iterations, ">"*(bar_length*it//iterations), "-"*(bar_length-bar_length*it//iterations),
                          (time.time()-start_time)/it, train_loss / it, train_psnr / it)#run_test()
                    sys.stdout.write(s1)
                    sys.stdout.flush()
            if ep % per_epoch_save == 0:
                path = model_save_path + '/save/' + str(ep) + '/'
                save_model = saver.save(sess, path + 'DRCN_model')
                new_path = model_save_path + '/' + str(ep) + '/'
                shutil.move(path, new_path)
                '''
                模型首先是被保存在save下面的,直接保存的话，前面的epoch对应的文件夹会出现内部文件被删除的情况，原因不明；所以这里用shutil.move把模型所在的文件夹移动了一下
                '''
                print("\nModel saved in file: %s" % save_model)
        path = './final_model/DRCN_model'
        save_model = saver.save(sess, path)
        print("\nModel saved in file: %s" % save_model)

    
def main():
    train_file = 'train.h5'
    test_file = 'test2.h5'
    model_save_path = 'DRCN_checkpoint'
    
    if os.path.exists(model_save_path) == False:
        print('The ' + '"' + model_save_path + '"' + 'can not find! Create now!')
        os.mkdir(model_save_path)
        
    if os.path.exists(model_save_path + '/save') == False:
        print('The ' + '"save' + '"' + ' can not find! Create now!')
        os.mkdir(model_save_path+'/save')
        
    DRCN_train(train_file, test_file, model_save_path)

    
if __name__ == '__main__':
    main()