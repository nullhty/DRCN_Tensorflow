# -*- coding: utf-8 -*-
import tensorflow as tf
from utility import *
import os
import scipy.io as sio
import numpy as np
import math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)


def file_name(file_dir, f):
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == f:  
                L.append(os.path.join(root, file))  
    return L


def cal_psnr(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    p = 20 * math.log10(255. / rmse)
    return p


if __name__ == '__main__':
    
    database = "Set14"
    up = 2
    root_path = './DRCN_checkpoint/'
    with tf.Session(config=config) as sess:
        
        LR_path = file_name("./" + database + "/LRX" + str(up), ".mat")
        Gnd_path = file_name("./" + database + "/Gnd", ".mat")

        start_model = 63
        step = 1
        model_num = 1
        recusive = 16
        #程序会从 start_model这个文件夹开始，按照step的步长，测试model_num个模型
        #result.txt会存放测试的结果，包括每个图片的psnr和集合上的平均值
        images = tf.placeholder(tf.float32, [1, None, None, 1], name='images')
        
        pred, _, _ = DRCN(images, recusive)
        #with open('result.txt','w') as f:#w这个参数会清空result文件，再写入；若要让内容不清空，则使用参数a
        #    f.write("******************************")
        #    f.write("\nwrite now!\n")
        for model in range(1, 1+model_num):
            p1 = 0.0
            p2 = 0.0
            check_point_path = root_path + str(start_model + step * (model-1)) + '/'
            print('*'*30)
            print('epoch : ' + str(start_model + step * (model-1)))
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            
            #with open('result.txt','a') as f:#设置文件对象
            #    f.write("\nepoch: %d\n" %(start_model + step * (model-1)))
                        
            for i in range(len(LR_path)):
                bic = sio.loadmat(LR_path[i])['im_b']
                gnd = sio.loadmat(Gnd_path[i])['im_gnd']
                shape = bic.shape
                img_bic = np.zeros((1, shape[0], shape[1], 1), dtype=float)
                img_bic[0, :, :, 0] = bic*255
                    
                pred0 = sess.run([pred], feed_dict={images: img_bic})
                output = pred0[0]
                pre1 = output
                pre1[pre1[:] > 255] = 255
                pre1[pre1[:] < 0] = 0
                img_h = pre1[0, :, :, 0]
                img_h = np.round(img_h)
                img_h2 = img_h[up:shape[0]-up , up:shape[1]-up]
                            
                bic1 = bic * 255
                bic1[bic1[:] > 255] = 255
                bic1[bic1[:] < 0] = 0
                bic1 = np.round(bic1)
                bic1 = bic1[up:shape[0]-up , up:shape[1]-up]
                
                gnd1 = gnd * 255.
                gnd1[gnd1[:] > 255] = 255
                gnd1[gnd1[:] < 0] = 0
                gnd1 = np.round(gnd1)
                gnd1 = gnd1[up:shape[0]-up , up:shape[1]-up]
                
                pp1 = cal_psnr(bic1, gnd1)
                pp2 = cal_psnr(img_h2, gnd1)
                print(str(i+1) + ". bicubic: " + str(pp1) + ", DRCN: "+str(pp2))
                #with open('result.txt','a') as f:
                #    f.write("%d. bicubic: %s, srcnn: %s\n" %(i+1, str(pp1), str(pp1)))
                p1 = p1 + pp1
                p2 = p2 + pp2
            print "bicubic psnr = " + str(p1/len(LR_path))
            print "DRCN psnr = " + str(p2/len(LR_path))
            #with open('result.txt','a') as f:
            #    f.write("bicubic psnr =%s\n" %(str(p1/len(LR_path))))
            #    f.write("srcnn psnr = %s\n" %(str(p2/len(LR_path))))