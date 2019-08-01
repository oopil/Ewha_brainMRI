import tensorflow as tf
import matplotlib.image as mpimg
# import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from data import *
import os
import pandas as pd
import numpy as np
import csv
import argparse

#%%

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#%%

img_folder = './tiny_train'
# img_folder = '/user/Datasets/dogs_and_cats/train'

img_list = os.listdir(img_folder)
img_list.sort()
number_of_imgs = len(img_list)


#%% 

train_list = img_list[:int(number_of_imgs*0.25)] + \
    img_list[int(number_of_imgs*0.5):int(number_of_imgs*0.75)]

val_list = img_list[int(number_of_imgs*0.25):int(number_of_imgs*0.3)] + \
    img_list[int(number_of_imgs*0.75):int(number_of_imgs*0.8)]

test_list = img_list[int(number_of_imgs*0.3):int(number_of_imgs*0.5)] + \
    img_list[int(number_of_imgs*0.8):]


#%%

def dataloader(n_fold, i_fold):
    subj_list, data, label = SINCHON_NN_reader()
    t_set = split_data_by_fold(data,label, n_fold)
    return t_set[i_fold]

#%%

print()
print("Loading data...")
print()

# -------- Read data ---------#
tr_x, tr_y, tst_x, tst_y = dataloader(5,0)
tr_x, val_x, tr_y, val_y = train_test_split(tr_x, tr_y, test_size=0.33, random_state=42)
print(tr_x[0])
print(tr_x[0].shape)
raise()
print()
print("train data: {}".format(tr_x.shape))
print("train label: {}".format(tr_y.shape))
print()
print("validation data: {}".format(val_x.shape))
print("validation label: {}".format(val_y.shape))
print()


#%%

x = tf.placeholder(tf.float32, (None, img_size[0], img_size[1], 3))
y_gt = tf.placeholder(tf.float32, (None, 2))
keep_prob = tf.placeholder(tf.float32)


layer_1_w = tf.Variable(tf.truncated_normal(shape=(5,5,3,32), mean=0, stddev=0.1))
layer_1_b = tf.Variable(tf.zeros(32))
layer_1 = tf.nn.conv2d(x, layer_1_w, strides=[1,1,1,1], padding='SAME') + layer_1_b
layer_1 = tf.nn.relu(layer_1)
layer_1 = tf.nn.max_pool(layer_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

layer_2_w = tf.Variable(tf.truncated_normal(shape=(5,5,32,64), mean=0, stddev=0.1))
layer_2_b = tf.Variable(tf.zeros(64))
layer_2 = tf.nn.conv2d(layer_1, layer_2_w, strides=[1,1,1,1], padding='SAME') + layer_2_b
layer_2 = tf.nn.relu(layer_2)
layer_2 = tf.nn.max_pool(layer_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

flat_arr = tf.reshape(layer_2, [-1, int(img_size[0]/4)*int(img_size[1]/4)*64])

fcl_1_w = tf.Variable(tf.truncated_normal(shape=(int(img_size[0]/4)* \
    int(img_size[1]/4)*64,1024), mean=0, stddev=0.1))
fcl_1_b = tf.Variable(tf.zeros(1024))
fcl_1 = tf.matmul(flat_arr, fcl_1_w) + fcl_1_b
fcl_1 = tf.nn.dropout(fcl_1, keep_prob)

fcl_2_w = tf.Variable(tf.truncated_normal(shape=(1024,2), mean=0, stddev=0.1))
fcl_2_b = tf.Variable(tf.zeros(2))
fcl_2 = tf.matmul(fcl_1, fcl_2_w) + fcl_2_b

y = fcl_2


#%%

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_gt, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

saver = tf.train.Saver(max_to_keep = 0)
init = tf.global_variables_initializer()


#%%

batch = 50
dropout_prob = 0.5


with tf.Session() as sess:

    try:
        saver.restore(sess, "./train/dogs_cats_cnn")
        print()
        print('Initialization loaded')
        print()
    except:
        sess.run(init)
        print()
        print('New initialization done')
        print()

    for epoch in range(201):

        tr_x, tr_y = shuffle(tr_x, tr_y)

        accum_loss = 0
        accum_acc = 0

        for m in range(0, tr_x.shape[0], batch):
            m2 = min(tr_x.shape[0], m+batch)
            
            _, loss_scr, acc_scr = sess.run((train_step, loss, accuracy), \
                feed_dict = {x: tr_x[m:m2], y_gt: tr_y[m:m2], \
                keep_prob: dropout_prob})

            accum_loss += loss_scr*(m2-m)
            accum_acc += acc_scr*(m2-m)

        if epoch%10 == 0:
            print("Epoch: {}".format(epoch))
            print("Train loss = {}".format(accum_loss/tr_x.shape[0]))
            print("Train accuracy = {:03.4f}".format(accum_acc/tr_x.shape[0]))

            accum_acc = 0

            for m in range(0, val_x.shape[0], batch):
                m2 = min(val_x.shape[0], m+batch)
                
                acc_scr = sess.run((accuracy), \
                    feed_dict = {x: val_x[m:m2], y_gt: val_y[m:m2], \
                    keep_prob: 1})

                accum_acc += acc_scr*(m2-m)
            print("Validation accuracy = {:03.4f}".format(accum_acc/val_x.shape[0]))
            print()

        save_path = saver.save(sess, "./train/dogs_cats_cnn")


#%%

print("This is the end of the training")
print("Entering in testing mode")
print()

img_size = (28,28)

test_data, test_label = load_data(img_folder, test_list, img_size)

print("test data: {}".format(test_data.shape))
print("test label: {}".format(test_label.shape))
print()


#%%

with tf.Session() as sess:

    saver.restore(sess, "./train/dogs_cats_cnn")
    print()
    print('Initialization loaded')
    print()

    accum_acc = 0

    for m in range(0, test_data.shape[0], batch):
        m2 = min(test_data.shape[0], m+batch)
        
        acc_scr = sess.run((accuracy), \
            feed_dict = {x: test_data[m:m2], y_gt: test_label[m:m2], \
            keep_prob: 1})

        accum_acc += acc_scr*(m2-m)
    print("Test accuracy = {:03.4f}".format(accum_acc/test_data.shape[0]))
    print()
