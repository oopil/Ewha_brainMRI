import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
# from ConvNeuralNet.CNN_net import *

sys.path.append('..')
# sys.path.append('/home/soopil/Desktop/Dataset/github/brainMRI_classification/ConvNeuralNet')
# server setting
from excel_data_reader import *
from NN_net import *

# %%
def parse_args() -> argparse:
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',                default='0', type=str)
    parser.add_argument('--setting',            default='desktop', type=str) # desktop sv186 sv202 sv144
    parser.add_argument('--mask',               default=False, type=str2bool)
    parser.add_argument('--buffer_scale',       default=30, type=int)
    parser.add_argument('--epoch',              default=3, type=int)
    parser.add_argument('--network',            default='simple', type=str) # simple attention siam
    parser.add_argument('--lr',                 default=0.000000000000000000000000000000001, type=float)
    parser.add_argument('--ch',                 default=32, type=int)
    parser.add_argument('--fold_try',           default=1, type=int)
    parser.add_argument('--batch_size',         default=10, type=int)
    return parser.parse_args()

# %%
args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sv_set_dict = {
    "desktop":0,
    "sv186":186,
    "sv144":144,
    "sv202":202,
}
sv_set = sv_set_dict[args.setting]

ch = args.ch
batch = args.batch_size # 10
dropout_prob = 0.5
epochs = args.epoch
is_mask = args.mask
print_freq = 1
learning_rate = args.lr

class_num = 2
train_x, train_y, test_x, test_y = EWHA_excel_datareader()
sampling_option = "SIMPLE"
train_x, train_y = over_sampling(train_x, train_y, sampling_option)
test_x, test_y = valence_class(test_x, test_y, class_num)
train_y = one_hot_pd(train_y)
test_y = one_hot_pd(test_y)
shape = train_x.shape
print(shape)
print(train_y, test_y)
# for line in test_x:
#     print(line)
# assert False
'''
model building parts
'''

class_num = 2
images = tf.placeholder(tf.float32, (None, shape[1]), name='inputs')
y_gt = tf.placeholder(tf.float32, (None, class_num))
keep_prob = tf.placeholder(tf.float32)
my_model = SimpleNet(tf.truncated_normal_initializer(mean=0, stddev=0.1), tf.nn.relu, class_num)
y = my_model.model(images)
# %%
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y)
cross_entropy = tf.squared_difference(y_gt, y)
loss = tf.reduce_mean(cross_entropy)

with tf.name_scope('learning_rate_decay'):
    start_lr = learning_rate
    global_step = tf.Variable(0, trainable=False)
    total_learning = epochs
    # lr = tf.train.exponential_decay(start_lr, global_step,total_learning,0.99999, staircase=True)
    lr = tf.train.exponential_decay(start_lr, global_step, decay_steps=epochs // 100, decay_rate=.96, staircase=True)

with tf.variable_scope('optimizer'):
    # optimizer = tf.train.AdamOptimizer(lr)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_gt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()
model_vars = tf.trainable_variables()
tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

train_accur = []
valid_accur = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('../log/train', sess.graph)
    test_writer = tf.summary.FileWriter('../log/test')

    for epoch in range(epochs):
        train_feed_dict = {
            images: train_x,
            y_gt: train_y
        }
        test_feed_dict = {
            images: test_x,
            y_gt: test_y
        }
        accum_loss = 0
        accum_acc = 0

        _, loss_scr, acc_scr, logit, train_summary = \
            sess.run((train_step, loss, accuracy, y, merged_summary), feed_dict=train_feed_dict)

        train_writer.add_summary(train_summary)
        if epoch % print_freq == 0:
            print("Epoch: {}/{}".format(epoch, epochs))
            print("Train loss = {}".format(loss_scr))
            print("Train accuracy = {:03.4f}".format(acc_scr // 0.01))

            val_acc, val_logit, test_summary = \
                sess.run((accuracy, y, merged_summary), feed_dict=test_feed_dict)

            print("Validation accuracy = {:03.4f}".format(val_acc // 0.01))
            print(logit[:5])
            print(train_y[:5])
            # print(val_logit[:5])
            train_writer.add_summary(test_summary)
            train_accur.append(acc_scr)
            valid_accur.append(val_acc)

assert False

# Summarize
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

model_vars = tf.trainable_variables()
tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

whole_set = read_cnn_data(sv_set)
top_train_accur_list = []
top_valid_accur_list = []
saturation_train_accur_list = []
saturation_valid_accur_list = []
train_result = []
valid_result = []
train_accur = []
valid_accur = []
count = 0
for fold in whole_set:
    acc_scr, val_acc = 0,0
    train_accur = []
    valid_accur = []
    class_num = 2
    # sampling_option = "None"
    # sampling_option = "RANDOM"
    sampling_option = "SIMPLE"
    train_x, train_y, test_x, test_y = fold
    test_x, test_y = valence_class(test_x, test_y, class_num)
    if sampling_option != "None":
        train_x, train_y = over_sampling(train_x, train_y, sampling_option)
        train_y = one_hot_pd(train_y)

    print()
    print("Loading data...")
    print()
    print()
    print("train data: {}".format(train_x.shape))
    print("train label: {}".format(train_y.shape))
    print()
    print("validation data: {}".format(test_x.shape))
    print("validation label: {}".format(test_y.shape))
    print()
    # saver = tf.train.Saver(max_to_keep=0)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print()
        print('Cross Validation Step ... ')
        print()
        # tensorflow dataset setting
        next_element, iterator = get_patch_dataset(train_x, train_y, args.buffer_scale, is_mask, batch)
        sess.run(iterator.initializer)

        # test_element, test_iterator = get_patch_dataset(test_x, test_y, args.buffer_scale, is_mask, len(test_y))
        # sess.run(test_iterator.initializer)
        # val_data_ts, test_label_ts = sess.run(test_element)
        val_data_ts, test_label_ts = read_test_data(test_x, test_y, is_masking=is_mask)
        test_label_ts = one_hot_pd(test_y)
        # print(test_label_ts)
        print(test_label_ts.shape)

        train_writer = tf.summary.FileWriter('../log/train', sess.graph)
        test_writer = tf.summary.FileWriter('../log/test')

        for epoch in range(epochs):
            train_x, train_y = sess.run(next_element)
            # print(train_x.shape, train_y.shape)
            # train_x, train_y = over_sampling(train_x, train_y, "RANDOM")
            # print(train_x.shape, train_y.shape)
            # assert False

            train_feed_dict = {
                images: train_x,
                y_gt: train_y
            }
            test_feed_dict = {
                images: val_data_ts,
                y_gt: test_label_ts
            }
            accum_loss = 0
            accum_acc = 0

            _, loss_scr, acc_scr, logit, train_summary = \
                sess.run((train_step, loss, accuracy, y, merged_summary), feed_dict=train_feed_dict)

            train_writer.add_summary(train_summary)
            if epoch % print_freq == 0:
                print("Epoch: {}/{}".format(epoch, epochs))
                print("Train loss = {}".format(loss_scr))
                print("Train accuracy = {:03.4f}".format(acc_scr // 0.01))

                val_acc, val_logit, test_summary = \
                    sess.run((accuracy, y, merged_summary), feed_dict=test_feed_dict)

                print("Validation accuracy = {:03.4f}".format(val_acc // 0.01))
                print(logit[:5]//0.01)
                # print(val_logit[:5])
                train_writer.add_summary(test_summary)
                train_accur.append(acc_scr)
                valid_accur.append(val_acc)
                # save trained model
                # save_path = saver.save(sess, "../train/cnn_lh")

    saturation_count = 5
    train_result.append(train_accur)
    valid_result.append(valid_accur)
    top_train_accur = np.max(train_accur, 0)
    top_valid_accur = np.max(valid_accur, 0)
    top_train_accur_list.append(top_train_accur)
    top_valid_accur_list.append(top_valid_accur)
    saturation_train_accur_list.append(np.mean(train_accur[-saturation_count:]))
    saturation_valid_accur_list.append(np.mean(valid_accur[-saturation_count:]))

    count += 1
    if count >= args.fold_try:
        break
file_contents = []

for i in range(len(train_result)):
    file_contents.append("<< fold {} result>>".format(i))
    file_contents.append("CNN lh and rh model")
    file_contents.append("masking : {}".format(args.mask))
    file_contents.append("train : {}".format(train_result[i]))
    file_contents.append("valid : {}".format(valid_result[i]))
file_contents.append("top train : {}".format(top_train_accur_list))
file_contents.append("top valid : {}".format(top_valid_accur_list))
file_contents.append("avg train top : {} , avg vaidation top : {}".format(np.mean(top_train_accur_list), np.mean(top_valid_accur_list)))
file_contents.append("saturation train : {}".format(saturation_train_accur_list))
file_contents.append("saturation valid : {}".format(saturation_valid_accur_list))
file_contents.append("avg saturation train : {} , avg saturation vaidation : {}".format(np.mean(saturation_train_accur_list), np.mean(saturation_valid_accur_list)))

for result in file_contents:
    print(result)

result_file_name = '../nn_result_'+args.network+'/cv.txt'
file = open(result_file_name, 'a+t')
for result in file_contents:
    result += '\n'
    file.writelines(result)

"""

with tf.variable_scope("Model"):
    with tf.variable_scope("Left"):
        lh = batch_norm(lh)
        lh = tf.layers.conv3d(inputs=lh, filters=32, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
        lh = tf.layers.conv3d(inputs=lh, filters=64, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
        lh = tf.layers.conv3d(inputs=lh, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
        lh = tf.layers.conv3d(inputs=lh, filters=256, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        lh = tf.layers.flatten(lh)

    with tf.variable_scope("Right", reuse=False):
        rh = batch_norm(rh)
        rh = tf.layers.conv3d(inputs=rh, filters=32, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
        rh = tf.layers.conv3d(inputs=rh, filters=64, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
        rh = tf.layers.conv3d(inputs=rh, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
        rh = tf.layers.conv3d(inputs=rh, filters=256, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        rh = tf.layers.flatten(rh)

    with tf.variable_scope("FCN"):
        x = tf.concat([lh, rh], -1)
        x = tf.layers.dense(x, units=2048, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=512, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=class_num, activation=tf.nn.sigmoid)
        y = x
"""