"""
@ author: S.J.Huang
"""

from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from excel_data_reader import *
from data import *
import os
import pandas as pd
import numpy as np
import csv
import argparse
import matplotlib.pyplot as plt

def CNN_model():
    """
    in the keras, dropout prob is keep_prob ??
    """
    prob = 0.1
    model = Sequential()
    # model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
    #                 activation ='relu', input_shape = (28,28,1)))
    # model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
    #                 activation ='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(prob))

    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(2048, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(2048, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(2048, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(2048, activation = "relu"))
    model.add(Dropout(prob))
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(prob))

    # model.add(Dense(2048, activation = "relu"))
    # model.add(Dropout(prob))
    #
    # model.add(Dense(2048, activation = "relu"))
    # model.add(Dropout(prob))

    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(prob))

    # model.add(BatchNormalization())
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(prob))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation = "softmax"))

    return model

def parse_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    return parser.parse_args()

def dataloader(n_fold, i_fold):
    subj_list, data, label = SINCHON_NN_reader()
    t_set = split_data_by_fold(data,label, n_fold)
    return t_set[i_fold]

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # -------- Read data ---------#
    train_x, train_y, test_x, test_y = dataloader(5,0)
    print(train_y)
    assert False
    sampling_option = "SIMPLE"
    train_x, train_y = over_sampling(train_x, train_y, sampling_option)
    # test_x, test_y = valence_class(test_x, test_y, class_num)
    train_y = one_hot_pd(train_y)
    test_t = test_y
    test_y = one_hot_pd(test_y)
    print(train_x[:5,:5])
    print(type(train_y))
    # assert False

    # ------ Preprocess data -----#
    # x_train = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    # x_train_norm = x_train / 255.0
    # t_train_onehot = np_utils.to_categorical(train_t)
    #
    # x_test = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')
    # x_test_norm = x_test / 255.0

    # ------- Model training----- #
    LR = 1e-4 # 1e-3
    epoch = 70
    model = CNN_model()
    adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=LR * (1 / epoch), amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    # model.compile(loss='cross', optimizer='adam', metrics=['accuracy'])
    # test_one_hot = np_utils.to_categorical(test_t)
    # test_depend_encode = test_one_hot
    train_history = model.fit(x=train_x, y=train_y,
                            epochs=epoch, batch_size=len(train_y), verbose=1, validation_split=0.1)
    model.save('neuralnet_model.h5')
    assert False
    # model save part
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    #
    # model.save_weights("model.h5")

    # train_history = model.fit(x=x_train_norm, y=t_train_onehot, epochs=20, batch_size=600, verbose=1)
    # y_test = model.predict_classes(x_test_norm)
    validation_data = (test_x, test_y)


    y_test = model.predict(test_x)
    print(y_test[:5])
    # print(model.predict(train_x)[:5])
    arg_ytest = np.argmax(y_test, axis=1)
    print(classification_report(test_y, arg_ytest, target_names=['low','high']))
    assert False
    acc = np.sum(arg_ytest == test_t) / len(arg_ytest)  # count correct prediction
    print("accuracy is {}".format(acc))
    print(np.amax(train_history.history['val_acc']))


    plt.subplot(1, 2, 1)
    plt.plot(train_history.history['loss'], label = 'loss')
    plt.legend(loc = 'upper left')
    plt.xlabel('Epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_history.history['val_acc'], 'g', label = 'val acc')
    plt.plot(train_history.history['acc'], 'r', label = 'train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'upper left')

    plt.show()

    # with open('answer.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['id', 'label'])
    #     for i in range(len(y_test)):
    #         writer.writerow((i, y_test[i]))
    #     print("Answer is saved in csv.")