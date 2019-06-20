import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from excel_data_reader import *

def report():
    result = np.load('test_prediction.npy', allow_pickle=True)
    test_y, arg_ytest = result
    test_y, arg_ytest = list(test_y), list(arg_ytest)
    report = classification_report(test_y, arg_ytest, target_names=['low','high'])
    print(type(report))
    print(report) #
    return report

if __name__ == "__main__":
    is_report = True
    is_test = True
    if is_test:
        class_num = 2
        train_x, train_y, test_x, test_y = EWHA_excel_datareader()
        sampling_option = "SIMPLE"
        train_x, train_y = over_sampling(train_x, train_y, sampling_option)
        # test_x, test_y = valence_class(test_x, test_y, class_num)
        train_y = one_hot_pd(train_y)
        test_t = test_y

        from keras.models import load_model
        model = load_model('neuralnet_model.h5')
        pred = model.predict_classes(test_x)
        y_test = model.predict(test_x)
        print(y_test[:5])
        # print(model.predict(train_x)[:5])
        arg_ytest = np.argmax(y_test, axis=1)
        np.save('test_prediction.npy',[arg_ytest, test_y])
        # assert False
        # print(classification_report(test_y, arg_ytest)) # , target_names=['low','high']
    if is_report:
        report()