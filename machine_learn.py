import os
import argparse
import datetime
import subprocess
from excel_data_reader import *
from FD_data import FD_dataloader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing

def dataloader():
    class_num = 2
    train_x, train_y, test_x, test_y = EWHA_excel_datareader()
    sampling_option = "SIMPLE"
    train_x, train_y = over_sampling(train_x, train_y, sampling_option)
    test_x, test_y = valence_class(test_x, test_y, class_num)
    return train_x, train_y, test_x, test_y


def svm():
    print('implement support vector machine ... ')
    train_x, train_y, test_x, test_y = dataloader()
    clf = SVC(gamma='auto')
    clf.fit(train_x, train_y)
    Pred = clf.predict(train_x)
    print('label\t:', train_y)
    print('predict :', Pred)
    total_num = len(train_y)
    correct_answer = 0
    for i in range(total_num):
        if train_y[i] == Pred[i]:
            correct_answer += 1

    train_accur = correct_answer * 100 / total_num
    # print('the probability is {}'.format(train_accur))

    Pred = clf.predict(test_x)
    print('label\t:', test_y)
    print('predict :', Pred)
    total_num = len(test_y)
    correct_answer = 0
    for i in range(total_num):
        if test_y[i] == Pred[i]:
            correct_answer += 1
    # print('the probability is {}'.format(test_accur))

    test_accur = correct_answer * 100 / total_num
    print('the train accuracy is {}'.format(train_accur))
    print('the test accuracy is {}'.format(test_accur))

def logistic():
    print('implement logistic regression classifier ... ')

    train_x, train_y, test_x, test_y = dataloader()
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(train_x, train_y)

    Pred = logreg.predict(train_x)
    print('label\t:', train_y)
    print('predict :', Pred)
    total_num = len(train_y)
    correct_answer = 0
    for i in range(total_num):
        if train_y[i] == Pred[i]:
            correct_answer += 1

    train_accur = correct_answer * 100 / total_num
    # print('the probability is {}'.format(train_accur))

    Pred = logreg.predict(test_x)
    print('label\t:', test_y)
    print('predict :', Pred)
    total_num = len(test_y)
    correct_answer = 0
    for i in range(total_num):
        if test_y[i] == Pred[i]:
            correct_answer += 1
    # print('the probability is {}'.format(test_accur))

    test_accur = correct_answer * 100 / total_num
    print('the train accuracy is {}'.format(train_accur))
    print('the test accuracy is {}'.format(test_accur))

def main():
    svm()
    # logistic()

def print_result_file(result_file_name):
    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

if __name__ == '__main__':
    main()