import os
import argparse
import datetime
import subprocess
from excel_data_reader import *
from FD_data import FD_dataloader
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def logistic_regression(one_dataset, sampling_option, class_num)->str and int:
    train_data, train_label, test_data, test_label = one_dataset
    test_data, test_label = valence_class(test_data, test_label, class_num)
    train_data, train_label = over_sampling(train_data, train_label, sampling_option)
    # train_data, train_label = valence_class(train_data, train_label, class_num)
    # print(train_data.shape, test_data.shape)

    if class_num == 2:
        logreg = LogisticRegression(solver='lbfgs')
    elif class_num > 2:
        logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial') # multinomial / auto/ ovr
    logreg.max_iter = 1000
    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(train_data, train_label)

    Pred = logreg.predict(train_data)
    print('label\t:', train_label)
    print('predict :', Pred)
    total_num = len(train_label)
    correct_answer = 0
    for i in range(total_num):
        if train_label[i] == Pred[i]:
            correct_answer += 1

    train_accur = correct_answer * 100 / total_num
    # print('the probability is {}'.format(train_accur))

    Pred = logreg.predict(test_data)
    print('label\t:',test_label)
    print('predict :',Pred)
    total_num = len(test_label)
    correct_answer = 0
    for i in range(total_num):
        if test_label[i] == Pred[i]:
            correct_answer += 1
    # print('the probability is {}'.format(test_accur))

    test_accur = correct_answer*100 / total_num
    print('the probability is {}'.format(test_accur))
    return 'train and test number : {} / {:<5}'.format(len(train_label),len(test_label))+\
           ',top Test  : {:<10}' .format(test_accur // 1)+\
           ',top Train : {:<10}\n'.format(train_accur // 1), test_accur

def main():
    data, label = FD_dataloader()
    data, label = shuffle_static(data, label)
    whole_set = split_data_by_fold(data,label,5)
    # print(whole_set)
    print()
    print(label)
    # assert False
    result_file_name = './FD_logistic_regression_result.txt'
    is_remove_result_file = True
    if is_remove_result_file:
        # command = 'rm {}'.format(result_file_name)
        # print(command)
        subprocess.call(['rm',result_file_name])
        # os.system(command)
    # assert False

    sampling_option = "RANDOM"
    line_length = 100
    total_test_accur = []
    test_accur_list = []
    results = []
    for fold_index, one_fold_set in enumerate(whole_set):
        train_num, test_num = len(one_fold_set[0]), len(one_fold_set[2])
        contents = []
        line, test_accur = logistic_regression(one_fold_set, sampling_option, 2)
        contents.append(line)
        test_accur_list.append(test_accur)
        results.append(contents)

    test_accur_avg = int(sum(test_accur_list) / len(test_accur_list))
    results.append('{} : {}\n'.format('avg test accur', test_accur_avg))
    results.append('=' * line_length + '\n')
    total_test_accur.append(test_accur_avg)

    for e in results:
        print(e)
    assert False
    file = open(result_file_name, 'a+t')
    # print('<< results >>')
    for result in results:
        file.writelines(result)
        # print(result)
    # print(contents)
    file.close()



def print_result_file(result_file_name):
    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

if __name__ == '__main__':
    main()