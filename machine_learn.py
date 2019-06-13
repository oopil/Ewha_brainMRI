import numpy as np
import pandas as pd
import seaborn as sns
from excel_data_reader import *
from FD_data import FD_dataloader, Lac_dataloader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def dataloader():
    class_num = 2
    train_x, train_y, test_x, test_y = EWHA_excel_datareader()
    sampling_option = "SIMPLE"
    train_x, train_y = over_sampling(train_x, train_y, sampling_option)
    test_x, test_y = valence_class(test_x, test_y, class_num)
    train_x, train_y = np.array(train_x), np.array(train_y.astype(np.int32))
    return train_x, train_y.astype(np.int32), test_x, test_y.astype(np.int32)

def visualize(train_x, train_y):
    data = pd.DataFrame(
        {
            '1': train_x[:, 0],
            '2': train_x[:, 1],
            '3': train_x[:, 2],
            '4': train_x[:, 3],
            'class': train_y
        }
    )
    sns.pairplot(data, hue='class')

def check_result(model, train_x, train_y, test_x, test_y) -> list:
    Pred = model.predict(train_x)
    print('label\t:', train_y)
    print('predict :', Pred)
    total_num = len(train_y)
    correct_answer = 0
    for i in range(total_num):
        if train_y[i] == Pred[i]:
            correct_answer += 1

    train_accur = correct_answer * 100 / total_num
    # print('the probability is {}'.format(train_accur))

    Pred = model.predict(test_x)
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

    print(confusion_matrix(test_y, Pred))
    print(classification_report(test_y, Pred))
    print(accuracy_score(test_y, Pred))

    return [train_accur, test_accur]

def voting_classifier(train_x, train_y, test_x, test_y):
    rf = RandomForestClassifier(n_estimators=2000, random_state=0)
    svm = SVC(gamma='auto', kernel='linear', random_state=0)
    logreg = LogisticRegression(solver='lbfgs', random_state=0)
    voting_clf = VotingClassifier(estimators=[('lr', logreg), ('rf', rf), ('svc', svm)],
                                  voting='hard')
    voting_clf.fit(train_x, train_y)
    for clf in (logreg, rf, svm, voting_clf):
        clf.fit(train_x, train_y)
        pred = clf.predict(test_x)
        print(clf.__class__.__name__, accuracy_score(test_y, pred))
    return voting_clf

def bagging(train_x, train_y):
    print('<< implement bagging classifier... >>')
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=0), n_estimators=1000,
        max_samples=50, bootstrap=True, n_jobs=-1, random_state=0)
    bag_clf.fit(train_x, train_y)
    return bag_clf

def random_forest(train_x, train_y):
    print('<< implement random forest classifier... >>')
    rf = RandomForestClassifier(n_estimators=1000, random_state=123456)
    rf.fit(train_x, train_y)
    return rf

def svm(train_x, train_y):
    print('<< implement support vector machine ... >>')
    svm = SVC(gamma='auto', kernel='linear')
    svm.fit(train_x, train_y)
    return svm

def logistic(train_x, train_y):
    print('<< implement logistic regression classifier ... >>')
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(train_x, train_y)
    return logreg

def Lac_logistic_classification():
    '''
    train with the lacunarity fractal dimension ...
    :return:
    '''
    data, label = Lac_dataloader()
    data, label = shuffle_static(data, label)
    data = np.log(data)
    data = np.ones_like(data) / data
    # data = normalize_col(data, axis=0)
    # print(data)
    # assert False

    fold_num = 5
    whole_set = split_data_by_fold(data, label, fold_num)
    print(np.shape(whole_set),np.shape(whole_set[0]))
    results = []
    for fold in whole_set:
        train_x, train_y, test_x, test_y = fold
        class_num = 2
        sampling_option = "SIMPLE"
        train_x, train_y = over_sampling(train_x, train_y, sampling_option)
        test_x, test_y = valence_class(test_x, test_y, class_num)
        model = logistic(train_x, train_y)
        result = check_result(model, train_x, train_y, test_x, test_y)
        results.append(result)
        # print(result)
        pass

    for i in range(fold_num):
        print('fold {} training accur : {} / testing accur : {}'.format(i,results[i][0], results[i][1]))

    print(np.mean(results, axis=0))
    assert False

    # this part use the train_test_split method from sklearn
    train_x, test_x, train_y, test_y = train_test_split(data,label, test_size=0.2, random_state=0)
    train_x = np.array(train_x)
    print(np.shape(train_x), np.shape(train_y), np.shape(test_x), np.shape(test_y))
    class_num = 2
    sampling_option = "SIMPLE"
    train_x, train_y = over_sampling(train_x, train_y, sampling_option)
    test_x, test_y = valence_class(test_x, test_y, class_num)
    model = logistic(train_x, train_y)
    check_result(model, train_x, train_y, test_x, test_y)
    pass

def excel_classification():
    '''
    train the machine learning technique like logistic regression, support vector machine, etc...
    with the excel data from EWHA
    :return:
    '''
    train_x, train_y, test_x, test_y = dataloader()
    # visualize(train_x, train_y)
    # assert False
    # model = voting_classifier(train_x, train_y, test_x, test_y)
    # model = bagging(train_x, train_y)
    # model = random_forest(train_x, train_y)
    model = svm(train_x, train_y)
    # model = logistic(train_x, train_y)
    check_result(model, train_x, train_y, test_x, test_y)

def print_result_file(result_file_name):
    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

if __name__ == '__main__':
    Lac_logistic_classification()
    # excel_classification()