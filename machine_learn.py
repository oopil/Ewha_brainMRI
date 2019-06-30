import pandas as pd
import seaborn as sns
from excel_data_reader import *
from FD_python.FD_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def dataloader():
    class_num = 2
    train_x, train_y, test_x, test_y = EWHA_excel_datareader()
    sampling_option = "SIMPLE"
    train_x, train_y = over_sampling(train_x, train_y, sampling_option)
    # test_x, test_y = valence_class(test_x, test_y, class_num)
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

    # print(test_y, Pred)
    print(confusion_matrix(test_y, Pred))
    print(classification_report(test_y, Pred, target_names=['low','high'])) # str type
    print(accuracy_score(test_y, Pred))

    return [train_accur, test_accur]

def voting_classifier(train_x, train_y, test_x, test_y):
    rf = RandomForestClassifier(n_estimators=2000, random_state=0)
    svm = SVC(gamma='auto', kernel='linear', random_state=0)
    logreg = LogisticRegression(solver='lbfgs', random_state=0)
    voting_clf = VotingClassifier(estimators=[('lr', logreg), ('rf', rf), ('svc', svm)],
                                  voting='soft')
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

def FD_logistic_classify():
    '''
    train with the box counting fractal dimension ...
    :return:
    '''
    xl_test = '/home/soopil/Desktop/Dataset/brain_ewha/Test_Meningioma_20180508.xlsx'
    xl_train = '/home/soopil/Desktop/Dataset/brain_ewha/Train_Meningioma_20180508.xlsx'
    tr_high_fdim, tr_low_fdim, tr_label_high, tr_label_low = FD_dataloader(xl_train, '20171108_New_N4ITK corrected')
    tst_high_fdim, tst_low_fdim, tst_label_high, tst_label_low = FD_dataloader(xl_test, 'Sheet1')
    train_x = tr_high_fdim + tr_low_fdim
    train_y = tr_label_high + tr_label_low
    test_x = tst_high_fdim + tst_low_fdim
    test_y = tst_label_high + tst_label_low
    train_x = np.array(train_x)

    class_num = 2
    sampling_option = "SIMPLE"
    train_x, train_y = over_sampling(train_x, train_y, sampling_option)
    # test_x, test_y = valence_class(test_x, test_y, class_num)
    model = logistic(train_x, train_y)
    result = check_result(model, train_x, train_y, test_x, test_y)
    print(result)

def Lac_logistic_classify():
    '''
    train with the lacunarity fractal dimension ...
    :return:
    '''
    xl_test = '/home/soopil/Desktop/Dataset/brain_ewha/Test_Meningioma_20180508.xlsx'
    xl_train = '/home/soopil/Desktop/Dataset/brain_ewha/Train_Meningioma_20180508.xlsx'
    tr_high_fdim, tr_low_fdim, tr_label_high, tr_label_low = Lac_dataloader(xl_train, '20171108_New_N4ITK corrected')
    tst_high_fdim, tst_low_fdim, tst_label_high, tst_label_low = Lac_dataloader(xl_test, 'Sheet1')
    train_x = tr_high_fdim + tr_low_fdim
    train_y = tr_label_high + tr_label_low
    test_x = tst_high_fdim + tst_low_fdim
    test_y = tst_label_high + tst_label_low
    train_x, test_x = np.array(train_x),np.array(test_x)

    class_num = 2
    sampling_option = "SIMPLE"
    train_x, train_y = over_sampling(train_x, train_y, sampling_option)
    # test_x, test_y = valence_class(test_x, test_y, class_num)

    # train_x, test_x = train_x[:,:3], test_x[:,:3]
    # train_x, test_x = np.log(train_x), np.log(test_x)
    train_x, test_x = np.ones_like(train_x) / train_x, np.ones_like(test_x) / test_x

    # print(train_x)
    model = logistic(train_x, train_y)
    result = check_result(model, train_x, train_y, test_x, test_y)
    print(result)
    assert False

    data, label = Lac_dataloader()
    data, label = shuffle_static(data, label)
    data = np.log(data)
    data = np.ones_like(data) / data
    # data = normalize_col(data, axis=0)
    # print(data)
    # assert False

    fold_num = 4
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

def excel_classify():
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
    # model = svm(train_x, train_y)
    model = logistic(train_x, train_y)
    check_result(model, train_x, train_y, test_x, test_y)

def print_result_file(result_file_name):
    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

if __name__ == '__main__':
    # FD_logistic_classify()
    Lac_logistic_classify()
    # excel_classify()