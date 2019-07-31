import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from excel_data_reader import *
from FD_python.FD_data import *
from sklearn.metrics import classification_report
from machine_learn import check_result

# --------------------- original data --------------------- #
# 1. data load
# 1-1. box counting FD
xl_test = '/home/soopil/Desktop/Dataset/brain_ewha_early/Test_Meningioma_20180508.xlsx'
xl_train = '/home/soopil/Desktop/Dataset/brain_ewha_early/Train_Meningioma_20180508.xlsx'
# FD_dataloader(xl_train,'20171108_New_N4ITK corrected')
# FD_merge_dataloader()
# assert False

tr_high_fdim, tr_low_fdim, tr_label_high, tr_label_low, tr_high_subj, tr_low_subj= FD_dataloader(xl_train,'20171108_New_N4ITK corrected','./fd_result/fd_result_file.txt')
tst_high_fdim, tst_low_fdim, tst_label_high, tst_label_low, tst_high_subj, tst_low_subj= FD_dataloader(xl_test,'Sheet1','./fd_result/fd_result_file.txt')

train_x = tr_high_fdim + tr_low_fdim
train_y = tr_label_high + tr_label_low
test_x =  tst_high_fdim + tst_low_fdim
test_y =  tst_label_high + tst_label_low
data = train_x + test_x
label = train_y + test_y

# 1-2. Lacunarity
tr_high_lac, tr_low_lac, tr_label_high, tr_label_low, tr_high_subj_lac, tr_low_subj_lac = Lac_dataloader(xl_train,'20171108_New_N4ITK corrected','./fd_result/Lac_result_v2.txt')
tst_high_lac, tst_low_lac, tst_label_high, tst_label_low, tst_high_subj_lac, tst_low_subj_lac = Lac_dataloader(xl_test,'Sheet1','./fd_result/Lac_result_v2.txt')

train_x_lac = tr_high_lac + tr_low_lac
train_y_lac = tr_label_high + tr_label_low
test_x_lac =  tst_high_lac + tst_low_lac
test_y_lac =  tst_label_high + tst_label_low

data_lac = train_x_lac + test_x_lac
label_lac = train_y_lac + test_y_lac
print(np.shape(data_lac), np.shape(label_lac))

assert set(tr_high_subj) == set(tr_high_subj_lac)
assert tr_high_subj == tr_high_subj_lac

# 2. t test
high_fd = tr_high_fdim + tst_high_fdim
low_fd = tr_low_fdim + tst_low_fdim
print(np.shape(high_fd),np.shape(low_fd))
tTestResult = stats.ttest_ind(high_fd, low_fd)
print(tTestResult)
print('statistics : ',tTestResult[0])
print('p value : ',tTestResult[1])

high_lac = tr_high_lac + tst_high_lac
low_lac = tr_low_lac + tst_low_lac
print(np.shape(high_lac),np.shape(low_lac))
tTestResult = stats.ttest_ind(high_lac, low_lac)
print(tTestResult)
print('statistics : ',tTestResult[0])
print('p value : ',tTestResult[1])

data = np.array(data)
data_lac = np.array(data_lac)

# normalization part
def normalize(data, index):
    data = np.array(data)
    data = (data - np.amin(data)) / np.amax(data)
    return data

data = normalize_col(data)
data_lac = normalize_col(data_lac)
print(data[0])
print(data_lac[0])
# assert False

# this is wrong !! and it is correct now .. i matched the subject list
data = np.concatenate([data,data_lac], axis=1)
print(np.shape(data))

# 3. logistic regression
n_fold = 5
whole_set = split_data_by_fold(data[:,9:12], label, n_fold)
results = []
for i in range(n_fold):
    fold = i
    tr_x, tr_y, tst_x, tst_y = whole_set[fold]
    print(np.shape(tr_x))
    class_num = 2
    sampling_option = "SIMPLE" # ADASYN SMOTE SMOTEENN SMOTETomek RANDOM BolderlineSMOTE None
    tr_x, tr_y = over_sampling(tr_x, tr_y, sampling_option)
    # txt_x, tst_y = valence_class(txt_x, tst_y, class_num)
    model = LogisticRegression()
    model.fit(tr_x, tr_y)
    print()
    print('<< fold {} >>'.format(fold))
    result, report = check_result(model, tr_x, tr_y, tst_x, tst_y)
    results.append([result[0]//1, report['low']['recall']//0.01, report['high']['recall']//0.01,report['macro avg']['recall']//0.01,report['weighted avg']['recall']//0.01])
    # print(report['low']['recall'])
    # print(report['high']['recall'])
    # print(report['macro avg']['recall'])

results = np.array(results)
for i, line in enumerate(results):
    print(i, line)
print('avg : ',np.mean(results,axis=0).astype(np.int32))

if __name__ =='__main__':
    pass