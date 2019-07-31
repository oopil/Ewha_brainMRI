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

# # --------------------- original data --------------------- #
# # 1. data load
# # 1-1. FD
# # 1-2. Lacunarity
subj_list, data, label = SINCHON_FD_reader()
# subj_list, data, label = EWHA_CSV_reader()
data = np.array(data)
print(np.shape(data))
# # 2. t test
low, high = [],[]
for i in range(len(label)):
    if label[i] == 0:
        low.append(data[i])
    elif label[i] == 1:
        high.append(data[i])

tTestResult = stats.ttest_ind(low, high)
pv = tTestResult[1]
print(high[0])
print('pvalue : \n')
for i, v in enumerate(pv):
    if i == 10:
        print()
    print(v)
# print(tTestResult[1])
# print(high)
# # 3. logistic regression
# --------------------- load data --------------------- #
n_fold = 5
whole_set = split_data_by_fold(data[:,10:10+3], label, n_fold)
results = []
for fold in range(n_fold):
    tr_x, tr_y, tst_x, tst_y = whole_set[fold]
    print(np.shape(tr_x))
    sampling_option = "SIMPLE"
    # ADASYN SMOTE SMOTEENN SMOTETomek RANDOM BolderlineSMOTE None
    tr_x, tr_y = over_sampling(tr_x, tr_y, sampling_option)
    # txt_x, tst_y = valence_class(txt_x, tst_y, class_num)
    model = LogisticRegression()
    model.fit(tr_x, tr_y)
    print()
    print('<< fold {} >>'.format(fold))
    result, report = check_result(model, tr_x, tr_y, tst_x, tst_y)
    results.append([result[0]/1, report['low']['recall']/0.01, report['high']['recall']/0.01,report['macro avg']['recall']/0.01,report['weighted avg']['recall']/0.01])

results = np.array(results)
for i, line in enumerate(results):
    print(i, line)
print('avg : ',np.mean(results,axis=0))

if __name__ =='__main__':
    pass