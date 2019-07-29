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
subj_list, data, label = SINCHON_FD_reader()
# subj_list, data, label = EWHA_CSV_reader()
data = np.array(data)
print(np.shape(data))
# # 1-2. Lacunarity
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
whole_set = split_data_by_fold(data[:,:10+3], label, n_fold)
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
    results.append([result[0]//1, report['low']['recall']//0.01, report['high']['recall']//0.01,report['macro avg']['recall']//0.01,report['weighted avg']['recall']//0.01])

results = np.array(results)
for i, line in enumerate(results):
    print(i, line)
print('avg : ',np.mean(results,axis=0).astype(np.int32))

assert False
# X = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3})

#Scale your data
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = pd.DataFrame(scaler.transform(X),columns = X.columns)

clf = LogisticRegression(random_state = 0)
clf.fit(tr_x, tr_y)

feature_importance = abs(clf.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(str(sorted_idx), fontsize=8)
# featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')
assert False

#%%
lr = LogisticRegression(C=1e5)
lr.fit(X, Y)

print(lr.coef_) # returns a matrix of weights (coefficients)

np.hstack((clf.intercept_[:,None], clf.coef_))

if __name__ =='__main__':
    pass