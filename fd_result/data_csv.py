'''
from box counting and lacunarity result .txt file
to csv file converter

2019 07 31 WED modified

i need to
'''

import os
import csv
import pandas as pd
import numpy as np

result_dpath = '/home/soopil/Desktop/github/Ewha_brainMRI/fd_result/'
# result_fname = 'SINCHON_FD_result_20190718.txt' # external
# result_fname = 'SINCHON_FD_result_20190719_rescale.txt'
result_fname = 'SINCHON_FD_result_20190723.txt' # original dataset
result_fpath = os.path.join(result_dpath, result_fname)

fd = open(result_fpath)
lines = fd.readlines()
print(lines)

# ===================== read FD features ===================== #
def FD(box_count):
    box_count = np.array(box_count).astype(np.float32)
    length = len(box_count)
    r = np.array([2 ** i for i in range(0, length)]).astype(np.float32)
    n_grad = np.gradient(np.log(box_count))
    r_grad = np.gradient(np.log(r))
    return (np.divide(n_grad, r_grad) * (-1))

subj_list_fd, data_fd = [],[]

for line in sorted(lines):

    if line == 'box counting fractal dimension.\n':
        continue

    print(line)
    split = line.split('/')
    subj = split[0]
    bx_count = split[2].split(',')
    lac = split[3].split(',')
    lac = list(np.array(lac).astype(np.float32))
    fdim = list(FD(bx_count))
    # lac = list(FD( np.flip(lac)))
    lac = list(np.log(lac))
    # print(fdim+lac)
    # print(lac)

    assert len(fdim) == 10 and len(lac) == 9
    subj_list_fd.append(subj)
    data_fd.append(fdim+lac)
    # print(subj, bx_count, lac)
    # print(len(bx_count), len(lac))
    # print(list(FD(np.flip(lac)))) # ???? let me try
    # print(list(fdim))
fd.close()

print(len(set(subj_list_fd)), len(subj_list_fd))

fd_csv = open('fd_analysis.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(fd_csv)
fd_box = ['FD{}'.format(i) for i in range(10)]
lac_box = ['LAC{}'.format(i) for i in range(9)]
head = ['SubjectID'] + fd_box + lac_box
wr.writerow(head)

for i in range(len(subj_list_fd)):
    l = [int(subj_list_fd[i])] + list(data_fd[i])
    print(l)
    wr.writerow(l)
fd_csv.close()

'''
# codes to refer ; pandas data frame work
rpath = 'MICCAI_BraTS_2019_Data_Training'

data = pd.read_csv('MICCAI_BraTS_2019_Data_Training/survival_data.csv', index_col='BraTS19ID')
print(data.head())
# print(data['Age'])
length = len(data)
data.loc[:, 'Group'] = -1 #pd.Series([-1 for _ in range(length)], index=data.index)
print(data.head())

HGG_list = os.listdir(os.path.join(rpath, 'HGG'))
LGG_list = os.listdir(os.path.join(rpath, 'LGG'))
print(HGG_list)
print(LGG_list)

# there is no low grade patient in survival predictions.
for subj in data.index:
    if subj in HGG_list:
        assert subj not in LGG_list
        data.loc[subj,'Group'] = 1

    if subj in LGG_list:
        assert subj not in HGG_list
        data.loc[subj,'Group'] = 0
        assert False

# data = data.dropna(how='all')
new_data = data.dropna(how='any', axis=0, subset=['Age','Survival'])
print(new_data.head())
for i, e in enumerate(new_data.index):
    # print(e)
    sur = new_data.loc[e,'Survival']
    if ('ALIVE' in sur):
        print(e,sur) #
new_data = new_data.drop(index=['BraTS19_CBICA_BFB_1', 'BraTS19_CBICA_BFP_1'], axis=0)

'''

