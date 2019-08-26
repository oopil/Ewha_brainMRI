import os
import sys
import nrrd
import numpy as np
import openpyxl
import csv
from skimage.transform import resize
import matplotlib.pyplot as plt
sys.path.append('..')
from excel_data_reader import *

# =========== file path setting ============== #
# dir_path =  "/home/soopil/Desktop/Dataset/EWHA_brain_tumor/0_Fwd_ MENINGIOMA 추가 자료 1_190711/MASKS" # SINCHON : internal data set
# dir_path =  "/home/soopil/Desktop/Dataset/EWHA_brain_tumor/Meningioma_External validation/EWHA/MASKS" # EWHA : external validation set
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha_early/Meningioma_only_T1C_masks" # SINCHON : original internal validation set
file_list = os.listdir(dir_path)
fd_result_file = "../fd_result/SINCHON_FD_result_20190731_orig.txt"
print(len(file_list))

# ===================== excel file read ===================== #
xl_path = '/home/soopil/Desktop/Dataset/brain_ewha_early/20180508_Entire_Train and Test_Meningioma.xlsx'
# xl_path = '/home/soopil/Desktop/Dataset/brain_ewha_early/20180508_Entire_Train and Test_Meningioma_add15.xlsx' # add 15 subjects
subj_list_excel, label_list = read_xl(xl_path, '20171108_New_N4ITK corrected')
print(len(subj_list_excel), len(set(subj_list_excel)),len(subj_list_excel)-len(set(subj_list_excel)), subj_list_excel)
# raise()
# ===================== read FD features ===================== #
result_dpath = '/home/soopil/Desktop/github/Ewha_brainMRI/fd_result/'
# result_fname = 'SINCHON_FD_result_20190718.txt' # external
# result_fname = 'SINCHON_FD_result_20190719_rescale.txt'
# result_fname = 'SINCHON_FD_result_20190723.txt' # original dataset
# result_fname = 'SINCHON_FD_result_20190731_orig.txt' # original dataset
# result_fname = 'SINCHON_FD_result_20190802_orig_missing_add.txt' # original dataset
# result_fname = 'SINCHON_FD_result_20190801_missing.txt'
# result_fname = 'SINCHON_FD_result_20190801_redrawn.txt'
result_fname = 'SINCHON_FD_result_20190813.txt' # additional dataset from 20190813

result_fpath = os.path.join(result_dpath, result_fname)
subj_list_fd, data_fd, mask_name = read_FD(result_fpath)
print(len(subj_list_fd), len(set(subj_list_fd)), subj_list_fd)
# exception for missing masks

# fd_csv = open('fd_analysis.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(fd_csv)
# fd_box = ['FD{}'.format(i) for i in range(10)]
# lac_box = ['LAC{}'.format(i) for i in range(9)]
# head = ['SubjectID','Grade','CE','T1C','T1C_norm','used mask'] + fd_box + lac_box
# wr.writerow(head)
#
# print(type(subj_list_excel[0]), type(mask_name[0]), type(subj_list_fd[0]))
#
# t_dict = {
#     0:'CE',
#     1:'T1C',
#     2:'T1C_norm'
# }
# cnt = 0
# for subj in sorted(subj_list_fd): #range(len(subj_list_excel)):
#     def bool2OX(is_:bool)->str:
#         if is_:
#             return 'O'
#         elif not is_:
#             return 'X'
#         else:
#             print(is_)
#             raise()
#
#     def type_check(name):
#         if 'T1C' in name:
#             if 'norm' in name:
#                 return 2
#             else:
#                 return 1
#
#         elif 'CE' in name:
#             return 0
#
#     blist = [False, False, False] # CE, T1C, T1C_norm
#     fd = [0 for _ in range(19)]
#     # subj = subj_list_excel[i]
#     # ixl = list(subj_list_excel).index(subj)
#     # grade = label_list[ixl]
#     grade = ''
#     used_mask = ''
#     print()
#     flist = []
#     for e in mask_name:
#         if str(subj) in e:
#             flist.append(e)
#             blist[type_check(e)] = True
#
#     ifd = mask_name.index(flist[0])
#     fd = data_fd[ifd]
#     used_mask = t_dict[type_check(mask_name[ifd])]
#     cnt += 1
#     print(ifd, mask_name[ifd])
#     print()
#
#     l = [int(subj), grade, bool2OX(blist[0]), bool2OX(blist[1]), bool2OX(blist[2]), used_mask] + list(fd)
#     print(l)
#     wr.writerow(l)
#
# fd_csv.close()
# raise()

# ===================== write csv file ===================== #

fd_csv = open('fd_analysis_final.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(fd_csv)
fd_box = ['FD{}'.format(i) for i in range(10)]
lac_box = ['LAC{}'.format(i) for i in range(9)]
head = ['SubjectID','Grade','CE','T1C','T1C_norm','used mask'] + fd_box + lac_box
wr.writerow(head)

print(type(subj_list_excel[0]), type(mask_name[0]), type(subj_list_fd[0]))

# for a,b in zip(subj_list_excel, label_list):
#     print(a,b)
# raise()
t_dict = {
    0:'CE',
    1:'T1C',
    2:'T1C_norm',
    -1:'no type'
}
cnt = 0
for subj_xl in sorted(subj_list_excel): #range(len(subj_list_excel)):
    def bool2OX(is_:bool)->str:
        if is_:
            return 'O'
        elif not is_:
            return 'X'
        else:
            print(is_)
            raise()

    def type_check(name):
        if 'T1C' in name:
            if 'norm' in name:
                return 2
            else:
                return 1

        elif 'CE' in name:
            return 0

        else:
            return -1

    # CE = False # 0
    # T1C = False # 1
    # T1C_norm = False # 2
    blist = [False, False, False] # CE, T1C, T1C_norm
    fd = [0 for _ in range(19)]
    # subj_xl = subj_list_excel[i]
    ixl = list(subj_list_excel).index(subj_xl)
    grade = label_list[ixl]

    used_mask = ''


    if str(subj_xl) in subj_list_fd:
        print()
        flist = []
        for e in mask_name:
            if str(subj_xl) in e:
                flist.append(e)
                blist[type_check(e)] = True

        ifd = mask_name.index(flist[0])
        fd = data_fd[ifd]
        used_mask = t_dict[type_check(mask_name[ifd])]
        cnt += 1
        print(ifd, mask_name[ifd])
        print()
    # print(flist)

    # fd = data_fd[i]

    l = [int(subj_xl), grade, bool2OX(blist[0]), bool2OX(blist[1]), bool2OX(blist[2]), used_mask] + list(fd)
    print(l)
    wr.writerow(l)

fd_csv.close()
print(mask_name)
print('subject with mask count : ',cnt)
print()

tmp = []
for subj in subj_list_fd:
    if int(subj) not in subj_list_excel:
        tmp.append(subj)
        print(subj)
print(len(tmp), len(set(tmp)))
assert False

for i, name in enumerate(sorted(file_list)):
    split = name.split('_')
    if split[0] == 'norm':
        subj_name = split[1]
    else:
        subj_name = split[0]
    print(name)


    # assert False

'''
s1 = os.path.join(dir_path,'8501307_T1C_norm-label.nrrd')
s2 = os.path.join(dir_path,'8501307_T1C_norm-label_1.nrrd')
'''

