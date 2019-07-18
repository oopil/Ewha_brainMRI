# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
base_dir = '/home/soopil/Desktop/Dataset/EWHA_brain_tumor'

file_list = os.listdir(base_dir)
print(file_list)
nifti_dir= 'nifti'
nifti_list = os.listdir(os.path.join(base_dir, nifti_dir))
#print(nifti_list)
csv_list = os.listdir(os.path.join(base_dir,'0_Fwd_ MENINGIOMA 추가 자료 1_190711/Additional_Pt_csv/Additional_Pt_csv'))
mask_list = os.listdir(os.path.join(base_dir,'0_Fwd_ MENINGIOMA 추가 자료 1_190711/Additional_Pt_masks/Additional_Pt_masks'))
print(csv_list)
print(mask_list)
print(len(csv_list), len(mask_list), len(nifti_list))

# 6818236, 7987012 absent files already

csv_T1, csv_T2, new_mask, nifti_T1, nifti_T2 = [],[],[],[],[]
for csv in sorted(csv_list):
    subj = csv.split('_')[0]
    if 'T1' in csv:
        assert subj not in csv_T1
        csv_T1.append(subj)
        
    if 'T2' in csv:
        assert subj not in csv_T2
        csv_T2.append(subj)
        
for mask in sorted(mask_list):
    subj = mask.split('_')[0]
    assert subj not in new_mask
    new_mask.append(subj)
#    print(subj)
    
for nifti in sorted(nifti_list):
    subj = nifti.split('_')[0]
    if 'T1' in nifti:
        assert subj not in nifti_T1
        nifti_T1.append(subj)
        
    if 'T2' in nifti:
        assert subj not in nifti_T2
        nifti_T2.append(subj)
        
print(len(csv_T1))
print(len(csv_T2))
print(len(set(csv_T1) & set(csv_T2)))
print(set(csv_T1) == set(csv_T2))
print('T1 - T2 : ',set(csv_T1) - set(csv_T2))
print('T2 - T1 : ',set(csv_T2) - set(csv_T1))

print(new_mask, len(new_mask))

print(len(nifti_T1), len(nifti_T2))
print(set(nifti_T1) == set(nifti_T2))
print(len(set(nifti_T1) & set(nifti_T2)))
print('T1 - T2 : ',set(nifti_T1) - set(nifti_T2))
print('T2 - T1 : ',set(nifti_T2) - set(nifti_T1))

complete_set = set(csv_T1)&set(csv_T2)&set(new_mask)&set(nifti_T1)&set(nifti_T2)
print('<< complete set >> \n ', len(complete_set), complete_set)

def sub_set(a,b):
    return set(a) - set(b)

print(sub_set(csv_T1, complete_set))
print(sub_set(csv_T2, complete_set))
print(sub_set(new_mask, complete_set))
print(sub_set(nifti_T1, complete_set))
print(sub_set(nifti_T2, complete_set))
#print(len(set(csv_T1)&set(csv_T2)&set(new_mask)&set(nifti_T1)&set(nifti_T2)))
assert False

T1_absent, T2_absent = [], []
for mask in sorted(mask_list):
#    print(csv)
    T1, T2 = False, False
    subj = mask.split('_')[0]
    
    for csv in sorted(csv_list):
        if subj in csv:
#            print(subj, csv)
            if 'T1' in csv:
                assert not T1
                T1 = True
            elif 'T2' in csv:
                assert not T2
                T2 = True
    
    if not T1:
        T1_absent.append(subj)
    if not T2:
        T2_absent.append(subj)
#    assert False

print('<< csv file check >>')
print('T1 absent subject : ', T1_absent)
print('T2 absent subject : ', T2_absent)

T1_absent, T2_absent = [], []

for mask in sorted(mask_list):
#    print(csv)
    T1, T2 = False, False
    subj = mask.split('_')[0]
    
    for nifti in sorted(nifti_list):
        if subj in nifti:
#            print(subj, nifti)
            if 'T1' in nifti:
                assert not T1
                T1 = True
            elif 'T2' in nifti:
                assert not T2
                T2 = True
    
    if not T1:
        T1_absent.append(subj)
    if not T2:
        T2_absent.append(subj)

print('<< nifti file check >>')
print('T1 absent subject : ', T1_absent)
print('T2 absent subject : ', T2_absent)

