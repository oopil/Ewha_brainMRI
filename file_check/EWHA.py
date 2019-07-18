# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os

file_list = os.listdir('.')
print(file_list)
#nifti_list = os.listdir('EWHA\\meningioma')
#nifti_list = os.listdir('EWHA\\n4itk_medial')
nifti_list = os.listdir('EWHA\\Preprocessing_finished_meningioma')
csv_list = os.listdir('EWHA\\CSV')
mask_list = os.listdir('EWHA\\MASKS')
#print(nifti_list)
#print(csv_list)
#print(mask_list)
print(len(csv_list), len(mask_list), len(nifti_list))

#assert False

csv_T1, csv_T2, new_mask, nifti_T1, nifti_T2, nifti_T2_REG = [],[],[],[],[],[]
for csv in sorted(csv_list):
    subj = csv.split('_')[0]
    if 'T1' in csv:
        assert subj not in csv_T1
        csv_T1.append(subj)
        
    if 'T2' in csv:
        assert subj not in csv_T2
        csv_T2.append(subj)

#print(len(csv_T1))
#print(len(csv_T2))
#assert False

for mask in sorted(mask_list):
    if mask == '10630810_SEP_T2-label.nrrd':
        # T1, T2 label both exist
        continue
    subj = mask.split('_')[0]
#    print(mask, subj)
    assert subj not in new_mask
    new_mask.append(subj)
#    print(subj)

for nifti in sorted(nifti_list):
    subj = nifti.split('_')[0]
    print(nifti,subj)
    if 'label' in nifti:
        print('label file : ' , nifti)
        continue
    
    if 'T1' in nifti:
        assert subj not in nifti_T1
        nifti_T1.append(subj)
        continue
    
    if 'REG' in nifti:
        assert subj not in nifti_T2_REG
        nifti_T2_REG.append(subj)
        continue
        
    if 'T2' in nifti and 'REG' not in nifti:
        assert subj not in nifti_T2
        nifti_T2.append(subj)
        continue
    
    
        
print(len(csv_T1))
print(len(csv_T2))
print(len(set(csv_T1) & set(csv_T2)))
print('T1 and T2 CSV same ? : ', set(csv_T1) == set(csv_T2))
print('T1 - T2 : ',set(csv_T1) - set(csv_T2))
print('T2 - T1 : ',set(csv_T2) - set(csv_T1))

print(new_mask, len(new_mask))

print(len(nifti_T1), len(nifti_T2), len(nifti_T2_REG))
print('T1 and T2 NIFTI same ? : ', set(nifti_T1) == set(nifti_T2))
print(len(set(nifti_T1) & set(nifti_T2)))
print('T1 - T2 : ',set(nifti_T1) - set(nifti_T2))
print('T2 - T1 : ',set(nifti_T2) - set(nifti_T1))

complete_set = set(csv_T1)&set(csv_T2)&set(new_mask)&set(nifti_T1)&set(nifti_T2)&set(nifti_T2_REG)
print('<< complete set >> \n ',len(complete_set), complete_set)

def sub_set(a,b):
    return set(a) - set(b)

print(sub_set(csv_T1, complete_set))
print(sub_set(csv_T2, complete_set))
print(sub_set(new_mask, complete_set))
print(sub_set(nifti_T1, complete_set))
print(sub_set(nifti_T2, complete_set))
print(sub_set(nifti_T2_REG, complete_set))
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

