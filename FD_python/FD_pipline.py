import os
import nrrd
import numpy as np
from FD_python.Lacunarity import box_count_FD, lacunarity

file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
dir_path =  "/home/soopil/Desktop/Dataset/EWHA_brain_tumor/0_Fwd_ MENINGIOMA 추가 자료 1_190711/MASKS"

file_list = os.listdir(dir_path)
print(len(file_list))

fd_result_file = "../fd_result/FD_result_20190718.txt"
fd = open(fd_result_file, 'a+t')
fd.write('box counting fractal dimension.\n')
for name in sorted(file_list):
    split = name.split('_')
    if split[0] == 'norm':
        subj_name = split[1]
    else:
        subj_name = split[0]
    # print(name, subj_name)
    # print()
    file_path = os.path.join(dir_path, name)
    data, header = nrrd.read(file_path)
    # print(subj_name, np.shape(data))
    # print(str(np.shape(data)))

    result_FD = box_count_FD(data)
    result_LAC = lacunarity(data)
    line = subj_name +'/'+str(np.shape(data))+'/'+','.join(str(e) for e in result_FD)+'/'+','.join(str(e) for e in result_LAC)

    print(line)
    fd.write(line+'\n')

    # assert False

fd.close()