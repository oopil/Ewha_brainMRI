"""
run this code by python3.5 not in anaconda environment
because i have to use matlab engine, the interpreter language must be connect to my desktop.

this code use matlab engine
ex)
>> python3.5 fractal_dimension.py
"""

import os
import matlab.engine as Matlab
file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/"
fd_result_file = "./fd_result_file.txt"
file_list = os.listdir(dir_path)
print(len(file_list))

gr = [file_list[:60],file_list[60:120],file_list[120:]]

eng = Matlab.start_matlab()
fd = open(fd_result_file, 'a+t')
fd.write('box counting fractal dimension.\n')
for name in file_list:
    split = name.split('_')
    if split[0] == 'norm':
        subj_name = split[1]
    else:
        subj_name = split[0]
    # print(name, subj_name)
    print(subj_name)
    file_path = os.path.join(dir_path, name)
    array = eng.nrrdread(file_path)
    result= eng.boxcount(array)
    # print(','.join(str(e) for e in [12, 3, 4, 5]))
    print(subj_name, result)
    line = subj_name +'/'+','.join(str(e) for e in result[0])
    # line = subj_name +'/'+','.join(str(e) for e in result)
    print(line)
    fd.write(line+'\n')

fd.close()