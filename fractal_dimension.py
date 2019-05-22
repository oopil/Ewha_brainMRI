import os
import matlab.engine as Matlab

file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/"
fd_result_file = "./fd_result_file"
file_list = os.listdir(dir_path)
print(len(file_list))

eng = Matlab.start_matlab()
fd = open(fd_result_file, 'a+t')
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
    line = subj_name +'/'+','.join(str(e) for e in result)
    fd.write(line+'\n')

fd.close()