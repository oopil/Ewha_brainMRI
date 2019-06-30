import os
import nrrd
from FD_python.Lacunarity import lacunarity

file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/"

file_list = os.listdir(dir_path)
print(len(file_list))

fd_result_file = "./fd_result/Lac_result_v2.txt"
fd = open(fd_result_file, 'a+t')
fd.write('lacunarity fractal dimension.\n')
for name in sorted(file_list):
    split = name.split('_')
    if split[0] == 'norm':
        subj_name = split[1]
    else:
        subj_name = split[0]
    # print(name, subj_name)
    print(subj_name)
    file_path = os.path.join(dir_path, name)
    data, header = nrrd.read(file_path)
    result = lacunarity(data)
    # print(','.join(str(e) for e in [12, 3, 4, 5]))
    print(subj_name, result)
    # line = subj_name +'/'+','.join(str(e) for e in result[0])
    line = subj_name +'/'+','.join(str(e) for e in result)
    print(line)
    fd.write(line+'\n')
    # assert False

fd.close()