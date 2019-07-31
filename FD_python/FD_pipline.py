import os
import nrrd
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from FD_python.Lacunarity import box_count_FD, lacunarity

def rescale_threshold_3D(array, zoom_size):
    # ----------- array rescaling part ----------- #
    array = np.swapaxes(array, 0, 2)
    rescaled = resize(array, zoom_size, anti_aliasing=False) * 1e5
    # print(rescaled)
    # ----------- array thresholding part ----------- #
    # th = 2
    # rescaled[np.where(rescaled < th)] = 0
    # rescaled[np.where(rescaled > th)] = 1
    return rescaled

# ----------- check sample  -------------- #
# file_name = "/home/soopil/Desktop/Dataset/brain_ewha_early/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
# data, header = nrrd.read(file_name)
# # ----------- recaling part -------------- #s
# rescaled_data = rescale_threshold_3D(data, (256,256,256))
# data = rescaled_data
# print(data[84,165,161])
# print(np.where(data))
#
# hist, bin_edges = np.histogram(data.flatten)
# print(hist)

# plt.hist(data.flatten(), bins='auto')
# plt.show()

# implement both box-counting and Lacunarity
# result_FD = box_count_FD(data)
# result_LAC = lacunarity(data)
# line = '/'+str(np.shape(data))+'/'+','.join(str(e) for e in result_FD)+'/'+','.join(str(e) for e in result_LAC)
# print(line)
# assert False
# ----------- pipeline part -------------- #

# dir_path =  "/home/soopil/Desktop/Dataset/EWHA_brain_tumor/0_Fwd_ MENINGIOMA 추가 자료 1_190711/MASKS" # SINCHON : internal data set
# dir_path =  "/home/soopil/Desktop/Dataset/EWHA_brain_tumor/Meningioma_External validation/EWHA/MASKS" # EWHA : external validation set
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha_early/Meningioma_only_T1C_masks" # SINCHON : original internal validation set

file_list = os.listdir(dir_path)
print(len(file_list))

fd_result_file = "../fd_result/SINCHON_FD_result_20190731_orig.txt"
fd = open(fd_result_file, 'a+t')
fd.write('box counting fractal dimension.\n')

for i, name in enumerate(sorted(file_list)):
    split = name.split('_')
    if split[0] == 'norm':
        subj_name = split[1]
    else:
        subj_name = split[0]
    # print(name, subj_name)
    # print()
    file_path = os.path.join(dir_path, name)
    data, header = nrrd.read(file_path)

    # ----------- recaling part -------------- #s
    is_rescale = False
    if is_rescale:
        rescaled_data = rescale_threshold_3D(data, (256,256,256))
        data = rescaled_data

    # print(subj_name, np.shape(data))
    # print(str(np.shape(data)))

    # implement both box-counting and Lacunarity
    result_FD = box_count_FD(data)
    result_LAC = lacunarity(data)
    line = subj_name +'/'+name+'/'+str(np.shape(data))+'/'+','.join(str(e) for e in result_FD)+'/'+','.join(str(e) for e in result_LAC)

    print(i, line)
    print()
    fd.write(line+'\n')

    # assert False

fd.close()