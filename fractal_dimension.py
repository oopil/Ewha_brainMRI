import os
import matlab.engine


file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/"

file_list = os.listdir(dir_path)
print(file_list)
eng = matlab.engine.start_matlab()
tf = eng.boxcount()
