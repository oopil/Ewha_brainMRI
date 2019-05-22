file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/"
file_list = dir(dir_path)
% file_number=length(file_list) % 182 files

for a=3:182
    file_name = file_list(a).name
    number = number +1
end

print(number)
for a=0
   file_list(a).name 
end

for f = [file_list]
   disp(f.name)
end


assert false
array = nrrdread(file_name);
[n,r] = boxcount(array);
%n = boxcount(array, 'slope');
n
r