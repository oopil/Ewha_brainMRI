file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/"
file_list = dir(dir_path)
% file_number=length(file_list) % 182 files
I = nrrdread(file_name);

i=log(2:7); % Normalised scale range vector
i
sum(i)

a = [[2, 0,0,0,3]
    [1,2,3,4,0]]
size(a)
size(a,1)
size(a,2)
find(a)

%------- computing the slope using linear regression -------%
dot(i,i)

% Nxx=dot(i,i)-(sum(i)^2)/6
assert false

% ROI= I(find(S==1));close;
% ROI= find(I==7)
ROI= find(FD)
sum(ROI.^2)
I(12112636)
sum(ROI)
numel(ROI)
FDavg= sum(ROI)/ numel(ROI) % Average FD for selected area
FDsd= std(ROI) % Standard deviation of FD for selected area
FDlac= ((sum(ROI.^2)/(length(ROI)))./((sum(ROI)/(length(ROI)))^2))-1 % Lacunarity for selected area

assert false

array = nrrdread(file_name);

boxcount(array)
figure, boxcount(array, 'slope')
% [n,r]=boxcount(array, 'slope')
% print(n,r)


% for a=3:182
%     file_name = file_list(a).name
%     number = number +1
% end
% 
% print(number)
% for a=0
%    file_list(a).name 
% end
% 
% for f = [file_list]
%    disp(f.name)
% end
% 
% 
% assert false
% array = nrrdread(file_name);
% [n,r] = boxcount(array);
%n = boxcount(array, 'slope');