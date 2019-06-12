file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1550930_CE-label.nrrd"
% file_name = "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/1733705_CE-label.nrrd"
dir_path =  "/home/soopil/Desktop/Dataset/brain_ewha/Meningioma_only_T1C_masks/"
file_list = dir(dir_path)
% file_number=length(file_list) % 182 files
I = nrrdread(file_name);
size(I)
num = 90;
FD = zeros(90,560,560);

for j=1:num
[M,N]= size(I(:,:,j));
%------- performing non-linear filtering on a varying size pixel block -------%
% h = waitbar(0,'Performing 3-D Box Counting...');
for r=2:7
rc = @(x) floor(((max(x)-min(x))/r))+ 1; % non-linear filter
F= colfilt(I(:,:,j), [r r],'sliding', rc);
B{r}= log(double(F * (49/(r^2))));
% B{r}
% waitbar(r/6)
end
% close(h)

i=log(2:7); % Normalised scale range vector

%------- computing the slope using linear regression -------%
Nxx=dot(i,i)-(sum(i)^2)/6;
% h = waitbar(0,'Transforming to FD...');
for m = 1:M
    for n = 1:N
        fd= [B{7}(m,n), B{6}(m,n), B{5}(m,n), B{4}(m,n), B{3}(m,n), B{2}(m,n)]; % Number of boxes multiscale vector
        Nxy=dot(i,fd)-(sum(i)*sum(fd))/6; 
        FD(j,m, n)= (Nxy/Nxx); % slope of the linear regression line
    end
%     waitbar(m/M)
end
% close(h)
end

% FD = zeros(size(I));
% size(FD)
% B = zeros(7-2+1, 560,5c
% size(B)
% assert false



% for j=1:num
% [M,N]= size(I(:,:,j));
% % iname = char(strtok(filename(j), '.'));
% 
% %------- performing non-linear filtering on a varying size pixel block -------%
% h = waitbar(0,'Performing 3-D Box Counting...');
% for r=2:7
% rc = @(x) floor(((max(x)-min(x))/r))+ 1; % non-linear filter
% F= colfilt(I(:,:,j), [r r],'sliding', rc);
% B(r)= log(double(F * (49/(r^2))));
% waitbar(r/6)
% end
% close(h)
% 
% i=log(2:7); % Normalised scale range vector
% 
% %------- computing the slope using linear regression -------%
% Nxx=dot(i,i)-(sum(i)^2)/6;
% h = waitbar(0,'Transforming to FD...');
% for m = 1:M
%     for n = 1:N
%         fd= [B(7,m,n), B(6,m,n), B(5,m,n), B(4,m,n), B(3,m,n), B(2,m,n)]; % Number of boxes multiscale vector
%         Nxy=dot(i,fd)-(sum(i)*sum(fd))/6; 
%         FD(j,m, n)= (Nxy/Nxx); % slope of the linear regression line
%     end
%     waitbar(m/M)
% end
% close(h)
% end
%  
% assert false

%*----------- selecting a Region of Interest & finding corresponding average FD and Lacunarity -----------*%
% figure,[S, c, r]= roipoly(mat2gray(FD{45}));
% S
% c
% r
% assert false
size(FD)
% FD(45,:,:)
ROI= FD(find(FD));close;
FDavg= sum(ROI)/ numel(ROI) % Average FD for selected area
FDsd= std(ROI) % Standard deviation of FD for selected area
FDlac= ((sum(ROI.^2)/(length(ROI)))./((sum(ROI)/(length(ROI)))^2))-1 % Lacunarity for selected area
