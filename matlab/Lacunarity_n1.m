% Fractal dimension (FD) calculation using differential box counting algorithm.
% Omar S. Al-Kadi (e-mail: o.al-kadi@sussex.ac.uk)
% University of Sussex, Brighton, UK.
%------------------------------------------------------------------------
clc;
clear all;
close all;
%*---------- Reading image(s) with a specific format -----------*%
[filename, pathname, filterindex] = uigetfile( ...
{  '*.dcm','DICOM (*.dcm)'; ...
   '*.jpg','JPEG (*.jpg)'; ...
   '*.bmp','Windows Bitmap (*.bmp)'; ...
   '*.fig','Figures (*.fig)'; ...
   '*.*',  'All Files (*.*)'}, ...
   'Choose image(s) to be processed', ...
   'MultiSelect', 'on');
%if filterindex==0, break;end
filename=cellstr(filename);
rem=filename;
while true
    [token, rem] = strtok(rem,'.');
    if isempty(char(rem)), break; end
end
num=numel(token);
h = waitbar(0,'Loading selected image(s) in progress...');
for i= 1:num
    switch(lower(char(token(i))))
        case 'dcm'
            J= dicomread(horzcat(pathname,char(filename(i))));
            if ndims(J) >3
               Js=squeeze(J);
               Js1=Js(:,:,1);
               Js2=Js(:,:,2);
               Js3=Js(:,:,3);
               Jc=cat(3,Js1,Js2,Js3);
               Ig=rgb2gray(Jc);
               I{i}= Ig;
             else I{i}= J; end
            waitbar(i/num)
        case {'fig','m'}
            I{i}= load(horzcat(pathname,char(filename(i))));
            waitbar(i/num)
        case {'jpg','jpeg','bmp','png','tiff','tif','gif'}
            J= imread(horzcat(pathname,char(filename(i))));
            if ndims(J) >2
               I{i}= rgb2gray(J);
            else I{i}= J; end
            waitbar(i/num)
        otherwise
            I{i}= load(horzcat(pathname,char(filename(i))));
            waitbar(i/num)
    end
end
close(h);
for j=1:num
[M,N]= size(I{j});
iname = char(strtok(filename(j), '.'));
%------- performing non-linear filtering on a varying size pixel block -------%
h = waitbar(0,'Performing 3-D Box Counting...');
for r=2:7
rc = @(x) floor(((max(x)-min(x))/r))+ 1; % non-linear filter
F= colfilt(I{j}, [r r],'sliding', rc);
B{r}= log(double(F * (49/(r^2))));
waitbar(r/6)
end
close(h)
i=log(2:7); % Normalised scale range vector
%------- computing the slope using linear regression -------%
Nxx=dot(i,i)-(sum(i)^2)/6;
h = waitbar(0,'Transforming to FD...');
for m = 1:M
    for n = 1:N
        fd= [B{7}(m,n), B{6}(m,n), B{5}(m,n), B{4}(m,n), B{3}(m,n), B{2}(m,n)]; % Number of boxes multiscale vector
        Nxy=dot(i,fd)-(sum(i)*sum(fd))/6; 
        FD{j}(m, n)= (Nxy/Nxx); % slope of the linear regression line
    end
    waitbar(m/M)
end
close(h)
end
%*----------- selecting a Region of Interest & finding corresponding average FD and Lacunarity -----------*%
figure,[S, c, r]= roipoly(mat2gray(FD{1}));
for j=1:size(I,2)
 