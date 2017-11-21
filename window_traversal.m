clc
clear
close all

% shapeInserter = vision.ShapeInserter;
% files = dir('D:\Mandeep\Summer\BTP\Tracking\positives_test_dataset\*.jpg');
% count = length(files);

filename = '2.jpg'; % 77 | 654 | 352 | 447 | 1 | 2
img = imread(filename);
% img = rgb2gray(img);
% img = imresize(img,0.5);
% imshow(img);
% PTS = getrect;%[10 10 10 10];%;
% img = insertShape(img,'rectangle',PTS,'LineWidth',1);
% imshow(img);
% J = step(shapeInserter,img,PTS); 


%max_y - width of image || max_X - height of image
max_y = size(img,2); max_x = size(img,1);

%s_h - scale_height || s_w - scale_width
s_h = 20;s_w = 45;

%i - traversing left to right || j - traversing top to bottom || i_n -
%image number
i = 1; j = 1;

% while( (i <= max_y - s_w) && (j <= max_x - s_h))
i_n = 1;
% for scale = 1:-0.1:0.5
%     for i = max_x/2:10:(max_x/2 + max_x/4)
%         for j = 1:10:max_y - s_w
%         %c_i - cropped_image
%         crop = [j,i,s_w,s_h] ;
%         c_i = imcrop(img, crop);
%         filename = strcat('D:\Mandeep\Summer\BTP\Tracking\Hog\test_data3\',num2str(i_n),'.jpg');
%         imwrite(c_i,filename);
%         i_n = i_n + 1;
%         end
%     end
% end
rand_i = randi([max_x/2  (max_x - s_h)],1,500);
rand_j = randi([1  (max_y-s_w)],1,500);
scale = [ 1 0.9 0.8 0.7 0.6];
rand_k = randi([1 5],1,5);

for 
for i = 1:500
    %c_i - cropped_image
    crop = [rand_j(i),rand_i(i),s_w,s_h] ;
    c_i = imcrop(img, crop);
    filename = strcat('D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_test_data\',num2str(i_n),'.jpg');
    imwrite(c_i,filename);
    i_n = i_n + 1;
end
% 
% img = imresize(img,0.9);
% max_y = size(img,2); max_x = size(img,1);
% 
% rand_i = randi([max_x/2  (max_x - s_h)],1,300);
% rand_j = randi([1  (max_y-s_w)],1,300);
% i_n
% for i = 1:300
%     %c_i - cropped_image
%     crop = [rand_j(i),rand_i(i),s_w,s_h] ;
%     c_i = imcrop(img, crop);
%     filename = strcat('D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_test_data\',num2str(i_n),'.jpg');
%     imwrite(c_i,filename);
%     i_n = i_n + 1;
% end
% 
% img = imresize(img,0.8);
% max_y = size(img,2); max_x = size(img,1);
% 
% rand_i = randi([max_x/2  (max_x-s_h)],1,200);
% rand_j = randi([1  (max_y-s_w)],1,200);
% 
% for i = 1:200
%     %c_i - cropped_image
%     crop = [rand_j(i),rand_i(i),s_w,s_h] ;
%     c_i = imcrop(img, crop);
%     filename = strcat('D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_test_data\',num2str(i_n),'.jpg');
%     imwrite(c_i,filename);
%     i_n = i_n + 1;
% end

img = imresize(img,0.5);
max_y = size(img,2); max_x = size(img,1);

rand_i = randi([1  (max_x)],1,200);
rand_j = randi([1  (max_y)],1,200);
crop = [2,2,(1/0.5)*45,(1/0.5)*20] ;
c_i = imcrop(img, crop);
imshow(c_i);
