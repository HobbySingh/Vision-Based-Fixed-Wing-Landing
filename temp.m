clc
clear
close all

% filename1 = 'D:\Mandeep\Summer\BTP\Tracking\tight_dataset\544.jpg';
% filename2 = 'D:\Mandeep\Summer\BTP\Tracking\Hog\test_data\11.jpg';
% 
% img1 = imread(filename1);
% imshow(img1);
% figure
% 
% hog1 = vl_hog(im2single(img1),4);
% img3 = vl_hog('render',hog1);
% imshow(img3);
% figure
% 
% img2 = imread(filename2);
% imshow(img2);
% figure
% 
% hog2 = vl_hog(im2single(img2),4);
% img4 = vl_hog('render',hog2);
% imshow(img4);

filename = 'D:\Mandeep\Summer\BTP\Tracking\Hog\positives_training_data\501.jpg';

img1 = imread(filename);
imshow(img1);
hold on;
img = im2single(rgb2gray(img1));

[f,d] = vl_sift(img);

perm = randperm(size(f,2)) ;
sel = perm(1:5) ;
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
set(h3,'color','g') ;

