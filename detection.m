clc
clear
close all

run('D:/Softwares/vlfeat-0.9.20/toolbox/vl_setup');
files = dir('D:\Mandeep\Summer\BTP\Tracking\negatives_training_data\*.jpg');
count = length(files);
files2 = dir('D:\Mandeep\Summer\BTP\Tracking\HOG\negatives_training_data\*.jpg');
count2 = length(files2);

load('weights.mat');
load('offsets.mat');
load('vocabulary.mat');

filename = '77.jpg';%1 | 2 | 352| 447 |654 | 77 |
img = imread(filename);
original = img;
%imshow(img);


%i - traversing left to right || j - traversing top to bottom || i_n -
%image number
i = 1; j = 1;
% while( (i <= max_y - s_w) && (j <= max_x - s_h))
i_n = 1;
bbox ={};
scores = [];
images_tried = 0;
for scale = 1:-0.1:0.6
    img = imresize(original,scale);
    %max_y - width of image || max_X - height of image
    max_y = size(img,2); max_x = size(img,1);
    %s_h - scale_height || s_w - scale_width
    s_h = 20;s_w = 45;
    for i = max_x/2:10:(max_x/2 + max_x/4)
        for j = 1:10:max_y - s_w
            %c_i - cropped_image
            crop = [j,i,s_w,s_h] ;
            c_i = imcrop(img, crop);
            %c_i = imread('D:\Mandeep\Summer\BTP\Tracking\Hog\test_data3\1762.jpg');
            features = extract_features(c_i,vocabulary);
            [plane,score] = classify(features,weights,offsets);
            if(plane)
                i_n
                scores = [scores;score];
                crop = [(crop(1)/scale), (crop(2)/scale), (crop(3)/scale),(crop(4)/scale)];
                bbox{i_n,1} = crop;
                i_n = i_n + 1;
                count2 = count2 + 1;
                filename = strcat('D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_training_data\',num2str(count2),'.jpg');
                imwrite(c_i,filename);
            end
            plane = 0;
            images_tried = images_tried + 1;
        end
    end
    images_tried
    images_tried = 0;
end
bbox = cell2mat(bbox);

[selectedBbox,selectedScore] = selectStrongestBbox(bbox,scores,'RatioType','Min','OverlapThreshold',0.2);
img = original;
for i = 1:size(selectedBbox,1)
    img = insertShape(img,'rectangle',selectedBbox(i,:),'LineWidth',1);
    
end
imshow(img);

function [features] = extract_features(img,vocabulary)
    img = im2single(rgb2gray(img));
    vocab_size = size(vocabulary, 1);
    [~, features] = vl_dsift(img, 'Fast', 'Step', 4);
    features = single(features);
    [indices] = knnsearch(vocabulary, features');
    imhist=histc(indices, 1:vocab_size);
    imhist_norm=imhist./numel(imhist);
    features = imhist_norm';    
end

function [plane,score] = classify(features,weights,offsets)
    training_score = [];
    for i = 1:2
        training_score = [training_score; weights{i}'*features' + offsets{i}];
    end
    [~,label_indices] = max(training_score);
    if (label_indices == 2)
        score = training_score(label_indices);
        plane = 1;
    else
        plane = 0;
        score = 0;
    end
end
