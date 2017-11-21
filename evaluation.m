clc
clear
close all

run('D:/Softwares/vlfeat-0.9.20/toolbox/vl_setup');
files = dir('D:\Mandeep\Summer\BTP\Tracking\HOG\negatives_test_data\*.jpg');
files2 = dir('D:\Mandeep\Summer\BTP\Tracking\HOG\negatives_training_data\*.jpg');
count = length(files);
count2 = length(files2)

load('weights.mat');
load('offsets.mat');
load('vocabulary.mat');
i_n = 0;
for i = 1:6000
    filename = strcat('D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_test_data\',num2str(i),'.jpg');
    img = imread(filename);
    features = extract_features(img,vocabulary);
    [plane,score] = classify(features,weights,offsets);
    if(plane)
        i
        i_n = i_n + 1;
        count2 = count2 + 1;
        filename = strcat('D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_training_data\',num2str(count2),'.jpg');
        imwrite(img,filename);
    end    
end


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
