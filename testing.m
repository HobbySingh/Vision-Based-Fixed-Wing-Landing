clc
clear
close all

run('D:/Softwares/vlfeat-0.9.20/toolbox/vl_setup');
plane_source = dir('D:\Mandeep\Summer\BTP\Tracking\Hog\positives_test_data\');
count = length(plane_source);
plane_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\positives_test_data\';

load('weights.mat');
load('offsets.mat');
load('vocabulary.mat');

false_positive = 0;
images_tried = 0;
for counter = 1:count-2 
    num2str(counter);
    filename = strcat(plane_source,num2str(counter),'.jpg'); 
    c_i = imread(filename);
    features = extract_features(c_i,vocabulary);
    [plane,score] = classify(features,weights,offsets);
    if(plane)
        false_positive = false_positive + 1;
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
