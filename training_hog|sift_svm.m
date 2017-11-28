clc
clear
close all
% Using SIFT from VL FEAT | including path
run('D:/Softwares/vlfeat-0.9.20/toolbox/vl_setup');

%{
 no. of training samples use 
 posiitves : 1000
 negatives : 1000
 total no. of features : 2000 x 10 = 20,000
%}

%Vocab size : 2000 cluster centers

%Generate Features
% descriptors =hog_descriptor();
if ~exist('vocabulary.mat', 'file')
    %vocabulary = hog_descriptor();
    vocabulary = sift_descriptor();
    save('vocabulary.mat', 'vocabulary');
else
    load('vocabulary.mat');
    fprintf(' Vocabulary.mat exists\n');
end
% load('vocabulary.mat','vocabulary');
%Get bag of features for both all datasets
fprintf('BOW for Training Dataset\n');
if ~exist('training_image_features.mat', 'file')
    %training_image_features = bag_of_hogs(vocabulary,1);
    training_image_features = bag_of_sifts(vocabulary,1);
    save('training_image_features.mat', 'training_image_features');
else
    load('training_image_features.mat');
    fprintf(' Training_IMage_Features.mat exists\n');
end

fprintf('BOW for Testing Dataset\n');
if ~exist('testing_image_features.mat', 'file')
    %testing_image_features = test_features(vocabulary);
    testing_image_features = test_features_sift(vocabulary);
    save('testing_image_features.mat', 'testing_image_features');
else
    load('testing_image_features.mat');
    fprintf(' Testing_Image_Features.mat exists\n');
end

[predicted_categories, label_indices,weights, offsets] = svm_classify(testing_image_features, training_image_features);

if ~exist('weigths.mat', 'file')
    save('weights.mat', 'weights');
end
if ~exist('offsets.mat', 'file')
    save('offsets.mat', 'offsets');
end

    
