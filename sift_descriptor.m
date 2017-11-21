function [vocabulary] = sift_descriptor()
    %no_of_images to consider while creating set of sift features
    fprintf('Making Vocabulary : Computing SIFT Features | Positives\n');
    %Step size for SIFT feature detection
    step_size = 4;

    counter = 1;
    %Dataset Directories
    vehicles_training_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\positives_training_data\';
    non_vehicles_training_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_training_data\';

    %Intialising features as sift returns 128 X no. of features
%     features = zeros(128,200*10);
    no_of_features = 10;

    %Computing sift for 100 images
    while (counter <= 2000)
        filename = strcat(vehicles_training_source,num2str(counter),'.jpg'); 
        if exist(filename,'file')
            img = im2single(rgb2gray(imread(filename)));
            [~, features] = vl_dsift(img, 'Fast', 'Step', step_size);
            total_features = size(features,2);
            %choosing 10 random features
            random_features=randi([1 total_features],1,no_of_features);
            descriptors(:, no_of_features*(counter-1)+1 : no_of_features*counter)= features(:,random_features);        
            counter = counter + 1;
        else
            counter = counter + 1;
        end
    end 
    random_numbers=randi([1 5000],1,3000);
    index = 1;
    fprintf('Making Vocabulary : Computing SIFT Features | Negatives\n');
    while (counter <= 5107) %5000 
        filename = strcat(non_vehicles_training_source,num2str(counter),'.jpg'); % random_numbers(index)
        if exist(filename,'file')
            img = im2single(rgb2gray(imread(filename)));
            [~, features] = vl_dsift(img, 'Fast', 'Step', step_size);
            total_features = size(features,2);
            %chossing 10 random features
            random_features=randi([1 total_features],1,no_of_features);
            descriptors(:, no_of_features*(counter-1)+1 : no_of_features*counter)= features(:,random_features);        
            counter = counter + 1;
            index = index + 1;
        else
            index = index + 1;
            counter = counter + 1;
        end
    end    
    size(descriptors);
    fprintf('Making Vocabulary : Clustering\n');    
    vocab_size = 500;
    [centroids, ~] = vl_kmeans(single(descriptors),vocab_size,'Initialization','RANDSEL');
    vocabulary = centroids';
    size(vocabulary)    
end