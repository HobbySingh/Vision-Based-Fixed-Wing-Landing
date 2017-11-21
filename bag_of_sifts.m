function [image_features] = bag_of_sifts(vocabulary,type_of_dataset) 


%Step size for SIFT detection
step_size=4;
size(vocabulary);
vocab_size = size(vocabulary, 1);

if(type_of_dataset)
    plane_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\positives_training_data\';
    non_plane_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_training_data\';
    fprintf('Will find features for training datasets\n');
    total_images_positive = 2000;
    total_images_negative = 5107; %3000
    random_numbers=randi([1 5000],1,total_images_negative);    
%     image_features = zeros(total_images, vocab_size);
else
    plane_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\positives_test_dataset\';
    non_plane_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\negatives_test_dataset\';    
    fprintf('Will find features for testing datasets \n');
    total_images_positive = 5;
    total_images_negative = 5;
    random_numbers=randi([1 20],1,total_images_negative);
%     image_features = zeros(total_images, vocab_size);
end   
%generating histograms of 2000 bins , where value of bin tells most no. of
%matches with different cluster centres


%for plane dataset

counter = 1;
image_counter = 1;
fprintf('Computing Bag of words : Positives\n');
while(image_counter <= total_images_positive)
        filename = strcat(plane_source,num2str(counter),'.jpg');
        if exist(filename,'file')
            if (mod(image_counter,100)==0)
                fprintf('      ..Processed %d Images\n',image_counter);
                %fprintf(' Size of hog features : \n');
                %size(hog_features)                
            end
            try
            img = im2single(rgb2gray(imread(filename)));
            catch
                fprintf('Probelm in file %d',counter);
                counter = counter + 1;
                continue
            end
            %gives 128 x num_of_features with one descriptor per column
            %hog = vl_hog(img,step_size);
            %[x,y,z] = size(hog);
            %hog_features = (reshape(hog,[x*y,z]))';
            [~, features] = vl_dsift(img, 'Fast', 'Step', step_size);
            %to reduce precision to 32
            features = single(features);
            size(features);
            [indices, distances] = knnsearch(vocabulary, features');
            
            %generating histogram and then normalizing it
            imhist=histc(indices, 1:vocab_size);
            imhist_norm=imhist./numel(imhist);
            image_features(image_counter,:) = imhist_norm';      
            image_counter = image_counter + 1;
            counter = counter + 1;
        else
            counter = counter + 1;
        end    
end

fprintf('Computing Bag of words : Negatives\n');
counter2 = 1;

% for non_vehicle_dataset
image_counter = 1;
while(image_counter <= total_images_negative)    
        filename = strcat(non_plane_source,num2str(random_numbers(counter2)),'.jpg');
        if exist(filename,'file')
            if (mod(image_counter,100)==0)
                fprintf('      ..Processed %d Images\n',image_counter);
                %fprintf(' Size of hog features : \n');
                %size(hog_features)
            end
            try
            img = im2single(rgb2gray(imread(filename)));
%             img = imresize(img,0.2);
            catch
                fprintf('Probelm in file %d',counter2);
                counter2 = counter2 + 1;
                continue
            end
            %fprintf('Processed %d Images\n',image_counter);
            %hog = vl_hog(img,step_size);
            %[x,y,z] = size(hog);
            %hog_features = (reshape(hog,[x*y,z]))';
            %to reduce precision to 32
            [~, features] = vl_dsift(img, 'Fast', 'Step', step_size);
            features = single(features);
            
            %Find the nearest cluster center in vocabulary for each local feature
            %in the image based on the Euclidean distance
            %[indices, distances] = KNNSEARCH(vocabulary,features) returns a vector
            %   distances containing the distances between each row of features and its
            %   closest point in vocabulary. Each row in 'indices' contains the index of
            %   the nearest neighbor in vocabulary for the corresponding row in features.
            [indices, distances] = knnsearch(vocabulary, features');
            
            %generating histogram and then normalizing it
            hist=histc(indices, 1:vocab_size);
            hist_norm=hist./numel(hist);
            image_features2(image_counter,:) = hist_norm';
            
            image_counter = image_counter + 1;
            counter2 = counter2 + 1;
        else
            counter2 = counter2 + 1;
        end
end
        fprintf('Size image features : %d\n', size(image_features,1));
        fprintf('Size image features2 : %d\n', size(image_features2,1));
        image_features = [image_features; image_features2];
        fprintf('Size image features again: %d\n', size(image_features,1));
end        