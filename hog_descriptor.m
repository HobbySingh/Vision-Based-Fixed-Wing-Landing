function [vocabulary] = hog_descriptor()
    %no_of_images to consider while creating set of sift features
    
    fprintf('Making Vocabulary : Computing HOG Features | Positives\n'); 
    %Step size for SIFT feature detection
    step_size = 4;

    %Dataset Directories
    plane_training_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\tight_dataset\';
    non_plane_training_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\Negatives2\';

    no_of_features = 10;

    %Computing sift for 1000 positive images
    counter1 = 1;
    while (counter1 <= 1000)
        filename = strcat(plane_training_source,num2str(counter1),'.jpg');
        if exist(filename,'file')
            %fprintf('file exists\n');
            img = im2single(imread(filename));
            %imshow(img);
            hog = vl_hog(img,step_size);
            [x,y,z] = size(hog);
            hog_features = (reshape(hog,[x*y,z]))';
            total_features = size(hog,2);
            %descriptors = [descriptors, hog];
            %return;
            %choosing 10 random features
            random_features=randi([1 total_features],1,no_of_features);
            descriptors(:, no_of_features*(counter1-1)+1 : no_of_features*counter1)= hog_features(:,random_features);        
            counter1 = counter1 + 1;
        else
            fprintf("File doesnt exist\n");
            counter1 = counter1 + 1;
            return;
        end
    end 
    fprintf('Making Vocabulary : Computing HOG Features | Negatives\n');     
    counter2 = 1;
    no_of_features = 10;
    random_numbers=randi([1 2000],1,1000);
    while (counter2 <= 1000)
        if (mod(counter2,100) == 0)
            fprintf("Done %d images\n",counter2);
        end
        filename = strcat(non_plane_training_source,num2str(random_numbers(counter2)),'.jpg');
        if exist(filename,'file')
            %fprintf('file exists\n');
            img = im2single(imread(filename));
%             img = imresize(img,0.2);
            %imshow(img);
            hog = vl_hog(img,step_size);
            [x,y,z] = size(hog);
            hog_features = (reshape(hog,[x*y,z]))';
            total_features = size(hog,2);
            %return;
            %choosing 10 random features
            random_features=randi([1 total_features],1,no_of_features);
            descriptors2(:, no_of_features*(counter2-1)+1 : no_of_features*counter2)= hog_features(:,random_features);        
            counter2 = counter2 + 1;
        else
            fprintf("File doesnt exist\n");
            counter2 = counter2 + 1;
            return;
        end
    end    
    descriptors = [descriptors, descriptors2];
    size(descriptors)
    fprintf('Making Vocabulary : Clustering\n');    
    vocab_size = 500;
    [centroids, ~] = vl_kmeans(single(descriptors),vocab_size,'Initialization','RANDSEL');
    vocabulary = centroids';
    size(vocabulary)
end