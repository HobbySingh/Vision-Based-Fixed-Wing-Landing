function [image_features] = test_features(vocabulary) 


%Step size for SIFT detection
step_size=4;
vocab_size = size(vocabulary, 1);

plane_source = 'D:\Mandeep\Summer\BTP\Tracking\Hog\test_data\';

files = dir('D:\Mandeep\Summer\BTP\Tracking\Hog\test_data\*.jpg');
count = length(files);
total_images = count;

counter = 1;
image_counter = 1;
fprintf('Computing Bag of words\n');
while(image_counter <= total_images)
        filename = strcat(plane_source,num2str(counter),'.jpg');
        if exist(filename,'file')
            if (mod(image_counter,100)==0)
                fprintf('      ..Processed %d Images\n',image_counter);
                %fprintf(' Size of hog features : \n');
                %size(hog_features)                
            end
            try
            img = im2single(imread(filename));
            catch
                fprintf('Probelm in file %d',counter);
                counter = counter + 1;
                continue
            end
            %gives 128 x num_of_features with one descriptor per column
            hog = vl_hog(img,step_size);
            [x,y,z] = size(hog);
            hog_features = (reshape(hog,[x*y,z]))';
            %to reduce precision to 32
            features = single(hog_features);
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
end
