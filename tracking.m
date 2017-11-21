clc
clear
close all

run('D:/Softwares/vlfeat-0.9.20/toolbox/vl_setup');
load('weights.mat');
load('offsets.mat');
load('vocabulary.mat');

hVideoSrc = vision.VideoFileReader('Landing2.mp4');
hVideoOut = vision.VideoPlayer;
shapeInserter = vision.ShapeInserter;
tracker = vision.PointTracker('MaxBidirectionalError',2);
iter = 1;
frame = 0;
x = 205;y = 160;h = 15; w = 25;

old_bboxes = {};
new_bboxes = {};
bboxes = {};
begin = 1;
temp = 0;
consistent_box = [];
while ~isDone(hVideoSrc)
    RGB = step(hVideoSrc);
%     imshow(RGB);
    frame = frame + 1;
    if(~temp)
        new_bboxes = detection(RGB,vocabulary,weights,offsets);
        if(size(new_bboxes,1) == 0)
            continue;
        end
    else
        c_i = imcrop(RGB,crop);
        features = extract_features(c_i,vocabulary);
        [plane,score] = classify(features,weights,offsets);
        if(~plane)
            fprintf('Plane box not classified as true\n'); 
            %begin = 1;
            %temp = 0;
        else
            fprintf('Plane box classified as true');
        end
        
    end
    RGB = rgb2gray(RGB);
    [bboxes,consistent_box] = consistency(new_bboxes,old_bboxes);
    %bboxes{1,1};
    %bboxes{1,2};
    old_bboxes = bboxes
    if ( size(consistent_box,2) == 4 && begin == 1 && temp == 0)
        fprintf('Consistency Begin \n');
        PTS = consistent_box;
        x = PTS(1);y = PTS(2);w = PTS(3);h = PTS(4);          
        J = step(shapeInserter,RGB,PTS); 
        imshow(J);
        corners = detectMinEigenFeatures(RGB, 'ROI', PTS, 'MinQuality', 0.0001);
        hold on;
        plot(corners.selectStrongest(200));  
        initialize(tracker,corners.Location,RGB);
        bboxPoints = [x, y; x+w, y; x+w, y+h; x, y+h];
        p_bboxPoints = bboxPoints;
        crop = bbox_to_roi(bboxPoints);
        oldPoints = corners.Location;
        begin = 0;temp = 1;
    elseif(temp)
        
        [points, validity] = step(tracker,RGB);
        visiblePoints = points(validity, :);
        oldInliers = oldPoints(validity, :);

        if size(visiblePoints, 1) >= 2 % need at least 2 points

            % Estimate the geometric transformation between the old points
            % and the new points and eliminate outliers
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box points
            p_bboxPoints = bboxPoints;
            bboxPoints = transformPointsForward(xform, bboxPoints);
            crop = bbox_to_roi(bboxPoints);
            bboxPolygon = reshape(bboxPoints', 1, []);
             RGB = insertShape(RGB, 'Polygon', bboxPolygon, ...
                'LineWidth', 2);
            RGB = insertMarker(RGB, visiblePoints, '+', ...
                'Color', 'white');
            RGB = rgb2gray(RGB);
            oldPoints = visiblePoints;
            setPoints(tracker, oldPoints);
        else
            release(tracker);
            release(shapeInserter);
            fprintf("Releasing Tracker\n");
            begin = 1;temp = 0;
        end
    end
step(hVideoOut, RGB);
end 

function [cell_bboxes,consistent_box] = consistency(new_bboxes,old_bboxes)
    fprintf('Checking consistency\n');
    cell_bboxes = {};
    consistent_box = [];
    if(size(old_bboxes) == 0)
        cell_bboxes = new_bboxes;
        return
    end

    num_new = size(new_bboxes{1,1},1);
    num_old = size(old_bboxes{1,1},1);
    bboxes = [];
    scores = [];
    n = 1;
    for i = 1:num_new
        a1 = [new_bboxes{1,1}(i,1),new_bboxes{1,1}(i,2)];
        for j = 1:num_old
            a2 = [old_bboxes{1,1}(j,1),new_bboxes{1,1}(j,2)];
            dist = pdist([a1;a2],'euclidean')
            if(dist < 56)
                fprintf('Match\n');
                bboxes = [bboxes;new_bboxes{1,1}(i,:)];
                old_bboxes{1,2}(j,1)
                scores = [scores;old_bboxes{1,2}(j,1) + 1]
            end
        end
    end
    [M,I] = max(scores);
    if(M(1) > 1)
        fprintf('Consistent with score of : %d\n', M(1));
        consistent_box = bboxes(I(1),:)
    end
    
    cell_bboxes = {};
    cell_bboxes{1,1} = bboxes;
    cell_bboxes{1,2} = scores;
end
function roi_pts = bbox_to_roi(bbox_pts)
    x = bbox_pts(1,1); y = bbox_pts(1,2);
    w = bbox_pts(2,1) - bbox_pts(1,1);
    h = bbox_pts(3,2) - bbox_pts(2,2);
    roi_pts = [x,y,w,h];    
end

function bboxes = detection(img,vocabulary,weights,offsets)
    fprintf('Detection Happening\n');
    bboxes = {};
    original = img;
    i = 1; j = 1;
    i_n = 1;
    bbox ={};
    scores = [];
    images_tried = 0;
    for scale = 1:-0.1:1
        img = imresize(original,scale);
        max_y = size(img,2); max_x = size(img,1);
        s_h = 20;s_w = 45;
        for i = max_x/2:10:(max_x/2 + max_x/4)
            for j = 1:10:max_y - s_w
                crop = [j,i,s_w,s_h] ;
                c_i = imcrop(img, crop);
                features = extract_features(c_i,vocabulary);
                [plane,score] = classify(features,weights,offsets);
                if(plane)
                    i_n
                    scores = [scores;score];
                    crop = [(crop(1)/scale), (crop(2)/scale), (crop(3)/scale),(crop(4)/scale)];
                    bbox{i_n,1} = crop;
                    i_n = i_n + 1;
                end
                plane = 0;
                images_tried = images_tried + 1;
            end
        end
        images_tried
        images_tried = 0;
    end
    bbox = cell2mat(bbox);
    if(size(bbox,1) == 0)
        return ;
    end
    [selectedBbox] = selectStrongestBbox(bbox,scores,'RatioType','Min','OverlapThreshold',0.2);
    img = original;
    for i = 1:size(selectedBbox,1)
        img = insertShape(img,'rectangle',selectedBbox(i,:),'LineWidth',1);

    end
    imshow(img);    
    bboxes{1,1} = selectedBbox;
    bboxes{1,2} = zeros(size(selectedBbox,1),1);
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