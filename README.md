# Vision-Based-Fixed-Landing-

## Abstract :
Autonomous landing has become a core technology of unmanned aerial vehicle (UAV) guidance, navigation and control system. Since a single GPS provide position accuracy of at most a few meters, an airplane equipped with a single GPS only is not guaranteed to land at a designated location with a sufficient accuracy. Therefore, a vision based algorithm is proposed to improve the accuracy of landing. In this scheme, the airplane is controlled to fly into the net by directly feeding back the pitch and yaw deviation angles sensed by the camera mounted on the ground near ground control station during the terminal landing phase. The air craft is detected by using sliding window, features extraction, classification and machine learning methods. The algorithms were tried on a real dataset that was
collected using a fixed wing Skywalker 1900 aircraft.
Functions from VLFeat Library are used for training the data purposes.
The program flows as follows :

##### 1. Training_hog_svm
  - Loading the VLFeat Library
  - SIFT Descriptor is called to generate the vocbulary.

##### 2. SIFT Descriptor
  - It is used to make vocabulary for our training data using SIFT features.
  - Training dataset consists of 2000 positive samples and 5000 negative samples.
  - SIFT features for each image is computed with step size of 4 and 10 features are randomly chosen corresponding to each image.
  - All 10 features from every image are stacked in an array Descriptors which will be used to compute vocabulary. 
  - KMeans is used to find centroids in the data which will serve as our vocabulary.
  
##### 3. Bag-of-sifts/Bag-of-hogs
  - It is used to compute Bag-Of-Words representation for positives training samples.
  - Histograms are generated for each image which will work as single descriptor for that image.
  - Each value of a bin in the histogram is an index of the nearest cluster centres from a particular feature of the image.
  - Histograms are further normalized.
  - Similarly Bag-OF-Words represntation is also computed for negative training samples.
##### 4. Test_features_sift
  - It is used to BOW representation for test training samples.
  - Its mostly similar to Bag-of-sifts.

##### 5. SVM Classify
  - It is used to gain the value of weights and offsets from the result of SVM classifier to which traing descriptors fed.
  - After performing Weights{category}'*features' + Offset{category} we get count(category) no. of scores.
  - The image is classified as a category with maximum score. 

