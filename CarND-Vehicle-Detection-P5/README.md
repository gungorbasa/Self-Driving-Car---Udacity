
##Vehicle Detection with HOG Features

The goals / steps of this project are the following:

* Apply color transformation to each image and transform them to YUV color channels.
* Perform feature extraction (HOG + Color Histogram + Spatial Binning)
* Find scaler on trainin data to normalze data.
* Normalize all the features.
* Use features to train SVM classiffier. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* To eleminate false positives use heat mapping and averaging techniques.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4).
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Images/vehicle.png
[image2]: ./Images/hog_vehicle.png
[image3]: ./Images/non_vehicle.png
[image4]: ./Images/non_vehicle_hog.png
[image5]: ./Images/vehicle_rec.png
[image6]: ./Images/heat_map.png
[image7]: ./Images/one_rec.png
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)
####1. Apply color transformation.
I applied YUV color channel transformation to all images. RGB gave really bad resutls. I get really good results with YUV channels. I got around %98 accuracy on test data with 1 channel. However, I used all channels to get %99.5 accuracy.
####2. Feature Extraction.
I combined HOG features with Color Histogram and Spatial Binning features. I calculated each features individually and concatenate them later. Since, the ranges are different for different type of features, I normalized features and used normalized features. 

`FeatureDetection.py` file contains codes for these operations. Given orient, pixel per block and cell per block `get_hog_features()` method calcualtes HOG features for an image. `color_hist()` method calculates histograms for each channel for an image and stack them together. `bin_spatial()` method creates spatial features. In general, `extract_features()` method combines all these features and creates a feature vector for an image.

I used `vehicle_images` and `non_vehicle` images for this project. I use different `color schemes`, different `orientation`, `pixel per cell` and `pixel per block`. I calculated the gradients for each `8x8` region and distributed them into 9 orientation bins. Longest of these 9 orientation bins gives the dominant direciton of this 64 pixels. Multiple research study used orient as 9 and got good results with vehicle detection. Also I used 2x2 cell per block and kind of normalized in this region.

![Example Vehicle Image][image1]
![Hog Features for Vehicle Image][image2]
![Example Non Vehicle Image][image3]
![Hog Features for Non-Vehicle Image][image4]

I used multiple different numbers to optimize vechicle detection. I found a sweet spot on not having many false positives and good detection of vehicles. These numbers can be seen in `P5.ipynb` file.

####3. Classifier
I used Linear-SVM classifier to classify vehicle and non vehicle images. I splitted my data into 2 parts. I used 80% of data to train 20% of the data for test. As a resutl, on the test set, my model achieved %99 percent accuracy.

####4. Sliding Window and Vehicle Finding
I specify a region in the image. Then used sliding window technique to find all possible windows. After that, I used trained classifier to decide if the current window has vehicle or not. If there is a vehicle in the window, window is added to our list. This procedure can be seen in `FeatureDetection.py` (`slide_window()` and `search_windows()`). Below picture show the final windows.
![Detected Vehicles][image5]

####5. Decreasing False Detection
To decrease the possibility of false detection, I used heat map method. In `FeatureDetection.py` file `add_heat()` and `apply_threshold()` methods are used to create one bounding box for a car.
![Heat map for above image][image6]
![Found rectangles][image7]

####6. Application on Videos
#####Application of Pipeline Function
######Test Video
Pipeline Output

[![Test Vdeo Output](https://img.youtube.com/vi/wkf2qWc_y9U/0.jpg)](https://youtu.be/wkf2qWc_y9U)

Average Pipeline Output

[![Test Vdeo Output](https://img.youtube.com/vi/75kevuYdVv0/0.jpg)](https://youtu.be/75kevuYdVv0)

######Project Video Output

Pipeline Output

[![Test Vdeo Output](https://img.youtube.com/vi/BPjx-X8Qh04/0.jpg)](https://youtu.be/BPjx-X8Qh04)


Average Pipeline Output

[![Test Vdeo Output](https://img.youtube.com/vi/QD2G1sGXRio/0.jpg)](https://youtu.be/QD2G1sGXRio)

###Discussion

I didn't have much time to complete this assignment. Even though it was an easy one, because of time constraint, I couldn't perfect my method. Especially, after I average 25 frames, I realized that it causes some false positives. As a next step, I will try this with a Deep Learning model. Also, this is a traditional computer vision model and it takes quite a bit time to calculate.