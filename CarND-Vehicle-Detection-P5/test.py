import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from FeatureDetection import *
from sklearn.model_selection import train_test_split
from Drawer import *
from pathlib import Path
from sklearn.externals import joblib

color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [500, None] # Min and max in y to search in slide_window()

print(color_space, "channel: ", hog_channel)

def read_image(name):
    img = cv2.imread(name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_features(name):
    img = read_image(name)
    extractor = FeatureExtractor()
    return extractor, extractor.extract_features(img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)[0]

if Path("./model.pkl").is_file():
    print("Saved model is Loaded..")
    svc = joblib.load('./model.pkl')
    extractor = joblib.load('extractor.pkl')
    X_scaler = joblib.load('./x_scalar.pkl')
else:
    vehicles = glob.glob('./train_images/vehicles/**/*.png', recursive=True)
    non_vehicles = glob.glob('./train_images/non-vehicles/**/*.png', recursive=True)

    vehicle_features = []
    non_vehicle_features = []

    extractor = None
    for vehicle in vehicles:
        extractor, features = create_features(vehicle)
        vehicle_features.append(features)

    for non_vehicle in non_vehicles:
        extractor, features = create_features(non_vehicle)
        non_vehicle_features.append(features)

    print("Vehicle Features Shape: ", np.shape(vehicle_features))
    print("Non-Vehicle Features Shape: ", np.shape(non_vehicle_features))

    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
    print(np.shape(X))
    # Normalizes the data
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    joblib.dump(svc, './model.pkl')
    joblib.dump(extractor, './extractor.pkl')
    joblib.dump(X_scaler, './x_scalar.pkl')
    print("New model is saved..")

img = read_image("bbox.jpg")

def pipeline(img):
    windows1 = extractor.slide_window(img, x_start_stop=[100, -100],
        y_start_stop=[450,None],xy_window=(64, 32), xy_overlap=(0.5, 0.5))
    windows2 = extractor.slide_window(img, x_start_stop=[None, None],
        y_start_stop=[450,None],xy_window=(128, 64), xy_overlap=(0.5, 0.5))
    windows3 = extractor.slide_window(img, x_start_stop=[None, None],
        y_start_stop=[450,None],xy_window=(256, 128), xy_overlap=(0.5, 0.5))
    windows4 = extractor.slide_window(img, x_start_stop=[None, None],
        y_start_stop=[450,None],xy_window=(32, 16), xy_overlap=(0.5, 0.5))


    windows = windows1+windows2+windows3+windows4
    # windows2 = extractor.slide_window(img, x_start_stop=[None, None],
    #     y_start_stop=[564,None],xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    # print(windows2)
    # windows = windows1+windows2

    hot_windows = extractor.search_windows(img, windows, svc, X_scaler,
        color_space=color_space,spatial_size=spatial_size, hist_bins=hist_bins,
        orient=orient, pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
        hog_channel=hog_channel, spatial_feat=spatial_feat,hist_feat=hist_feat,
        hog_feat=hog_feat)

    # drawer = Rectangle()
    # window_img = drawer.draw(img, hot_windows, color=(0, 0, 255), thick=6)
    # return window_img
    return extractor.find_boxes(img, hot_windows)

img = read_image("bbox.jpg")
# drawer = Rectangle()
# window_img = drawer.draw(img, hot_windows, color=(0, 0, 255), thick=6)
plt.imshow(pipeline(img))
plt.show()

# from moviepy.editor import VideoFileClip
# white_output = 'test_video_output.mp4'
# clip1 = VideoFileClip("test_video.mp4")
# white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)


# print(cv2.imread('./train_images/vehicles/GTI_Far/Image0000.png'))
