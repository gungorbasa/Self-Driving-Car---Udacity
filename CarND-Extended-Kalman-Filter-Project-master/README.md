# Extended Kalman Filter Project
Self-Driving Car Engineering Nanodegree Program Kalman Filter + Extended Kalman Filter project code. 

Kalman Filter is a linear quadratic estimator, which estimates of unknown variables that tend to be more precise than those based on a single measurement alone, by using Bayesian inference and joint probability. Given noisy sensor data, Kalman filter estimates the state of the moving object. EKF is the non-linear version of KF.

We used KF for Lidar data and EKF for Radar data and combined all these to predict state of the object.

https://en.wikipedia.org/wiki/Kalman_filter
https://en.wikipedia.org/wiki/Extended_Kalman_filter

---

# Data & Data Visualization
Data for the simulation and its visualization exist in the "data" directory. Data contains both data from Radar and Lidar sensors. This project combines both sensors and successfully predicts the location of an given moving object. 

### Visualization for "./data/sample-laser-radar-measurement-data-1.txt"
![image](https://github.com/gungorbasa/Self-Driving-Car---Udacity/blob/master/CarND-Extended-Kalman-Filter-Project-master/data/EKF_data1.png?raw=true "Visualization for sample-laser-radar-measurement-data-1")

### Visualization for "./data/sample-laser-radar-measurement-data-2.txt"
![image](https://github.com/gungorbasa/Self-Driving-Car---Udacity/blob/master/CarND-Extended-Kalman-Filter-Project-master/data/EKF_data2.png?raw=true "Visualization for sample-laser-radar-measurement-data-2")

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)


## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/12dd29d8-2755-4b1b-8e03-e8f16796bea8)
for instructions and the project rubric.
