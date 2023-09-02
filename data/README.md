# CSV Format

There are 64 values total, with the first 63 values being keypoint detections and the last value for the binary classification.  1 is for the up position, 0 indicates a down position.

Each keypoint detection has 3 values, x-coor, y-coor, z-coor, in that order, such that given the i-th keypoint the csv is formatted as (with a header representing column indices) 

0, 1, 2, ... i, ... 63 
x0, y0, z0, x1, y1, z1, ... xi, yi, zi, ... x62, y62, z62, classification

The mapping of the index of each keypoint index to each bodypart is as follows:

0 - nose \
1 - left eye \
2 - right eye \
3 - left ear \
4 - right ear \
5 - left shoulder \
6 - right shoulder \
7 - left elbow \
8 - right elbow \
9 - left wrist \
10 - right wrist \
11 - left hip \
12 - right hip \
13 - left knee \
14 - right knee \
15 - left ankle \
16 - right ankle \
17 - left heel \
18 - right heel \
19 - left foot index \
20 - right foot index 
