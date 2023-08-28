# Posey



# Prerequisites


# CSV Format

There are 64 values total, with the first 63 values being keypoint detections and the last value for the binary classification.

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
20 - right foot index \


# Citations

https://medium.com/analytics-vidhya/human-pose-comparison-and-action-scoring-using-deep-learning-opencv-python-c2bdf0ddecba

https://medium.com/@cavaldovinos/human-pose-estimation-pose-similarity-dc8bf9f78556

https://www.iieta.org/journals/ts/paper/10.18280/ts.390111#:~:text=PoseNet%20provides%20a%20total%20of,and%2070%20in%20the%20face.



