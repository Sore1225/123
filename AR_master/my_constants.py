import numpy as np
import cv2

# 硬编码的内在矩阵，我的网络摄像头
A = [[1019.37187, 0, 618.709848], [0, 1024.2138, 327.280578], [0, 0, 1]] 
A = np.array(A)

# 特定于跟踪实现的常量
lk_params = dict( winSize  = (19, 19),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.01,
                       minDistance = 8,
                       blockSize = 19 )

FREQUENCY = 100 # of finding the aruco marker from scratch
TRACKING_QUALITY_THRESHOLD_PERCENTAGE = 100 # 减少这个数字将使程序容忍较差的跟踪，而无需刷新来修复它


