# 借用相机标定码
import numpy as np
import cv2
import glob

# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备对象分,像(0,0,0),(1,0,0),(2 0 0)……(6 5 0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# 数组用于存储来自所有图像的对象点和图像点。
objpoints = [] # 现实空间中的三维点
imgpoints = [] # 图像平面上的二维点。


# 这里你给它你的照片的路径 ----------------------------------------------------------
images = glob.glob('data/calibration_j/*.jpg')

print(len(images))
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # 找到棋盘的角落
    ret, corners = cv2.findChessboardCorners(gray, (9,7),None)

    # 如果找到，添加对象点，图像点(细化后)
    # print(ret)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # 绘制并显示角落
        img = cv2.drawChessboardCorners(img, (9,7), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(300)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# 硬编码这个发现矩阵在另一个文件----------------------------------------------------------
print(mtx)