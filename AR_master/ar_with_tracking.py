# 这个程序使用稀疏光学跟踪一个更灵敏的程序
import cv2
import numpy as np
import math
from object_module import *
import sys
import aruco_module as aruco 
from my_constants import *
from utils import get_extended_RT

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
	p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
	p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
	d = abs(p0-p0r).reshape(-1, 2).max(-1)
	status = d < back_threshold
	return p1, status

if __name__ == '__main__':
	obj = three_d_object('data/3d_objects/low-poly-fox-by-pixelmannen.obj', 'data/3d_objects/texture.png')
	marker_colored = cv2.imread('data/m1.png')
	assert marker_colored is not None, "Could not find the aruco marker image file"
	# 解释了摄像头引起的侧翻
	marker_colored = cv2.flip(marker_colored, 1)

	marker_colored =  cv2.resize(marker_colored, (480,480), interpolation = cv2.INTER_CUBIC )
	marker = cv2.cvtColor(marker_colored, cv2.COLOR_BGR2GRAY)

	print("trying to access the webcam")
	cv2.namedWindow("webcam")
	vc = cv2.VideoCapture(0)
	assert vc.isOpened(), "couldn't access the webcam"
	
	h,w = marker.shape
	# 考虑到所有4个旋转
	marker_sig1 = aruco.get_bit_sig(marker, np.array([[0,0],[0,w], [h,w], [h,0]]).reshape(4,1,2))
	marker_sig2 = aruco.get_bit_sig(marker, np.array([[0,w], [h,w], [h,0], [0,0]]).reshape(4,1,2))
	marker_sig3 = aruco.get_bit_sig(marker, np.array([[h,w],[h,0], [0,0], [0,w]]).reshape(4,1,2))
	marker_sig4 = aruco.get_bit_sig(marker, np.array([[h,0],[0,0], [0,w], [h,w]]).reshape(4,1,2))

	sigs = [marker_sig1, marker_sig2, marker_sig3, marker_sig4]
	rval, frame = vc.read()
	assert rval, "couldn't access the webcam"
	h2, w2,  _ = frame.shape
	h_canvas = max(h, h2)
	w_canvas = w + w2
	
	################################################################

	# 初始化跟踪所需的所有变量
	p0 = None
	use_ransac = True
	print("Identifying important points to track")
	fixed_gp = cv2.goodFeaturesToTrack(marker, **feature_params)
	H_from_aruco = None # 将跟踪得到的初始单应性和部分单应性结合起来得到完全单应性
	counter = 0
	started_once = False
	lost_tracking = True # “Initially”的意思是我们还没有开始
	all_tracking_pts = fixed_gp.shape[0]
	TRACKING_QUALITY_THRESHOLD = max(4, int(0.01 * TRACKING_QUALITY_THRESHOLD_PERCENTAGE * all_tracking_pts))
	print(f"good pts in the marker are {all_tracking_pts}")
	print(f"homography is calculated from scratch when number of pts tracked falls below {TRACKING_QUALITY_THRESHOLD} ({TRACKING_QUALITY_THRESHOLD_PERCENTAGE}%); ")

	##################################################################

	while rval:
		counter += 1
		rval, frame = vc.read() # 从网络摄像头获取帧
		canvas = np.zeros((h_canvas, w_canvas, 3), np.uint8) # 创建现实与填充
		canvas[:h, :w, :] = marker_colored # 标记作为参考
		canvas[:h2 , w: , :] = np.flip(frame, axis = 1) # 现在把原始框架放进去

		frame_copy = frame.copy()
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if not lost_tracking:
			p2, trace_status = checkedTrace(frame_gray_previous, frame_gray, p1)
			p1 = p2[trace_status].copy()
			p0 = p0[trace_status].copy()
			frame_gray_previous = frame_gray
			if len(p0) < TRACKING_QUALITY_THRESHOLD:
				lost_tracking = True
				print("LOST YOU")
				continue
			H_mid, status = cv2.findHomography(p0, p1, (0, cv2.RANSAC)[use_ransac], 10.0)
			combined_homography = (H_mid).dot(H_from_aruco)
			R_T = get_extended_RT(np.float64(A), np.float64(combined_homography))
			transformation = np.float64(A).dot(R_T)
			augmented = np.flip(augment(frame, obj, transformation, marker), axis = 1)
			canvas[:h2 , w: , :] = augmented

		cv2.imshow("webcam", canvas)

		ch = 0xFF & cv2.waitKey(1)
		if ch == 27: # 按Escape推出程序
			break
		initiating_keystroke = (ch == ord(' ')) #spacebar
		time_to_refresh_from_scratch = (started_once and (counter - last_refreshed) >= FREQUENCY)
		havent_started = not started_once

		if initiating_keystroke or lost_tracking or time_to_refresh_from_scratch or havent_started:
			last_refreshed = counter
			if(time_to_refresh_from_scratch):
				print(f"refreshed, counter is {counter}")

			# wiping the state
			lost_tracking = True
			p0 = None
			H_from_aruco = None
			success = False
			try:
				# 寻找鲁棒标记
				success, H_from_aruco  = aruco.find_homography_aruco(frame_copy, marker, sigs)
			except Exception as e:
				print(e)
				print('couldnt find and draw the homography')
				pass

			if success and H_from_aruco is not None:
				started_once = True
				temp_pts = np.float32(fixed_gp).reshape(-1,1,2)
				p0 = cv2.perspectiveTransform(temp_pts,H_from_aruco)
				p1 = p0
				frame_gray_previous = frame_gray
				lost_tracking = False


		if ch == ord('r'):
			use_ransac = not use_ransac
			print("ransac turned {}".format("on" if use_ransac else "off"))

