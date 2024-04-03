import numpy as np
import math

def get_extended_RT(A, H):
	# 查找并附加r3
	# A是内垫，H是估计的单应性
	H = np.float64(H) # for better precision
	A = np.float64(A)
	R_12_T = np.linalg.inv(A).dot(H)

	r1 = np.float64(R_12_T[:, 0]) #col1
	r2 = np.float64(R_12_T[:, 1]) #col2
	T = R_12_T[:, 2] # translation

	# |r1|和|r2|应该相同
	# 因为总是有一些错误，我们取square_root(|r1||r2|)作为归一化因子
	norm = np.float64(math.sqrt(np.float64(np.linalg.norm(r1)) * np.float64(np.linalg.norm(r2))))
	
	r3 = np.cross(r1,r2)/(norm)
	R_T = np.zeros((3, 4))
	R_T[:, 0] = r1
	R_T[:, 1] = r2 
	R_T[:, 2] = r3 
	R_T[:, 3] = T
	return R_T