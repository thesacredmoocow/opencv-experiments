import cv2
import numpy as np
import time
ix, iy, k = 200,200,1
def onMouse(event, x, y, flag, param):
	global ix,iy,k
	if event == cv2.EVENT_LBUTTONDOWN:
		ix,iy = x,y 
		k = -1

cv2.namedWindow("window")
cv2.setMouseCallback("window", onMouse)

cap = cv2.VideoCapture('videos/fight4.avi')
cap.set(cv2.CAP_PROP_POS_FRAMES, 13650)

pt_A = [357, 28]
pt_B = [-289, 673]
pt_C = [1684, 710]
pt_D = [1113, 31]

width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))


height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
						[0, 720],
						[720, 720],
						[720, 0]])
M = cv2.getPerspectiveTransform(input_pts,output_pts)

while True:
	_, frm = cap.read()
	frm = cv2.warpPerspective(frm,M,(720, 720),flags=cv2.INTER_LINEAR)

	cv2.imshow("window", frm)
	time.sleep(0.05)
	if cv2.waitKey(1) == 27 or k == -1:
		old_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
		cv2.destroyAllWindows()
		break

old_pts1 = np.array([[ix,iy]], dtype="float32").reshape(-1,1,2)
old_pts2 = np.array([[ix,iy-30]], dtype="float32").reshape(-1,1,2)
mask = np.zeros_like(frm)

while True:
	_, frame2 = cap.read()
	frame2 = cv2.warpPerspective(frame2,M,(720, 720),flags=cv2.INTER_LINEAR)

	new_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

	new_pts,status,err = cv2.calcOpticalFlowPyrLK(old_gray, 
                         new_gray, 
                         old_pts1, 
                         None, maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                         15, 0.08))
	

	new_pts2,status,err = cv2.calcOpticalFlowPyrLK(old_gray, 
                         new_gray, 
                         old_pts2, 
                         None, maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                         15, 0.08))
	pt1 = (int(new_pts[0][0][0]), int(new_pts[0][0][1]))
	pt2 = (int(new_pts2[0][0][0]), int(new_pts2[0][0][1]))
	#print(pt)
	cv2.circle(mask, pt1, 2, (0,255,0), 2)
	cv2.circle(mask, pt2, 2, (255,0,0), 2)
	combined = cv2.addWeighted(frame2, 0.7, mask, 0.3, 0.1)

	cv2.imshow("new win", mask)
	cv2.imshow("wind", combined)

	old_gray = new_gray.copy()
	old_pts2 = new_pts2.copy()
	old_pts1 = new_pts.copy()

	if cv2.waitKey(1) == 27:
		cap.release()
		cv2.destroyAllWindows()
		break 


