
import cv2 as cv 
import numpy as np 
import time
  
# The video feed is read in as 
# a VideoCapture object 
cap = cv.VideoCapture('videos/fight5.avi')
cap.set(cv.CAP_PROP_POS_FRAMES, 11100)

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
M = cv.getPerspectiveTransform(input_pts,output_pts)
  
# ret = a boolean return value from 
# getting the frame, first_frame = the 
# first frame in the entire video sequence 
ret, first_frame = cap.read() 
first_frame = cv.warpPerspective(first_frame,M,(720, 720),flags=cv.INTER_LINEAR)
  
# Converts frame to grayscale because we 
# only need the luminance channel for 
# detecting edges - less computationally  
# expensive 
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) 
  
# Creates an image filled with zero 
# intensities with the same dimensions  
# as the frame 
mask = np.zeros_like(first_frame) 
  
# Sets image saturation to maximum 
mask[..., 1] = 255
nvof = cv.cuda_NvidiaOpticalFlow_1_0.create((first_frame.shape[1], first_frame.shape[0]), 10, True, False, False, 0)
nvof2 = cv.cuda_NvidiaOpticalFlow_2_0.create((first_frame.shape[1], first_frame.shape[0]), perfPreset=20, enableTemporalHints=True)

print(nvof2.getGridSize())
lasttime = time.time()
while(cap.isOpened()): 
      
    # ret = a boolean return value from getting 
    # the frame, frame = the current frame being 
    # projected in the video 
    ret, frame = cap.read() 
    ret, frame = cap.read() 
    frame = cv.warpPerspective(frame,M,(720, 720),flags=cv.INTER_LINEAR)
    
    # Opens a new window and displays the input 
    # frame 
    #cv.imshow("input", frame) 
      
    # Converts each frame to grayscale - we previously  
    # only converted the first frame to grayscale 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    
    start = time.time()
    # Calculates dense optical flow by Farneback method 
    flow = nvof2.calc(prev_gray, gray, None)
    print(time.time() - start)
    mid = time.time()
    flowUpSampled = nvof.upSampler(flow[0], (prev_gray.shape[1], prev_gray.shape[0]), nvof2.getGridSize(), None)
    #print(time.time() - mid)
    #cv.writeOpticalFlow('OpticalFlow.flo', flowUpSampled)  
    #flow = cv.calcOpticalFlowFarneback(prev_gray, gray,  
    #                                   None, 
    #                                   0.5, 3, 15, 3, 5, 1.2, 0) 
    
    # Computes the magnitude and angle of the 2D vectors 
    magnitude, angle = cv.cartToPolar(flowUpSampled[..., 0], flowUpSampled[..., 1]) 
    
    # Sets image hue according to the optical flow  
    # direction 
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow 
    # magnitude (normalized) 
    mask[..., 2] = magnitude * 20
    #mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) 
    
    # Converts HSV to RGB (BGR) color representation 
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR) 
    
    #if type(prev_of) == np.ndarray:
    #    out = (rgb/2).astype(np.uint8) + (prev_of/2).astype(np.uint8)
    #else:
    #    out = rgb
    rgb = cv.resize(rgb, (720, 720))
    # Opens a new window and displays the output frame 
    cv.imshow("dense optical flow", rgb) 
    
    # Updates previous frame 
    prev_gray = gray 
    #prev_of = rgb
    #print(time.time() - lasttime)
    lasttime = time.time()
    # Frames are read by intervals of 1 millisecond. The 
    # programs breaks out of the while loop when the 
    # user presses the 'q' key 
    
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
    
  
# The following frees up resources and 
# closes all windows 
cap.release() 
cv.destroyAllWindows() 
