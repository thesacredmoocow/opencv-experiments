import cv2
import sys
import numpy as np
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
    params = cv2.TrackerDaSiamRPN_Params()
    params.model = "dasiamrpn_model.onnx"
    params.kernel_cls1 = "dasiamrpn_kernel_cls1.onnx"
    params.kernel_r1 = "dasiamrpn_kernel_r1.onnx"
    params.backend = cv2.dnn.DNN_BACKEND_CUDA
    params.target = cv2.dnn.DNN_TARGET_CUDA
    tracker = cv2.TrackerDaSiamRPN_create(params)
    #t = cv2.TrackerDaSiamRPN_create()
 
    #tracker = cv2.legacy.TrackerBoosting_create()
    #tracker = cv2.TrackerMIL_create()
    #tracker = cv2.TrackerKCF_create()
    #tracker = cv2.legacy.TrackerTLD_create()
    #tracker = cv2.legacy.TrackerMedianFlow_create()
    #tracker = cv2.TrackerGOTURN_create()
    #tracker = cv2.legacy.TrackerMOSSE_create()
    #tracker = t()
 
    # Read video
    video = cv2.VideoCapture("videos/fight4.avi")
    video.set(cv2.CAP_PROP_POS_FRAMES, 12150)
    #video = cv2.VideoCapture(0) # for using CAM
 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()
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
    
    frame = cv2.warpPerspective(frame,M,(720, 720),flags=cv2.INTER_LINEAR)
    #cv2.imshow("warped", out)
    # Define an initial bounding box
    bbox = (226, 387, 90, 102)
 
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    print(bbox)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    outputVideo = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 90, (720,720))
 
    while True:
        # Read a new frame
        ok, frame = video.read()
        frame = cv2.warpPerspective(frame,M,(720, 720),flags=cv2.INTER_LINEAR)
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
        outputVideo.write(frame)
 
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
            outputVideo.release()
            bbox = cv2.selectROI(frame, False)
 
            # Initialize tracker with first frame and bounding box
            #tracker = t()
            ok = tracker.init(frame, bbox)
    video.release()
    cv2.destroyAllWindows()