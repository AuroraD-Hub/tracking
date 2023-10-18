#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
import socket

def main():
    ## INITIALISATION ##
    # Initialise UDP communication
    simulink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
    # Initialise device
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    profile = pipeline.start(config)

    ## SETUP ##
    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)
    # Get intrinsics parameters of the camera
    profile = pipeline.get_active_profile()
    rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = rgb_profile.get_intrinsics()
    mtx = np.matrix([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
    dist = np.array(intrinsics.coeffs)
    # Define Lucas-Kanade Optical Flow parameters
    lk_params = dict(winSize  = (21, 21),
             	criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
    # Create colors vector for drawing purpose
    color = np.random.randint(0, 255, (500, 3))
    # Discard frames at start for assessment
    assessment = 0

    ## STREAMING DATA ##
    while True:
        #timer_camera_start = time.perf_counter()
        if assessment<5:
            print("Discarding first frames for camera assessment...")    
            frames = pipeline.wait_for_frames()
            assessment = assessment+1
        elif assessment==5:
            print("Initialisation of tracking algorithm: detection of good features")  
            aligned_frames = align.process(pipeline.wait_for_frames())
            old_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
            gray_img = cv.cvtColor(old_image, cv.COLOR_RGB2GRAY)
            # Detect features
            detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            old_corners = detector.detect(gray_img)
            old_corners = np.array([x.pt for x in old_corners], dtype=np.float32)
            mask = np.zeros_like(old_image) # variables for drawing purpose
            assessment=assessment+1
        else:    
            print("Tracking camera pose") 
            aligned_frames = align.process(pipeline.wait_for_frames())
            new_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
            gray_img = cv.cvtColor(new_image, cv.COLOR_RGB2GRAY)
            # Use Optical Flow approach with RANSAC method to get feature motion
            new_corners, st, err = cv.calcOpticalFlowPyrLK(old_image, new_image, old_corners, None, **lk_params)
            st = st.reshape(st.shape[0])
            old_corners = old_corners[st==1]
            new_corners = new_corners[st==1]
            # Draw optical flow
            #for i, (new, old) in enumerate(zip(new_corners, old_corners)):
            #    a, b = new.ravel()
            #    c, d = old.ravel()
            #    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), 0, 2)
            #    frame = cv.circle(new_image, (int(a), int(b)), 5, color[i].tolist(), -1)
            #    img = cv.add(frame, mask)
            #    cv.imshow('Optical flow', img)
            # Retrive camera pose
            E, _ = cv.findEssentialMat(new_corners,old_corners, mtx)
            _, R, p, _ = cv.recoverPose(E, new_corners, old_corners, mtx)
            #print("Rotation: \n", R)
            #print("Translation: \n", p)
            # Send via socket camera pose data
            pose = np.concatenate((R,np.transpose(p)))
            #simulink.sendto(pose, ("192.168.1.200", 4021))

            # If less than 20% of previous features are tracked, perform a new detection 
            if(old_corners.shape[0] < int(old_corners.shape[0]*0.2)): 
                print("Detecting new features")    
                new_corners = detector.detect(gray_img)
                new_corners = np.array([x.pt for x in new_corners], dtype=np.float32)

            old_image = new_image
            old_corners = new_corners

        key = cv.waitKey(1)
        #timer_camera_stop = time.perf_counter()   
        #print(1/(timer_camera_stop-timer_camera_start))
        if key & 0xFF == ord('q') or key == 27:
            break

if __name__ == '__main__':
    main()