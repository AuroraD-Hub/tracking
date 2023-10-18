#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
import socket

def initialize_camera_imu(depth_unit=15/(2**16 - 1), l=640, h=480, fps=60):
    pipeline = rs.pipeline()

    config = rs.config()

    #config.enable_stream(rs.stream.depth, l, h, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, l, h, rs.format.bgr8, fps)
    #config.enable_stream(rs.stream.accel)
    #config.enable_stream(rs.stream.gyro)

    # Start streaming
    profile = pipeline.start(config)
    # Getting the depth sensor's depth scale
    #sensor_dep = profile.get_device().first_depth_sensor()
    #sensor_dep.set_option(rs.option.enable_auto_exposure, True) #set auto exposure
    #sensor_dep.set_option(rs.option.depth_units, depth_unit) #set desired depth_unit

    return pipeline

def main():
    ## INITIALISATION ##
    # Initialise UDP communication
    simulink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
    # Initialise device
    #depth_unit = 20/(2**16-1) #20 m 
    pipeline = initialize_camera_imu()

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
    # Define needed tracking structures
    old_image = []
    new_image = []
    old_corners = []
    new_corners = []
    # Define Lucas-Kanade detector parameters
    lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
    # Discard frames at start for assessment
    assessment = 0

    ## STREAMING DATA ##
    while True:
        #timer_camera_start = time.perf_counter()
        if assessment<5:
            print("Discarding first frames for camera assessment...")    
            frames = pipeline.wait_for_frames()
            assessment = assessment+1
        elif assessment=6:
            print("Initialisation of tracking algorithm: detection of good features.")  
            aligned_frames = align.process(pipeline.wait_for_frames())
            old_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
            gray_img = cv.cvtColor(old_image, cv.COLOR_RGB2GRAY)
            # Detect features
            old_corners = cv.goodFeaturesToTrack(gray_img)
        else:    
            print("Tracking camera pose.") 
            aligned_frames = align.process(pipeline.wait_for_frames())
            new_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
            gray_img = cv.cvtColor(new_image, cv.COLOR_RGB2GRAY)
            # Use Optical Flow approach with RANSAC method to get feature motion
            new_corners, st, err = cv.calcOpticalFlowPyrLK(old_image, new_image, old_corners, None, lk_params)  #shape: [k,2] [k,1] [k,1]
            # Retrive camera pose
            E, _ = cv.findEssentialMat(new_corners,old_corners, mtx)
	        _, R, p, _ = cv.recoverPose(E, new_corners, old_corners, mtx)
            # Send via socket camera pose data
            pose = np.matrix(R,np.transpose(p))
            simulink.sendto(pose, ("192.168.1.200", 4021))

            old_image = new_image
            old_corners = new_corners

        key = cv.waitKey(1)
        #timer_camera_stop = time.perf_counter()   
        #print(1/(timer_camera_stop-timer_camera_start))
        if key & 0xFF == ord('q') or key == 27:
            break

if __name__ == '__main__':
    main()