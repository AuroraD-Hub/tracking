#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import math
import time
import socket

#Rmtx_prev = np.empty((3,3))
#Tvec_prev = np.empty((3,1))

def initialize_camera_imu(depth_unit=15/(2**16 - 1), l=640, h=480, fps=30):
    pipeline = rs.pipeline()

    config = rs.config()

    config.enable_stream(rs.stream.depth, l, h, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, l, h, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    # Start streaming
    profile = pipeline.start(config)
    # Getting the depth sensor's depth scale
    sensor_dep = profile.get_device().first_depth_sensor()
    sensor_dep.set_option(rs.option.enable_auto_exposure, True) #set auto exposure
    sensor_dep.set_option(rs.option.depth_units, depth_unit) #set desired depth_unit

    return pipeline

def find_position(frames, detector, points, mtx, dist):
        #global Rmtx_prev, Tvec_prev

        aligned_depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # Detect the marker with Aruco
        corners, ids, _ = detector.detectMarkers(gray_img)
        
        # Find coordinates of the central pixel of the marker
        if len(corners)>0:
            print("Detected")
            _, Rvec, Tvec = cv.solvePnP(points, corners[0], mtx, dist)
            Rmtx, _ = cv.Rodrigues(Rvec)
            #Rmtx_prev = Rmtx
            #Tvec_prev = Tvec
            #cv.aruco.drawDetectedMarkers(image, corners, ids)
            #cv.drawFrameAxes(image, mtx, dist, Rvec, Tvec, 0.1)
        else: 
            print("Missed")
            #Rmtx = Rmtx_prev
            #Tvec = Tvec_prev
            
        #cv.imshow("Markers", image)

        return Rmtx, Tvec


def main():
    ## INITIALISATION ##
    # Initialise UDP communication
    simulink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
    # Initialise device
    depth_unit = 20/(2**16-1) #20 m 
    pipeline = initialize_camera_imu(depth_unit)

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
    # Create Aruco detector
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters =  cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    marker_len = 0.065 # meter
    points = np.array([[-marker_len/2, 0, marker_len/2], [marker_len/2, 0, marker_len/2], [marker_len/2, 0, -marker_len/2], [-marker_len/2, 0, -marker_len/2]]) # marker is 6.5x6.5 cm
    # Discard frames at start for assessment
    assessment = 0

    ## STREAMING DATA ##
    while True:
        timer_camera_start = time.perf_counter()
        if assessment<5:
            print("Discarding first frames for assessment...")    
            frames = pipeline.wait_for_frames()
            assessment = assessment+1
        else:    
            frames = pipeline.wait_for_frames()

            # Orientation tracking
            accel = frames[2].as_motion_frame().get_motion_data()
            gyro = frames[3].as_motion_frame().get_motion_data()
            IMU = np.array([accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z]).reshape(1,6)
            simulink.sendto(IMU, ("localhost", 4031))
        
            # Position tracking
            aligned_frames = align.process(frames)
            Rmtx, Tvec = find_position(aligned_frames, detector, points, mtx, dist)
            Pose_marker = np.concatenate((Rmtx.reshape(1,9), Tvec.reshape(1,3)), axis=1)
            simulink.sendto(Pose_marker, ("localhost", 4021))

        key = cv.waitKey(1)
        timer_camera_stop = time.perf_counter()   
        #print(1/(timer_camera_stop-timer_camera_start))
        if key & 0xFF == ord('q') or key == 27:
            break

if __name__ == '__main__':
    main()