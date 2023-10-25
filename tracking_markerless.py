#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
import socket

class RealSenseD435i:
    def __init__(self):
        # Initialise device
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        self.profile = self.pipeline.start(config)
        # Discard frames at start for assessment
        self.assessment = 0

    def get_camera_setup(self):
        # Create an align object
        align_to = rs.stream.color
        align = rs.align(align_to)
        # Get intrinsics parameters of the camera
        self.profile = self.pipeline.get_active_profile()
        rgb_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intrinsics = rgb_profile.get_intrinsics()
        mtx = np.matrix([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
        dist = np.array(intrinsics.coeffs)

        return mtx, dist, align

    def get_camera_ready(self):
        while self.assessment<5:
            print("Discarding first frames for camera assessment...")    
            frames = self.pipeline.wait_for_frames()
            self.assessment = self.assessment+1

        return self.pipeline

class VisualOdometry:
    def __init__(self, pipeline, mtx, dist, align):
        # Set camera info
        self.pipe = pipeline
        self.K = mtx
        self.dist = dist
        self.align = align
        # Define detector
        self.detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        # Define Lucas-Kanade Optical Flow parameters
        self.lk_params = dict(winSize  = (21, 21),
             	              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        # Define structures needed
        self.old_image = []
        self.old_corners = []
        #self.mask = [] # variables for drawing purpose
        #self.color = np.random.randint(0, 255, (500, 3))

    def process_image(self, aligned_frames):
        image_dist = np.asanyarray(aligned_frames.get_color_frame().get_data())
        image_undist = cv.undistort(image_dist, self.K, self.dist)
        image_gray = cv.cvtColor(image_undist, cv.COLOR_RGB2GRAY)

        return image_gray, image_undist

    def detect_new_features(self, gray_img):
        corners = self.detector.detect(gray_img)
        corners = np.array([x.pt for x in corners], dtype=np.float32)

        return corners

    def get_pose(self, string):
        if string=="init":
            print("Initialisation of tracking algorithm: detection of good features")  
            aligned_frames = self.align.process(self.pipe.wait_for_frames())
            self.old_image, mask_img = self.process_image(aligned_frames)
            #self.mask = np.zeros_like(mask_img)
            # Detect features
            self.old_corners = self.detect_new_features(self.old_image)

        elif string=="track":
            print("Tracking camera pose") 
            aligned_frames = self.align.process(self.pipe.wait_for_frames())
            new_image, rgb_img = self.process_image(aligned_frames)
            # Use Optical Flow approach with RANSAC method to get feature motion
            new_corners, st, _ = cv.calcOpticalFlowPyrLK(self.old_image, new_image, self.old_corners, None, **self.lk_params)
            st = st.reshape(st.shape[0])
            self.old_corners = self.old_corners[st==1]
            new_corners = new_corners[st==1]

            # Draw optical flow
            #for i, (new, old) in enumerate(zip(new_corners, self.old_corners)):
            #    a, b = new.ravel()
            #    c, d = old.ravel()
            #    self.mask = cv.line(self.mask, (int(a), int(b)), (int(c), int(d)), 0, 2)
            #    frame = cv.circle(rgb_img, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            #    img = cv.add(frame, self.mask)
            #    cv.imshow('Optical flow', img)

            # Retrive camera pose
            E, _ = cv.findEssentialMat(new_corners,self.old_corners, self.K)
            _, R, p, _ = cv.recoverPose(E, new_corners, self.old_corners, self.K)
            #print("Rotation: \n", R)
            #print("Translation: \n", p)
            pose = np.concatenate((R,np.transpose(p)),axis=0)
            #print("Pose: \n", pose)

            # If less than 20% of previous features are tracked, perform a new detection 
            if(self.old_corners.shape[0] < int(self.old_corners.shape[0]*0.2)): 
                print("Detecting new features")    
                new_corners = self.detect_new_features(new_image)

            self.old_image = new_image
            self.old_corners = new_corners

            return pose

        else:
            print("Tracking phase not recognized.")


def main():
    # Initialise UDP communication
    simulink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Create instance of RealSenseD435i VisualOdometry classes and initialise them
    d435i = RealSenseD435i()
    mtx, dist, align = d435i.get_camera_setup()
    pipeline = d435i.get_camera_ready()
    tracker = VisualOdometry(pipeline,mtx,dist,align)
    tracker.get_pose("init") 
    
    while True:
        # Start tracking camera pose
        pose = tracker.get_pose("track")

        # Send via socket camera pose data
        pose = np.reshape(pose, (1,12))
        simulink.sendto(pose, ("localhost", 4021)) #192.168.1.200
        
        key = cv.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

if __name__ == '__main__':
    main()