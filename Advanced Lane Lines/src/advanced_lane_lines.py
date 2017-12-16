import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import glob
import os

'''Directory'''
CAMERA_IMAGE_DIRECTORY = '../camera_cal/'
OUTPUT_DIRRECTORY = '../output_images/'
TEST_IMAGES_DIRECTORY = '../test_images/'

'''Read images directories'''
chessboard_images = glob.glob(CAMERA_IMAGE_DIRECTORY + 'calibration*.jpg')
test_images = glob.glob(TEST_IMAGES_DIRECTORY + '*.jpg')

#Prepare obj poinrts
objp = np.zeros((9 * 6, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
obj_points = [] #3d points in real world space
img_points = [] #2d points in image points

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i, fname in enumerate(chessboard_images):
    img = cv2.imread(fname)
    image_name=os.path.split(fname)[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret == True:
        obj_points.append(objp)
        
        # this step to refine image points was taken from:
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img_points.append(corners2)
        
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        corners_write_name = OUTPUT_DIRRECTORY + 'chessboard_corners/' +  'corners_found_'+ str(9) + '_' + str(6) + '_' + image_name
        cv2.imwrite(corners_write_name, img)

# Calcualte the undistortion matrix to calculate matrix and distance coefficient
img = cv2.imread(chessboard_images[0])
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)