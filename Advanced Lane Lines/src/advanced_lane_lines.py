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

'''Define Parameters'''
KERNEL_SIZE = 3
OFFSET = 100
NX = 9
NY = 6
THRESHOLD_MIN = 0
THRESHOLD_MAX = 255
MAGNITUDE_THRESHOLD = (0, 255)
DIRECTION_THRESHOLD = (0, np.pi/2)
SATURATION_THRESHOLD = (170, 255)

#Prepare obj poinrts
objp = np.zeros((NX * NY, 3), np.float32)
objp[:,:2] = np.mgrid[0:NX, 0:NY].T.reshape(-1,2)
obj_points = [] #3d points in real world space
img_points = [] #2d points in image points

images = glob.glob(CAMERA_IMAGE_DIRECTORY + 'calibration*.jpg')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    image_name=os.path.split(fname)[1]
    ret, corners = cv2.findChessboardCorners(img, (NX, NY), None)
    if ret == True:
        obj_points.append(objp)
        img_points.append(corners)
        cv2.drawChessboardCorners(img, (NX, NY), corners, ret)
        corners_write_name = OUTPUT_DIRRECTORY + 'chessboard_corners/' +  'corners_found_'+ str(NX) + '_' + str(NY) + '_' + image_name
        cv2.imwrite(corners_write_name, img)
