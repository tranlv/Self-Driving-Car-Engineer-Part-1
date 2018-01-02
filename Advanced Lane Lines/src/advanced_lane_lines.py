import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import glob
import os
from moviepy.editor import VideoFileClip

def undistort_image(img):           
    result = cv2.undistort(img, mtx, dist, None, mtx)
    return result

def absolute_sobel_threshold(img, orient='x', thresh_min = 0, thresh_max = 255):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        
    scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sobel_binary

def magnitude_threshold(gray_img, sobel_kernel = 3, mag_thresh = (0, 255)): 
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(grad_mag)/255 
    grad_mag = (grad_mag/scale_factor).astype(np.uint8) 
    mag_binary = np.zeros_like(grad_mag)
    mag_binary[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
    return mag_binary

def direction_threshold(gray_img, sobel_kernel = 3, dir_thresh = (0, np.pi/2)):
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(abs_grad_dir)
    dir_binary[(abs_grad_dir >= dir_thresh[0]) & (abs_grad_dir <= dir_thresh[1])] = 1
    return dir_binary 

def warper(image, source, destination):
    h,w = img.shape[:2]
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    inverse_matrix = cv2.getPerspectiveTransform(destination, source)
    warped = cv2.warpPerspective(image, transform_matrix , (w,h))
    return warped, transform_matrix, inverse_matrix

def source(img_size):
    src = np.float32(
        [
             [(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]]
            
        ])
    
    return src

def destination(img_size):
    dest = np.float32(
        [
             [(img_size[0] / 4), 0],
             [(img_size[0] * 3 / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]]
        ])
    
    return dest

def pipeline(img):
    
    img_size = (img.shape[1], img.shape[0])
    src = source(img_size)
    dest = destination(img_size)
    warp_img, M, inverse_M = warper(img, src, dest) 

    gray_img =  cv2.cvtColor(warp_img, cv2.COLOR_RGB2GRAY)
    grad_x_binary = absolute_sobel_threshold(gray_img, orient='x', thresh_min=30, thresh_max=250)
    grad_y_binary = absolute_sobel_threshold(gray_img, orient='y', thresh_min=20, thresh_max=100)
    mag_binary = magnitude_threshold(gray_img, sobel_kernel = 9, mag_thresh = (120, 255))
    dir_binary = direction_threshold(gray_img, sobel_kernel = 15, dir_thresh = (0.7, 1.1))

    combined_threshold = np.zeros_like(dir_binary)
    combined_threshold[((grad_x_binary == 1) & (grad_y_binary == 1)) & (mag_binary == 1) ] = 1
    
    hsv = cv2.cvtColor(warp_img, cv2.COLOR_RGB2HSV)
    hsv_v_channel = hsv[:,:,2]
    hsv_v_thresh = (210, 255)
    hsv_v_binary = np.zeros_like(hsv_v_channel)
    hsv_v_binary[(hsv_v_channel > hsv_v_thresh[0]) & (hsv_v_channel <= hsv_v_thresh[1])] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((combined_threshold, combined_threshold, hsv_v_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gray_img)
    combined_binary[(hsv_v_binary == 1) | (combined_threshold == 1)] = 1
    
    return combined_binary,  inverse_M

def calculate_curvature_pixel_space(fit, ploty):
    y_value = np.max(ploty)
    nominator =  (1 + (2 * fit[0] * y_value + fit[1])**2)**1.5
    denominator = np.absolute(2 * fit[0])
    return (nominator / denominator)

def calculate_curvature_world_space(fit,  ploty):
    # Define conversions in x and y from pixels space to meters
    y_in_m_per_pix = 30/720 # meters per pixel in y dimension
    x_in_m_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    fit_x = fit[0]*ploty**2 + fit[1] * ploty + fit[2]
    
    fit_word_space = np.polyfit(ploty * y_in_m_per_pix, fit_x* x_in_m_per_pix, 2)
    result = calculate_curvature_pixel_space(fit_word_space,  ploty)
    return result

def drawing(undistorted_img, binary_warped, left_fit, right_fit, inverse_M, img_size): 
    # Create an image to draw the lines on
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inverse_M, img_size) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

    left_curvature_world_space = calculate_curvature_world_space(left_fit,  ploty)
    right_curvature_world_space = calculate_curvature_world_space(right_fit,  ploty)
    
    curvature_string = "Left Radius of curvature: %.2f m" % left_curvature_world_space
    cv2.putText(result,curvature_string , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
    
    curvature_string = "Right Radius of curvature: %.2f m" % right_curvature_world_space
    cv2.putText(result,curvature_string , (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)

    
    # compute the offset from the center\

    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ym_per_pix = 30/720

    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_m = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    xMax = img.shape[1]*xm_per_pix
    yMax = img.shape[0]*ym_per_pix
    vehicleCenter = xMax / 2
    lineLeft = left_fit_m[0]*yMax**2 + left_fit_m[1]*yMax + left_fit_m[2]
    lineRight = right_fit_m[0]*yMax**2 + right_fit_m[1]*yMax + right_fit_m[2]
    lineMiddle = lineLeft + (lineRight - lineLeft)/2
    diffFromVehicle = lineMiddle - vehicleCenter
    offset_string = "Center offset: %.2f m" % diffFromVehicle
    cv2.putText(result, offset_string, (100, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
    return result

def sliding_window_polyfit(binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    ## fiding  left and right point
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/ nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        win_xleft_low = leftx_base - margin
        win_xleft_high = leftx_base + margin
        
        win_xright_low = rightx_base - margin
        win_xright_high = rightx_base + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & 
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & 
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)


    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    polyfit_left = np.polyfit(lefty, leftx, 2)
    polyfit_right = np.polyfit(righty, rightx, 2)
    
    return polyfit_left, polyfit_right


def polyfit_using_previous_fit(binary_warped, prev_left_fit, prev_right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] - margin))
                      & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2]- margin))
                       & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        polyfit_left = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        polyfit_right = np.polyfit(righty, rightx, 2)
    
    return polyfit_left, polyfit_right

def process_image(image):
    # global variables to store the polynomial coefficients of the line detected in the last frame
    global polyfit_left
    global polyfit_right
    
    img_size = (image.shape[1], image.shape[0])

    #1. Camera correction and distortion correction
    undistorted_img = undistort_image(image)
    
    #2. Applying perspective transformation, thresholded color and gradient filter
    binary_warped, inverse_M = pipeline(undistorted_img)
    
    #3. Detect lanes and return fit curves
    if (polyfit_left is not None) and (polyfit_right is not None):
        polyfit_left, polyfit_right =  polyfit_using_previous_fit(binary_warped, polyfit_left, polyfit_right)   
    else:
        polyfit_left, polyfit_right = sliding_window_polyfit(binary_warped)
    
    result = drawing(undistorted_img, binary_warped, polyfit_left, polyfit_right, inverse_M, img_size)
    return result


if __name__ == '__main__':

    '''Read images directories'''
    chessboard_images = glob.glob('../camera_cal/*.jpg')
    test_images = glob.glob('../test_images/*.jpg')

    #Prepare obj poinrts
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    obj_points = [] #3d points in real world space
    img_points = [] #2d points in image points

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    fig, axs = plt.subplots(5,4, figsize=(16, 11))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

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
            
            # Draw chessboard with corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            axs[i].axis('off')
            axs[i].imshow(img)

    polyfit_left = None
    polyfit_right = None

    clip1 = VideoFileClip('../test_video/project_video.mp4')
    video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

    video_output = '../output_videos/project_video.mp4'