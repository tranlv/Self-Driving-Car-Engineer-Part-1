#importing some  packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def draw_lines(img, lines, color=[255, 0, 0], thickness = 10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_x = []
    left_y = []
    
    right_x = []
    right_y = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            # 1. we dont need horizontal; i.e slope =0  they are noise
            # 2. slope >0 is right ; slope <0 is left
            slope = ((y2-y1)/(x2-x1))
            if (slope < 0):
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)
            elif (slope > 0):
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)
                
    if (len(left_x) > 0  and len(left_y) > 0):
        # find coefficient
        coeff_left = np.polyfit(left_x, left_y, 1)
        # construct y =xa +b
        func_left = np.poly1d(coeff_left)
        x1L = int(func_left(0))
        x2L = int(func_left(460))
        cv2.line(img, (0, x1L), (460, x2L), color, thickness)

    
    if (len(right_x) > 0  and len(right_y) > 0):  
        # find coefficient
        coeff_right = np.polyfit(right_x, right_y, 1)
        # construct y =xa +b
        func_right = np.poly1d(coeff_right)
        x1R = int(func_right(500))
        x2R = int(func_right(img.shape[1]))
        cv2.line(img, (500, x1R), (img.shape[1], x2R), color, thickness)
                
# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)
    img = grayscale(image)
    # further smoothing to blur image for better result
    img = gaussian_blur(img, kernel_size = 3)
    img = canny(img, low_threshold = 80, high_threshold = 240)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(460, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    img = region_of_interest(img, vertices)

    # Hough transform
    line_image = hough_lines(img, rho = 2, theta= np.pi/180, threshold = 50, min_line_len = 40, max_line_gap = 20)

    # Draw the lines on the edge image
    final = weighted_img(line_image, image, α = 0.8, β = 1., λ = 0.)
    return final

if __name__ == '__main__':
	white_line_output = '../test_videos_output/solidWhiteRight.mp4'
	clip1 = VideoFileClip('../test_videos/solidWhiteRight.mp4')
	white_line_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	
	yellow_output = '../test_videos_output/solidYellowLeft.mp4'
	clip2 = VideoFileClip('../test_videos/solidYellowLeft.mp4')
	yellow_clip = clip2.fl_image(process_image)

	challenge_output = '../test_videos_output/challenge.mp4'
	clip3 = VideoFileClip('../test_videos/challenge.mp4')
	challenge_clip = clip3.fl_image(process_image)
