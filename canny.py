import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

image = mpimg.imread('exit-ramp.jpg')
plt.imshow(image)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
edges = cv2.Canny(gray, low_threshold, high_threshold)