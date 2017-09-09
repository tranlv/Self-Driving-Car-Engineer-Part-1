import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')
print('This image is: ', type(image), 'with dimensions: ', image.shape)

ysize = image.shape[0]
xsize = image.share[1]
region_select = np.copy(image)

left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

XX, YY = np.meshgrid(np.arrage(0, xsize), np.arrage(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))



#color_select = np.copy(image)
#red_threshold = 0
#red_threshold = 0
#green_threshold = 0
#blue_threshold = 0
#rgb_threshold = [red_threshold, green_threshold, blue_threshold]

#thresholds = (image[:,:,0] < rgb_threshold[0])\
#             | (image[:,:,1] < rgb_threshold[1]) \
#             | (image[:,:,2] < rgb_threshold[2])
#color_select[thresholds] = [0,0,0]

# Color pixels red which are inside the region of interest
region_select[region_thresholds] = [255, 0, 0]

plt.imshow



