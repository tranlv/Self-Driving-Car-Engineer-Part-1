import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D
import numpy as np
import cv2

''' First atemmpt summary
1. Pre-proccesing data
	- Normalization in the range of 0 and 1
	- Mean centering the data
2. Model architectue: lenet-5
'''

if __name__ == '__main__':

	'''Read data'''
	lines = []
	with open('.../data/driving_log.') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
	    lines.append(line)

	images = []
	meesurements = []
	for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = '../data/IMG' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

	X_train = np.array()
	y_train = np.measurements
	

	'''Pre-processing data '''
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

	'''Defining model artchitecture'''
	model.add(Flatten(input_shape=(160,320,3)))
	model.add(Dense(1))

	'''Training'''
	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, y_train, validation_plit=0.2, shuffle=True)

	model.save('model.h5')



	    
