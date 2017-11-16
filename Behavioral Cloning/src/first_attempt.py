import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D,Lambda
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn


''' First atemmpt summary
1. Pre-proccesing data
	- Normalization in the range of 0 and 1
	- Mean centering the data
2. Model architectue: lenet-5
'''

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../../data/IMG/' + batch_image[0].split('/')[-1]
                center_name = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            x_train = np.array(images)
            y_train = np.array(angels)
            yield sklean.utils.shuffle(x_train, y_train)

if __name__ == '__main__':

	'''Read data'''
	rows = []
	with open('../../../data/driving_log.csv') as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	        rows.append(row)

	        
	images = []
	measurements = []
	for line in rows[1:]:
	    source_path = line[0]
	    filename = source_path.split('/')[-1]
	    current_path = '../../../data/IMG/' + filename
	    image = cv2.imread(current_path)
	    images.append(image)
	    measurement = float(line[3])
	    measurements.append(measurement)

	'''Pre-processing data '''

	augmented_images, augmented_measurements = [], []
	for image, measurement in zip(images, measurements):
	    image_flipped = np.fliplr(image)
	    measurement_flipped = measurement * (-1)
	    augmented_images.append(image)
	    augmented_images.append(image_flipped)
	    augmented_measurements.append(measurement)
	    augmented_measurements.append(measurement_flipped)

	X_train = np.array(augmented_images)
	y_train = np.array(augmented_measurements)

	model = Sequential()
	#The cameras in the simulator capture 160 pixel by 320 pixel images., after cropping, it is 90x320
	model.add(Cropping2D(cropping = ((50,20), (0,0)),input_shape=(160, 320, 3)))
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(90, 320, 3)))
		
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))      
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	'''Training: using MSE for regression'''
	model.compile(loss='mse', optimizer='adam')
	def generator(samples, batch_size = 32):
	    num_samples = len(samples)
	    while 1:
	        sklearn.utils.shuffle(samples)
	        for offset in range(0, num_samples, batch_size):
	            batch_samples = samples[offset:offset + batch_size]
	            images = []
	            angles = []
	            for batch_sample in batch_samples:
	                name = '../../../data/IMG/' + batch_sample[0].split('/')[-1]
	                center_image = cv2.imread(name)
	                center_angle = float(batch_sample[3])
	                images.append(center_image)
	                angles.append(center_angle)

	            # trim image to only see section with road
	            x_train = np.array(images)
	            y_train = np.array(angles)
	            yield sklearn.utils.shuffle(x_train, y_train)
	            
	train_samples, validation_samples = train_test_split(rows[1:], test_size = 0.2)

	train_generator = generator(train_samples, batch_size = 32)
	validation_generator = generator(validation_samples, batch_size = 32)

	history_object = model.fit_generator(train_generator,
	                                    samples_per_epoch = len(train_samples),
	                                    validation_data = validation_generator,
	                                    nb_val_samples = len(validation_samples),
	                                    nb_epoch = 25)
	model.save('model.h5')


	    
