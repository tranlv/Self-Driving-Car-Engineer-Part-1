import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D, Lambda, Dropout
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from keras.layers.pooling import MaxPooling2D


def append_data(col, images, measurement, steering_measurements):
    current_path = image_path + '/' + col.strip()
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    steering_measurements.append(measurement)
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = measurement * (-1)
    steering_measurements.append(measurement)
          

def images_and_measurements(sample):
    images = []
    steering_measurements = []
    for line in sample[0:]:
        measurement = float(line[3])
        #center
        col_center = line[0]
        append_data(col_center, images, measurement, steering_measurements)
        #left
        col_left = line[1]
        append_data(col_left, images, measurement + 0.2, steering_measurements)
        #right
        col_right = line[2]
        append_data(col_right, images, measurement - 0.2, steering_measurements)
    return images, steering_measurements

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]           
            images = []
            measurements = []
            for image, measurement in batch_samples:
                images.append(image)   
                measurements.append(measurement)
            # trim image to only see section with road
            x_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(x_train, y_train)

if __name__ == '__main__':

	'''Read data'''
	image_path = '../../../data'
	driving_log_path = '../../../data/driving_log.csv'

	rows = []
	with open(driving_log_path) as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	        rows.append(row)

	X_total, y_total = images_and_measurements(rows[1:])

	        
	'''Pre-processing data '''
	model = Sequential()
	#The cameras in the simulator capture 160 pixel by 320 pixel images., after cropping, it is 66x200
	model.add(Cropping2D(cropping = ((74,20), (60,60)),input_shape=(160, 320, 3)))

	'''Model architecture'''
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))
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
	model.compile(loss='mse', optimizer='adam')

	'''Training'''
	print('Training model')            
	samples = list(zip(X_total, y_total))          
	train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
	train_generator = generator(train_samples, batch_size = 32)
	validation_generator = generator(validation_samples, batch_size = 32)


	history_object = model.fit_generator(train_generator,
	                                    samples_per_epoch = len(train_samples),
	                                    validation_data = validation_generator,
	                                    nb_val_samples = len(validation_samples),
	                                    nb_epoch = 20, 
	                                    verbose=1)
	print('Endding training, starting to save model')
	model.save('../model.h5')

	    
