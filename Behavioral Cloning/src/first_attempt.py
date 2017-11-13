import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D,Lambda
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

	'''Pre-processing data 
	a. Data augmentation: Flipping images
	b. Normalization
	c. Mean centering
	'''

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
	#The cameras in the simulator capture 160 pixel by 320 pixel images.
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
	#model.add(Cropping2D(cropping = ((50,20), (0,0))))

	'''Defining model artchitecture'''
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
	model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, verbose = 1)

	model.save('model.h5')

	train_generator = generator(train_samples, batch_size=32)
	validation_generator = generator(validation_samples, batch_size=32)
	ch, row, col = 3, 80, 320

	history_object = model.fit_generator(train_generator,
	                                    sample_per_epoch = len(train_samples),
	                                    validation_data = validation_generator,
	                                    nb_val_Samples = len(validation_examples),
	                                    nb_epoch=5)
	print(history_object.history.key())

	###plot the training and validation loss for each epoch
	plt.plot(history(history_objet.history['loss']))
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()



	    
