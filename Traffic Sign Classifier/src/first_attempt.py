# Load pickled data
import pickle
import pandas as pd
import numpy as np
import cv2
from tensorflow.contrib.layers import flatten
import tensorflow as tf
from sklearn.utils import shuffle


"""SETTING PARAMTERS"""
MEAN = 0.0
SIGMA = 0.1
EPOCHS = 28
BATCH_SIZE = 128
LEARNING_RATE = 0.0001

'''First attempt pipeline
Step 1: Pre-processing data 
        - graysclale 
        - normalize
        - reshape input data to (32,32,1)
Step 2: Define Lenet_5 model
Step 3: Train original Lenet_5 with train and validation sets
''' 

# The data is probably in order RGB
# type uint8
training_file = '../../../train.p'
validation_file= '../../../valid.p'
testing_file = '../../../test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train_original, y_train_original = train['features'], train['labels']
X_valid_original, y_valid_original = valid['features'], valid['labels']
X_test_original, y_test_original = test['features'], test['labels']


def grayscale(input_image):
    """grayscale"""
    output = []
    for i in range(len(input_image)): 
        img = cv2.cvtColor(input_image[i], cv2.COLOR_RGB2GRAY)
        output.append(img) 
    return output

def normalization(input_image):
    """normalization
      Pre-defined interval [-1,1]
      from the forum :https://discussions.udacity.com/t/accuracy-is-not-going-over-75-80/314938/22 
      some said that using the decimal 128.0 makes huge diffference
    """
    output = []
    for i in range(len(input_image)): 
        img = np.array((input_image[i] - 128.0) / (128.0), dtype=np.float32)
        output.append(img) 
    return output

def get_weights(input_shape):
    return tf.Variable(tf.truncated_normal(shape = input_shape, 
                                           mean = MEAN, stddev = SIGMA))

def get_biases(length):
     return tf.Variable(tf.zeros(length))

#NOTE: number of filter is output channel  
def convolution_layer(input_image,
                      filter_size,
                      input_channel, 
                      number_of_filters,
                      padding_choice = 'VALID'):

    shape = [filter_size, filter_size, input_channel, number_of_filters]
    weights = get_weights(input_shape = shape)
    biases = get_biases(length = number_of_filters) 
    layer = tf.nn.conv2d(input = input_image, 
                         filter = weights, 
                         strides = [1, 1, 1, 1], 
                         padding = padding_choice) + biases
    return layer


def activation_relu(input_layer):
    return tf.nn.relu(input_layer)

def max_spooling(input_layer, padding_choice):
    return tf.nn.max_pool(value = input_layer,
                          ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1],
                         padding= padding_choice)

def flatten_layer(input_layer):        
    return flatten(input_layer)

def fully_connected_layer(input_layer,
                          number_of_inputs,
                          number_of_outputs):
    
    weights = get_weights(input_shape = [number_of_inputs, number_of_outputs])
    biases = get_biases(length = number_of_outputs) 
    layer = tf.matmul(input_layer, weights) + biases
    return layer

"""STEP 1"""

def pre_process_1st_attempt(input_image):
    gray_image = grayscale(input_image)
    normalization_image =  normalization(gray_image)
    output = np.expand_dims(normalization_image, 3)
    return output

X_train_final_1 = pre_process_1st_attempt(X_train_original) 
X_valid_final_1 = pre_process_1st_attempt(X_valid_original)


"""STEP 2"""
def Lenet_5_first_attempt(input_image):    
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = convolution_layer(input_image, 5, 1, 6, 'VALID')
    conv1 = activation_relu(conv1)
    conv1 = max_spooling(conv1, 'VALID')

    # Layer 2: Convolutional. Output = 10x10x16.   
    conv2 = convolution_layer(conv1, 5, 6, 16, 'VALID')
    conv2 = tf.nn.relu(conv2)
    conv2 = max_spooling(conv2, 'VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten_layer(conv2)        
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = fully_connected_layer(fc0, 400, 120)
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = fully_connected_layer(fc1, 120, 84)
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43 
    logits = fully_connected_layer(fc2, 84, 43)   
    
    return logits

"""STEP 3"""

def evaluate(X_data, y_data):
    """Evaluation function"""
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset : offset + BATCH_SIZE], y_data[offset : offset + BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss / num_examples, total_accuracy / num_examples



if __name__ == "__main__":

	keep_prob = tf.placeholder(tf.float32)
	x = tf.placeholder(tf.float32, (None, 32, 32, 1))
	y = tf.placeholder(tf.int32, (None))
	one_hot_y = tf.one_hot(y, 43)
	logits = Lenet_5_second_attempt(x)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
	training_operation = optimizer.minimize(loss_operation)
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()

	train_loss_history = []
	valid_loss_history = []
	
	#Start running tensor flow
	with tf.Session() as sess:
	training(Lenet_5_first_attempt)
	sess.run(tf.global_variables_initializer())
	num_examples = len(X_train_final_1)

	print("Training...")
	print()
	for i in range(EPOCHS):
	    X_train, y_train = shuffle(X_train_final_1, y_train_original)
	    for offset in range(0, num_examples, BATCH_SIZE):
	        end = offset + BATCH_SIZE
	        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
	        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
	            
	    valid_loss, valid_accuracy = evaluate(X_valid_final_1, y_valid_original)
	    valid_loss_history.append(valid_loss)
	    
	    train_loss, train_accuracy = evaluate(X_train_final_1, y_train_original)
	    train_loss_history.append(train_loss)
	    
	    print("EPOCH {} ...".format(i+1))
	    print("Training Accuracy = {:.3f}".format(train_accuracy))
	    print("Validation Accuracy = {:.3f}".format(valid_accuracy))
	        
	    print("Training Loss = {:.3f}".format(train_loss))
	    print("Validation Loss = {:.3f}".format(valid_loss))
	    
	    
	saver.save(sess, './lenet')
	print("Model saved")
	loss_plot = plt.subplot(2,1,1)
	loss_plot.set_title('Loss')
	loss_plot.plot(train_loss_history, 'r', label='Training Loss')
	loss_plot.plot(valid_loss_history, 'b', label='Validation Loss')
	loss_plot.set_xlim([0, EPOCHS])
	loss_plot.legend(loc=4)
