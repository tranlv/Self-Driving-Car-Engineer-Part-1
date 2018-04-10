import pickle
import pandas as pd
import numpy as np
import random
import cv2
import glob
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle


def grayscale(input_image):
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
    return tf.Variable(tf.truncated_normal(shape = input_shape, mean = 0.0, stddev = 0.1))

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

def dropout_layer(layer, keep_prob):
    layer = tf.nn.dropout(layer, keep_prob)
    return layer

"""Pre-processing data"""

def pre_process_second_attempt(input_image):
    gray_image = grayscale(input_image)
    normalization_image =  normalization(gray_image)
    output = np.expand_dims(normalization_image, 3)
    return output

X_train_final_2 = pre_process_second_attempt(X_train_original) 
X_valid_final_2 = pre_process_second_attempt(X_valid_original)

"""Pre-processing data"""
def preprocess_data(input_image):
    gray_image = grayscale(input_image)
    output =  normalization(gray_image)
    output = np.expand_dims(output, 3)
    return output

X_train_final = preprocess_data(X_train_original) 
X_valid_final = preprocess_data(X_valid_original)
print(X_train_final[0].shape)

"""Model design"""
def Lenet_5_model(input_image):    
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x10.
    conv1 = convolution_layer(input_image, 5, 1, 10, 'VALID')
    conv1 = activation_relu(conv1)
    
    # Layer 2: Convolutional. Input = 28x28x10. Output = 24x24x20.
    conv2 = convolution_layer(conv1, 5, 10, 20, 'VALID')
    conv2 = activation_relu(conv2)
    # drop-out
    conv2 = dropout_layer(conv2, keep_prob)
    
    # Layer 3: Convolutional. Input = 24x24x20. Output = 20x20x30.
    conv3 = convolution_layer(conv2, 5, 20, 30, 'VALID')
    conv3 = activation_relu(conv3)
    # drop-out
    conv3 = dropout_layer(conv3, keep_prob)
    
    # Layer 4: Convolutional. Input = 20x20x30. Output = 16x16x40.   
    conv4 = convolution_layer(conv3, 5, 30, 40, 'VALID')
    conv4 = tf.nn.relu(conv4)
    # max_pool: output = 8x8x40
    conv4 = max_spooling(conv4, 'VALID')
    # drop-out
    conv4 = dropout_layer(conv4, keep_prob)

    # Flatten. Input = 8x8x40. Output = 2560.
    fc0   = flatten_layer(conv4)        
    
    # Layer 5: Fully Connected. Input = 2560. Output = 1280.
    fc1 = fully_connected_layer(fc0, 2560, 1280)
    fc1 = tf.nn.relu(fc1)

    # Layer 6: Fully Connected. Input = 1280. Output = 640.
    fc2 = fully_connected_layer(fc1, 1280, 640)
    fc2 = tf.nn.relu(fc2)

    # Layer 7: Fully Connected. Input = 640. Output = 320 
    fc3 = fully_connected_layer(fc2, 640, 320)
    fc3 = tf.nn.relu(fc3)
    
    # Layer 8: Fully Connected. Input = 320. Output = 160 
    fc4 = fully_connected_layer(fc3, 320, 160)
    fc4 = tf.nn.relu(fc4)
    
    # Layer 9: Fully Connected. Input = 160. Output = 80 
    fc5 = fully_connected_layer(fc4, 160, 80)
    fc5 = tf.nn.relu(fc5)
    
    # Layer 10: Fully Connected. Input = 80. Output = 43 
    logits = fully_connected_layer(fc5, 80, 43)   
    
    return logits

"""Evaluation function"""
def evaluate(X_data, y_data, my_keep_prob):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset : offset + BATCH_SIZE], y_data[offset : offset + BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, 
                                                                                   y: batch_y, 
                                                                                   keep_prob: my_keep_prob})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss / num_examples, total_accuracy / num_examples 


"""Training data"""
if __name__ == "__main__":

  ''' Pre-processing pipeline 
      - graysclale 
      - normalize
      - reshape input data to (32,32,1)
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


    """Parameters setting"""
  EPOCHS = 40
  BATCH_SIZE = 128
  LEARNING_RATE = 0.0001

  '''Training and save'''

  keep_prob = tf.placeholder(tf.float32)
  # x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
  x = tf.placeholder(tf.float32, (None, 32, 32, 1))
  y = tf.placeholder(tf.int32, (None))
  # convert to 1 hot-coded data
  one_hot_y = tf.one_hot(y, 43)

  logits = Lenet_5_model(x)

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
      sess.run(tf.global_variables_initializer())
      num_examples = len(X_train_final)

      print("Training...")
      for i in range(EPOCHS):
          X_train, y_train = shuffle(X_train_final, y_train_original)
          for offset in range(0, num_examples, BATCH_SIZE):
              end = offset + BATCH_SIZE
              batch_x, batch_y = X_train[offset:end], y_train[offset:end]
              sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

          valid_loss, valid_accuracy = evaluate(X_valid_final, y_valid_original, 1.0)
          valid_loss_history.append(valid_loss)

          train_loss, train_accuracy = evaluate(X_train_final, y_train_original, 1.0)
          train_loss_history.append(train_loss)

          print("EPOCH {} ...".format(i + 1))
          print("Training Accuracy = {:.3f}".format(train_accuracy))
          print("Validation Accuracy = {:.3f}".format(valid_accuracy))

          print("Training Loss = {:.3f}".format(train_loss))
          print("Validation Loss = {:.3f}".format(valid_loss))
