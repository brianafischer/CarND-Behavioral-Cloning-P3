import csv
import os.path
import cv2
import numpy as np
import tensorflow as tf

print('Working directory: ' + os.getcwd())

# ---------------------------------------------------------
## Data Set
# ---------------------------------------------------------
# Import the CSV index for the Data Set
lines = []
missing = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    print(csvfile)
    for line in reader:
        # if the file exists, then add
        source_path = line[0]                               # this should be the center image
        filename = source_path.split('/')[-1]               # take the filename only and not the original path
        current_path = 'data/IMG/' + filename
        if os.path.exists(current_path):
            lines.append(line)
        else:
            missing.append(current_path)

            
# Summarize Data Set
print("Lines removed from data set: ", len(missing))
print("Number of lines: ", len(lines))


# Build the Measurement Data Set
images, measurements = [], []
for line in lines:
 
    # There are 3 images (center, left, right)
    for i in range(3):
        # Build the filename paths for each of the images, using the filename (split) along with the expected relative path
        image_filename = 'data/IMG/'+line[i].split('/')[-1]
        if not os.path.exists(image_filename):
            print('Ruhoh')
      
        # Note: The camera images are read by the OpenCV library in a different format (BGR) than what drive.py uses
        #       We convert each to the expected format (RGB) before training the model
        image = cv2.imread(image_filename)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Steering angle
    measurement = float(line[3])
    
    # Center camera steering angle (no correction needed)
    measurements.append(measurement)

    # Selecting a value for correction of the left and right cameras
    correction = 0.16

    # Add correction for the left camera
    measurements.append(measurement + correction)

    # Add correction for the right camera
    measurements.append(measurement - correction)

    
# ---------------------------------------------------------
## Data Preprocessing
# ---------------------------------------------------------
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append( np.fliplr(image) )
    augmented_measurements.append( measurement * -1 )

x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# ---------------------------------------------------------
## Network Architecture
# 
#  nVidia 9 layer
#  Source: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
# ---------------------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf as ktf

model = Sequential()

# layer 1
# Normalization [0 to 255] to [-1 to 1]
model.add( Lambda(lambda x: (x / 127.5) - 1, input_shape=(160, 320, 3)) )

# Cropping 70 pixels from the top removes above the horizon
# Cropping 25 pixel from the bottom removes the hood of the vehicle
model.add( Cropping2D(cropping=((70,25), (1,1))) )

# Layers 2 through 4
# 5x5 Convolutional (24, 36, 48)
model.add( Convolution2D(24,5,5, subsample=(2,2), activation="relu", border_mode='valid') )
model.add( Convolution2D(36,5,5, subsample=(2,2), activation="relu", border_mode='valid') )
model.add( Convolution2D(48,5,5, subsample=(2,2), activation="relu", border_mode='valid') )

# Layers 6 and 6
# 3x3 Convolutional (64, 64)
model.add( Convolution2D(64,3,3, activation="relu") )
model.add( Convolution2D(64,3,3, activation="relu") )

# Flatten
model.add(Flatten())

# Layers 7 through 9
# Fully-connected Layers (100, 50, 10)
model.add( Dropout(0.5)       )
model.add( Dense(100)         )
model.add( Activation('relu') )
model.add( Dropout(0.5)       )
model.add( Dense(50)          )
model.add( Activation('relu') )
model.add( Dense(10)          )
model.add( Activation('relu') )
model.add( Dense(1)           )

# Optimizer, split 80% training 20% validation, save model as HDF5
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5')