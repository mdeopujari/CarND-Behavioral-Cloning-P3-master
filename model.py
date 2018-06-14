import csv
import cv2
import numpy as np
import pandas as pd
print("A!")
import matplotlib
import matplotlib.pyplot as plt
print("B!")
## Read training images file names and steering measurements data
lines = []

with open('./mydata_recovery_slowturns/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = filename   
    # Load image as RGB
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images.append(image)
    # Load steering measurement
    measurement = float(line[3]) 
    measurements.append(measurement)

print(len(images))
print(len(measurements))

## Code to see spread of data
'''df = pd.Series(measurements)
df.plot(kind='hist')
plt.show()'''

## Code to augment existing dataset by flipping
'''aug_images = []
aug_measurements = []
for image,measurement in zip(images,measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    flipped_measurement = (measurement) *(-1)
    aug_images.append(flipped_image)
    aug_measurements.append(flipped_measurement)

df = pd.Series(aug_measurements)
df.plot(kind='hist')'''

X_train = np.array(images)
print(X_train.shape)
y_train = np.array(measurements)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

## Traditional LeNet5 architecture
model = Sequential()
# pre-process data and normalize
model.add(Lambda(lambda x:x / 255.0 - 0.5,input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.1))
model.add(Convolution2D(16,5,5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(120))
#model.add(Dropout(0.1))
model.add(Dense(84))
#model.add(Dropout(0.1))
model.add(Dense(1))

# Using Adam optimizer
model.compile(loss='mse', optimizer='adam')

# Using a training:validation ratio of 70:30 in dataset
history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=5,verbose=1)

model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
#plt.savefig('LossChart.png')