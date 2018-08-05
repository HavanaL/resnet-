import os
import cv2
import random
import numpy as np
from imutils import paths
from keras.optimizers import SGD,adam
from keras.utils import np_utils
from keras.applications import VGG16
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten

data = []
labels = []
path = 'D:\\photo_test'
imagepaths = sorted(list(paths.list_images(path)))
random.seed()
random.shuffle(imagepaths)
for i in imagepaths:
    image = cv2.imread(i)
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image)
    data.append(image)

    label = int(i.split(os.path.sep)[-2][9:])
    labels.append(label)
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)
# (x_train, x_test, y_train, y_test) = train_test_split(data, labels, random_state=42)
x_train=data
x_test=data
y_train=labels
y_test=labels

y_train = np_utils.to_categorical(y_train, num_classes=997)
y_test = np_utils.to_categorical(y_test, num_classes=997)

base_model=VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
x=base_model.output
x=Flatten()(x)
x=Dense(500,activation='relu')(x)
y=Dense(997,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=y)
for layer in base_model.layers:
    layer.trainable=False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')

x = model.fit_generator(datagen.flow(x_train, y_train), validation_data=(x_test, y_test), epochs=50, verbose=2)
# x=model.fit(x_train,y_train,epochs=50,shuffle=True,verbose=2,validation_split=0.25)
model.save('vgg16 model.h5')

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), x.history['acc'], label='train_acc')
plt.plot(np.arange(0, 50), x.history['val_acc'], label='val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='upper right')
plt.savefig('vgg16acc.png')
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), x.history['loss'], label='train_loss')
plt.plot(np.arange(0, 50), x.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig('vgg16loss.png')
plt.show()