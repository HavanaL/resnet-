import os
import cv2
import random
import numpy as np
from imutils import paths
from keras.utils import np_utils
from keras.models import Model
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten

data = []
labels = []
path = 'D:\\photo_test'
imagepaths = sorted(list(paths.list_images(path)))
random.seed()
random.shuffle(imagepaths)
for i in imagepaths:
    image = cv2.imread(i)
    image = cv2.resize(image, (197, 197))
    image = img_to_array(image)
    data.append(image)

    label = int(i.split(os.path.sep)[-2][9:])
    labels.append(label)
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)
x_train=data
x_test=data
y_train=labels
y_test=labels
y_train = np_utils.to_categorical(y_train, num_classes=997)
y_test = np_utils.to_categorical(y_test, num_classes=997)

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(197, 197, 3))
x=resnet_model.output
x=Flatten()(x)
y=Dense(997,activation='softmax')(x)
model=Model(inputs=resnet_model.input,outputs=y)
for layer in resnet_model.layers:
    layer.trainable=False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')
# x = model.fit_generator(datagen.flow(x_train, y_train), validation_data=(x_test, y_test), epochs=50, verbose=2)
x=model.fit(x_train,y_train,epochs=50,shuffle=True,verbose=2,validation_split=0.25)
model.save('resnet50 model.h5')

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), x.history['acc'], label='train_acc')
plt.plot(np.arange(0, 50), x.history['val_acc'], label='val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='upper right')
plt.savefig('resnetacc.png')
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), x.history['loss'], label='train_loss')
plt.plot(np.arange(0, 50), x.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig('resnetloss.png')
plt.show()