from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from models import *

#Data params
img_width, img_height = 320, 240
train_data_dir = 'data/TRAIN'
test_data_dir = 'data/TEST'
classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']


#Training params
epochs = 50
batch_size = 16
input_shape = (img_width, img_height, 3)


#Model
model = first_model(input_shape)


#Train and test data augmentors
train_datagen = ImageDataGenerator (
    rescale=1./255
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)


#Generators from TEST and TRAIN
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

model.fit_generator(
    train_generator,
    steps_per_epoch= 2000 // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps= 800 // batch_size
)

model.save_weights('first_try.h5')