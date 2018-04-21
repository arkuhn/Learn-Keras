from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from models import *
import time

#Data params
img_width, img_height = 250, 250
channels = 1
train_data_dir = 'data/TRAIN'
test_data_dir = 'data/TEST'

#Training params
epochs = 50
batch_size = 30
input_shape = (img_width, img_height, channels)


#Model
model = first_model(input_shape)


#Train and test data augmentors
train_datagen = ImageDataGenerator (
   horizontal_flip=True,
)
test_datagen = ImageDataGenerator(
   horizontal_flip=True,
)


#Generators from TEST and TRAIN
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
)

runtime = str(int(time.time()))
print('Now running: ' + runtime)
model.fit_generator(
    train_generator,
    steps_per_epoch= 2000 // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps= 800 // batch_size
)

model.save('model-' + runtime + '.h5')