from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from models import *
import time

#Data params
img_width, img_height = 500, 300
channels = 1
train_data_dir = 'data/TRAIN'
val_data_dir = 'data/VAL'

#Training params
epochs = 50
batch_size = 15
input_shape = (img_width, img_height, channels)

#Model
model = first_model(input_shape)

#Train and val data augmentors
train_datagen = ImageDataGenerator (
    rescale=1./255,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest'
)

#Generators for TRAIN and VAL
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
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
    validation_data=val_generator,
    validation_steps= 800 // batch_size
)

model.save('model-' + runtime + '.h5')