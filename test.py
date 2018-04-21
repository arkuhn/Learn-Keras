from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.metrics import classification_report
from keras.models import load_model
import numpy as np
import sys, os

try:
    model_name = sys.argv[1]
    print('Using model name: ' + model_name)
except (IndexError):
    print('ERROR: Test.py takes model file name as command line parameter')
    sys.exit(0)

#Model
model = load_model(model_name)
batch_size = 16

#Test data
classes = ['NORMAL', 'PNEUMONIA']
test_data_dir = 'data/TEST'
num_of_normal = len( os.listdir(test_data_dir + '/' + classes[0]))
num_of_pneumonia = len(os.listdir(test_data_dir + '/' + classes[1]))
total_images = num_of_normal + num_of_pneumonia
img_width, img_height = 500, 300

test_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest'
)

#Generators from TEST
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

steps = int(total_images / batch_size)
predicted_values = model.predict_generator(
    test_generator,
    steps = steps, 
    verbose = 1
)
predicted_values = [ int(x[0]) for x in predicted_values ]
print(predicted_values)

scores = model.evaluate_generator(
    test_generator 
)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

normal_actual_values = [0] * num_of_normal
pneumonia_actual_values = [1] * num_of_pneumonia
actual_values = normal_actual_values + pneumonia_actual_values

report = classification_report(
    actual_values, 
    predicted_values,
    target_names=classes
)
print(report)

