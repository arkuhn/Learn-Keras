from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from keras.models import load_model
import sys

try:
    model_name = sys.argv[1]
    print('Using model name: ' + model_name)
except (IndexError):
    print('ERROR: Test.py takes model file name as command line parameter')
    sys.exit(0)

#Model
model = load_model(model_name)
batch_size = 20

#Test data
test_data_dir = 'data/TEST'
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
    class_mode='binary'
)

samples = 10

predictions = model.predict_generator(
    test_generator
)

scores = model.evaluate_generator(
    test_generator 
)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


'''
target_names = ['NORMAL', 'PNEUMONIA']
report = classification_report(
    test_values, 
    predicted_values, 
    target_names=target_names
)
print(report)
'''