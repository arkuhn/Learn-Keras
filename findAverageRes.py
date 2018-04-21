from PIL import Image
import sys, os

classes = ['NORMAL', 'PNEUMONIA']
test_data_dir = 'data/TEST'


widths = []
heights = [] 
count = 0

for label in classes:
    labelpath = test_data_dir + '/' + label
    for img in (os.listdir(labelpath + '/')):
        im = Image.open(labelpath + '/' + img)
        print ('Width: ' + str(im.size[0]))
        print ('Height: ' + str(im.size[1]))
        widths.append((im.size[0]))
        heights.append((im.size[1]))
        count += 1


average_height = sum(heights) / count
print('Average height: ' + str(average_height))
average_width = sum(widths) / count
print('Average width: ' + str(average_width))



