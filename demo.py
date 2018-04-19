from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


image_augmentor = ImageDataGenerator (
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

base_image = load_img('data/preview/source/test.jpg')
x = img_to_array(base_image) # (3, 150, 150)
x = x.reshape((1,) + x.shape) # (1, 3, 150, 150)


i = 0
for batch in image_augmentor.flow(x, batch_size=1, save_to_dir='data/preview',
                                save_prefix='generated', save_format='.jpeg'):
    i += 1
    if i >= 5:
        break    

