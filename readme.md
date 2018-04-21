This project is a personal effort to better explore Keras and further understand building (binary) image classifiers using convolutional neural networks.

At a high level the composed model should take in a chest x-ray and predict (hopefully accurately) whether or not pneumonia is present.

The network is built using the following dependencies, dataset and directory structure:

**Dependencies:**
Python 3.6, Keras, Tensorflow, Pillow, h5py

**Data:**
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

**Directories:**

/data

--/TRAIN

​----/NORMAL

​----/PNEUMONIA

--/TEST

​----/NORMAL

​----/PNEUMONIA