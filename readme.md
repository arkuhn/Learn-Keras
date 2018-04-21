This project is a personal effort to better explore Keras and further understand building (binary) image classifiers using convolutional neural networks.

At a high level the composed model should take in a chest x-ray and predict (hopefully accurately) whether or not pneumonia is present.



**Results (WIP):**

- First model

  - Bad testing, bad data bad everything. Initial attempt.
  - Accuracy: 65%

- Second model

  - 3 convolution layers, ReLU activiation, maxpooling, binary crossentropy, rmsprop optimizer, output through sigmoid activation

  - Improvements: Test and validation set usage, model saving, statistical analysis through sklearn, greyscale > color (they are x-rays after all), resolution increase

  - Almost 80% accuracy! Not bad!

    ![1524341866831](C:\Users\ak101\Documents\PneumoniaClassifier\img\1524341866831.png)







The network is built using the following dependencies, dataset and directory structure:

**Dependencies:**
Python 3.6, Keras, Tensorflow, Pillow, h5py, scikit-learn

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

--/VAL

​----/NORMAL

​----/PNEUMONIA