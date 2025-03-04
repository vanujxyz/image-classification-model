import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#loading the cifar-10 dataset
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

#normalising the image
x_train,x_test = x_train / 255.0 , x_test/255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class_names = ['airpane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i].argmax()])
    plt.axis('off')

plt.show()