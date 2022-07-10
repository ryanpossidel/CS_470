import numpy as np
import tensorflow as tf
from cv2 import cv2
from tensorflow.keras.datasets import mnist, cifar10
import tensorflow.keras.utils as utils
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Activation, Add, Dense, Flatten, InputLayer, Conv2D, MaxPooling2D
from PIL import Image
from IPython.display import display

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
is_MNIST = False

def prepare_image_data(data, add_channel=False):
  if add_channel:
    data = np.expand_dims(data, axis=-1)
  data = data.astype("float32")
  data /= 255.0
  data -= 0.5
  data *= 2.0
  return data

x_train = prepare_image_data(x_train, add_channel=is_MNIST)
x_test = prepare_image_data(x_test, add_channel=is_MNIST)
class_cnt = 10
y_train = utils.to_categorical(y_train, num_classes=class_cnt)
y_test = utils.to_categorical(y_test, num_classes=class_cnt)

model = Sequential()
model.add(InputLayer(input_shape=(32,32,3)))

##########################
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Conv2D(64, kernel_size=3, activation="selu"))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="selu"))
########################

model.add(Dense(class_cnt, activation="softmax"))
model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

model.summary()
utils.plot_model(model, show_shapes=True)

model.fit(x_train, y_train, batch_size=32, epochs=3)
model.save('C:/Users/possi/PycharmProjects/CS470/Assignment06')

train_scores = model.evaluate(x_train, y_train)
test_scores = model.evaluate(x_test, y_test)

print("TRAIN:", train_scores)
print("TEST:", test_scores)