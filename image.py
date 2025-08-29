import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train_scale = x_train / 255
x_test_scale = x_test / 255
y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes=10)

def plot_sample(index):
    plt.figure(figsize =(10,1))
    plt.imshow(x_train[index])
    plt.show()

def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='softmax') # Softmax for multi-class
    ])
    model.compile(
        optimizer='SGD',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model # This line needs to be properly indented

# Train the model using the get_model() function
with tf.device('/CPU:0'):
    cpu_model = get_model()
    cpu_model.fit(x_train_scale, y_train_categorical, epochs=10)
    print('Lets measure our training time on a GPU')
    with tf.device('/GPU:0'):
        cpu_model = get_model()
        cpu_model.fit(x_train_scale,y_train_categorical,epochs=10)