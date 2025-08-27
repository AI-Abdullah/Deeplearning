import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
#plt.matshow(x_train[0])
#print(x_train.shape)
x_train = x_train / 255
x_test = x_test / 255
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)
#print(x_test_flattened.shape)
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)
print('this is the result of the model ',model.evaluate(x_test_flattened, y_test))
y_predicted = model.predict(x_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
#print(y_predicted_labels[:5])
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)
print('this is result by Using 100 hidden layer',model.evaluate(x_test_flattened,y_test))