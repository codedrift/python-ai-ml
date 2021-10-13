import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imutils import paths
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)

def load(paths):
    data = list()
    labels = list()
    for (i, imgpath) in enumerate(paths):
        
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        
        image = np.array(im_gray).flatten()

        # split out label from path
        label = imgpath.split(os.path.sep)[-2]

        # scale the image to [0, 1] and add to list
        data.append(image / 255)
        labels.append(label)
        # print(f"Loaded {imgpath} with label {label}. {i + 1}/{len(paths)}")
    return data, labels

# folder of folders: {label}/[files]
train_images_path = "./input/kaggle/trainingSet/trainingSet"

print(f'Loading images from {train_images_path}')

image_paths = list(paths.list_images(train_images_path))

print(f"Found {len(image_paths)} images")

image_list, label_list = load(image_paths)

# check current shape
print(f"Shape of image list: {tf.shape(image_list).numpy()}")
print(f"Shape of label list: {tf.shape(label_list).numpy()}")


# reshape images from 784 to 28x28
reshaped = tf.reshape(image_list, [len(image_list), 28, 28])

print(f"Reshaped image list {reshaped.shape}")

# inspect first image to check that loading and reshaping worked
# plt.figure()
# plt.imshow(reshaped[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    reshaped.numpy(), label_list, test_size=0.1, random_state=42
)

print(f'X_train:{len(X_train)} X_test:{len(X_test)} y_train:{len(y_train)} y_test:{len(y_test)} ')

# group a linear stack of layers
model = tf.keras.models.Sequential(layers=[
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

print(f'shape of X_train {tf.shape(X_train).numpy()}')

# print(X_train[:1])

# plt.figure()
# plt.imshow(X_train[:1][0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

predictions = model(X_train[:1]).numpy()

print(predictions)

max = tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# print(y_train[:1])

loss = loss_fn(tf.strings.to_number(y_train[:1]).numpy(), predictions).numpy()

print(f'max: {max} loss: {loss}')

print(f'compiling model')
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print(f'fitting model')
model.fit(X_train, tf.strings.to_number(y_train).numpy(), epochs=5)

print(f'evaluating model')
model.evaluate(X_test,  tf.strings.to_number(y_test).numpy(), verbose=2)


test_image = X_test[:1][0]

plt.figure()
plt.imshow(test_image)
plt.colorbar()
plt.grid(False)
plt.show()


print(f'create probability model')
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(f'predict test model')
preds = probability_model(X_test[:25])

# print(preds2.numpy())
# print(np.argmax(preds2.numpy()))


for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(hspace=1,wspace=1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[:25][i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(preds.numpy()[i]))
plt.show()
