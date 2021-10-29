import os
import random
import numpy as np
import shutil
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from PIL import ImageFile
from pathlib import Path
import PIL
import math


"""Before starting the CNN check your BASE_DIR"""

"""Check Tensorflow Version and GPU"""

print(tf.__version__)
warnings.filterwarnings("ignore")

if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please ensure you have installed Tensorflow correctly')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
print(tf.config.list_physical_devices('GPU'))

ImageFile.LOAD_TRUNCATED_IMAGES = True


BASE_DIR = 'images/405nm_x10_100pct/'
class_names = ["Ahorn", "Buche", "Eiche", "Fichte", "Kiefer", "Laerche"]


def relocate_images():
    tf.random.set_seed(1)

    if not os.path.isdir(BASE_DIR + 'train/'):
        for name in class_names:
            os.makedirs(BASE_DIR + 'train/' + name)
            os.makedirs(BASE_DIR + 'val/' + name)
            os.makedirs(BASE_DIR + 'test/' + name)

    orig_folders = ["Ahorn/", "Buche/", "Eiche/", "Fichte", "Kiefer/", "Laerche/"]
    for folder_idx, folder in enumerate(orig_folders):
        files = os.listdir(BASE_DIR + folder)
        number_of_images = len([name for name in files])
        n_train = int((number_of_images * 0.6) + 0.5)
        n_valid = int((number_of_images*0.25) + 0.5)
        n_test = number_of_images - n_train - n_valid
        print(number_of_images, n_train, n_valid, n_test)
        for idx, file in enumerate(files):
            file_name = BASE_DIR + folder + file
            if idx < n_train:
                shutil.move(file_name, BASE_DIR + "train/" + class_names[folder_idx])
            elif idx < n_train + n_valid:
                shutil.move(file_name, BASE_DIR + "val/" + class_names[folder_idx])
            else:
                shutil.move(file_name, BASE_DIR + "test/" + class_names[folder_idx])


#relocate_images()

path = Path(BASE_DIR).rglob("*.png")
for img_p in path:
    try:
        img = PIL.Image.open(img_p)
    except PIL.UnidentifiedImageError:
            print(img_p)


def declare_type(type):
    art = ""
    if 0 == type:
        art = "Ahorn"
    elif 1 == type:
        art = "Buche"
    elif 2 == type:
        art = "Eiche"
    elif 3 == type:
        art = "Fichte"
    elif 4 == type:
        art = "Kiefer"
    elif 5 == type:
        art = "Laerche"

    return art

shape_a = 1004
shape_b = 1008

#Hyperparameter
batchsize = 10
num_classes = len(class_names)
epochs = 25


preprocess_input = tf.keras.applications.mobilenet.preprocess_input

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, dtype=tf.dtypes.float32)
valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input, dtype=tf.dtypes.float32)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input, dtype=tf.dtypes.float32)

train_batches = train_gen.flow_from_directory(
    BASE_DIR + 'train',
    target_size=(shape_a, shape_b),
    color_mode="rgb",
    classes=class_names,
    class_mode="categorical",
    batch_size=batchsize,
    shuffle=True,
)

val_batches = valid_gen.flow_from_directory(
    BASE_DIR + 'val',
    target_size=(shape_a, shape_b),
    color_mode="rgb",
    classes=class_names,
    class_mode="categorical",
    batch_size=batchsize,
)

test_batches = test_gen.flow_from_directory(
    BASE_DIR + 'test',
    target_size=(shape_a, shape_b),
    color_mode="rgb",
    classes=class_names,
    class_mode="categorical",
    batch_size=1,
    shuffle=False,
)

# fig = plt.figure()
# axes = []
# x, y = train_batches.next()
# for i in range(0, 8):
#     title = y[i]
#     title = np.argmax(title)
#     title = declare_type(title)
#     image = x[i]
#     axes.append(fig.add_subplot(2, 4, i + 1))
#     axes[-1].set_title(title)
#     plt.imshow(image)
# fig.tight_layout()
# plt.show()

test_images = []

for t in range(0, test_batches.n):
    x, y = test_batches.__getitem__(t)
    image = np.reshape(x, (shape_a, shape_b, 3))
    test_images.append(image)

test_labels = test_batches.labels


def build_net(input_shape):

    base_model = tf.keras.applications.MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

    model.summary()

    return model


model = build_net((shape_a, shape_b, 3))


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Model compilation completed')


LOGDIR = "logs"
my_tensorboard = TensorBoard(log_dir=LOGDIR, histogram_freq=0, write_graph=True, write_images=True)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=True)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='../models/best.hdf5', monitor='val_loss', save_best_only=True, mode='auto')


"""Trainiere Modell"""

print(len(model.trainable_variables))


history = model.fit(train_batches, validation_data=val_batches,
                    callbacks=[early_stopping],
                    epochs=epochs,
                    verbose=True)


test_batches.reset()
predictions = model.predict(test_batches)
print(model.evaluate(x=test_batches))

print("Fitting the model completed")

plt.plot(history.history['loss'], label='MAE (training_batches)')
plt.plot(history.history['val_loss'], label='MAE (validation_data)')
plt.title('MAE')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'


def predict(img_name):
    img = image.load_img(img_name, target_size=(shape_a, shape_b))
    img = image.img_to_array(img)
    plt.imshow(img.astype(np.uint8))
    plt.show()
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
    prediction = model.predict(img.reshape(1, shape_a, shape_b, 3))
    output = np.argmax(prediction)
    print(class_names[output])


def plot_image(i, predictions_array, true_label,
               img):  # taking index and 3 arrays viz. prediction array, true label array and image array
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    predicted_label = np.argmax(predictions_array)
    if predicted_label == 0:
        predicted_label = 1
    if true_label == 0:
        true_label = 1

    thisplot = plt.bar(range(6), predicted_label, color='seashell')
    plt.ylim([0, 1])

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')


i = random.randint(0, test_batches.n)
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)

plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()


num_rows = 4
num_cols = 4
num_images = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)

    plot_image(i, predictions, test_labels, test_images)

plt.show()


Y_pred = model.predict_generator(test_batches)
y_pred = np.argmax(Y_pred, axis=1)


cm = confusion_matrix(test_batches.classes, y_pred)
plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), class_names=class_names, show_normed=False)
fig = plt.figure()
plt.tight_layout()
plt.show()
fig.savefig('cm.png')
