import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from imgaug import augmenters as iaa
import imgaug as ia


all_PLT_MLT_multi = {}
all_PLT_MLT = {}

with (open("../Auswertung_nd2/Auswertung/dictionary_matrix.pickle", "rb")) as openfile:
    while True:
        try:
            all_PLT_MLT.update(pickle.load(openfile))
        except EOFError:
            break

# with (open("../Auswertung_nd2/Auswertung/train_data.pickle", "rb")) as openfile:
#     while True:
#         try:
#             all_PLT_MLT_multi.update(pickle.load(openfile))
#         except EOFError:
#             break


train_images, train_labels = [], []
types = list(all_PLT_MLT["PLT"].keys())
for i in range(len(types)):
    train_images.append(all_PLT_MLT["PLT"][types[i]]["matrix"])
    if "Fichte" in types[i]:
        train_labels.append(0)
    elif "Ahorn" in types[i]:
        train_labels.append(1)
    elif "Buche" in types[i]:
        train_labels.append(2)
    elif "Eiche" in types[i]:
        train_labels.append(3)
    elif "Kiefer" in types[i]:
        train_labels.append(4)
    elif "Laerche" in types[i]:
        train_labels.append(5)

test_images, test_labels = [], []

test_types = types[0:16]
for i in range(len(test_types)):
    test_images.append((all_PLT_MLT["PLT"][types[i]]["matrix"]))
    if "Fichte" in test_types[i]:
        test_labels.append(0)
    elif "Ahorn" in test_types[i]:
        test_labels.append(1)
    elif "Buche" in test_types[i]:
        test_labels.append(2)
    elif "Eiche" in test_types[i]:
        test_labels.append(3)
    elif "Kiefer" in test_types[i]:
        test_labels.append(4)
    elif "Laerche" in test_types[i]:
        test_labels.append(5)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
print(train_images.shape)
print(test_images.shape)
arten = ["Fichte", "Ahorn", "Buche", "Eiche", "Kiefer", "Laerche"]

train_images = train_images / 255.0
test_images = test_images / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1004, 1008)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(arten))
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=16, epochs=10, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])

print(np.argmax(predictions[0]))

print(test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(arten[predicted_label],
                                         100*np.max(predictions_array),
                                         arten[true_label]),
                                         color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(len(arten)))
    plt.yticks([])
    thisplot = plt.bar(range(len(arten)), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
plt.savefig('uno.png')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
plt.savefig('mehr.png')

img = test_images[1]
print(img.shape)

img = (np.expand_dims(img, 0))
print(img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(len(arten)), arten, rotation=45)

print(np.argmax(predictions_single[0]))
