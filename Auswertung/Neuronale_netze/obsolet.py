images = []
labels = []


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


def create_dataset():
    for filename in os.listdir('images'):
        art = declare_type(filename)
        img = cv2.imread(os.path.join('images', filename))
        img_resized = cv2.resize(img, (331, 331), interpolation=cv2.INTER_CUBIC)
        images.append(img_resized)
        labels.append(art)


#create_dataset()

# """Lade Daten und bereite vor"""
# images = np.array(images)
# labels = np.array(labels)
# labels = to_categorical(labels, num_classes=6)
#
# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
# train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=1)
#
# print("Shape Trainingsdaten: {}".format(train_images.shape))
# print("Shape Testdaten: {}".format(test_images.shape))
# print("Shape Validationdaten: {}".format(val_images.shape))
# print("Dimension Bild Nr. 5: {}".format(train_images[5].shape))
# print("Label zu Bild Nr. 5: {}".format(train_labels[5]))
#
#
# inputshape = train_images.shape[1:4]
# print(inputshape)