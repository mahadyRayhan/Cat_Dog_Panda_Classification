# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from xlib.smallvggnet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import random
import pickle
import cv2
import os


class MixupImageDataGenerator():
    def __init__(self, generator, trainX, trainY, batch_size, alpha=0.2):

        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        self.generator1 = generator.flow(trainX,
                                        trainY,
                                        batch_size=batch_size)

        # Second iterator yielding tuples of (x, y)
        self.generator2 = generator.flow(trainX,
                                        trainY,
                                        batch_size=batch_size)

        # Number of images across all classes in image directory.
        self.n = len(trainX) // self.batch_size

    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.

        Returns:
            int -- steps per epoch.
        """

        return self.n // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.

        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # random sample the lambda value from beta distribution.
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)

        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
    help="full path to output trained model with name")
ap.add_argument("-l", "--label-bin", required=True,
    help="full path to output label binarizer with name")
ap.add_argument("-p", "--plot", required=True,
    help="full path to output accuracy/loss plot with name")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, resize it to 64x64 pixels (the required input
    # spatial dimensions of SmallVGGNet), and store the image in the
    # data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)
 
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
 
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

BS = 5
validation_split = 0.25
img_height = 64
img_width = 64
EPOCHS = 4
INIT_LR = 0.01

# Optional additional image augmentation with ImageDataGenerator.
input_imgen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0,
    shear_range=0.05,
    zoom_range=0,
    brightness_range=(1, 1.3),
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split)

# Create training and validation generator.
train_generator = MixupImageDataGenerator(generator=input_imgen,
                                          trainX=trainX,
                                          trainY = trainY,
                                          batch_size=BS)

print('training steps: ', len(trainX) // BS)

# sample_x, sample_y = next(train_generator)
# for i in range(BS):
#     data = sample_x[i]
#     data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
#     img = Image.fromarray(data, 'RGB')
#     img.show()
# print(sample_y)

# Build a Keras image classification model as usual.
model = SmallVGGNet.build(width=64, height=64, depth=3,
    classes=3)

print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])


# train the network
H = model.fit_generator(train_generator,
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // EPOCHS,
    epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))
 
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
 
# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

