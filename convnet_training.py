"""A source code developed by Shehan Caldera (shehancaldera@gmail.com) for the purpose of training the robot grasp
detection convolutional neural network (RGD-CNN) on the augmented Cornell Grasp data. The training data has the
following format ==> (RGB image, grasp rectangle params : x, y, theta, h, w ).

@SAMPLE_NMBR is 9000 from the augmented dataset. Real-time data augmentation is employed using keras ImagePreprocessing
class. The model has a convolutional base with the architecture of Inception-ResNet"""


# Define the imports
import keras
import os
import numpy as np
import math
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from PIL import Image
from PIL import ImageOps
import scipy as sp
from skimage.util import crop
import matplotlib.pyplot as plt
import numpy as np
import pypcd
import time
from keras import backend as K
from keras.applications import Xception, VGG16, ResNet50, InceptionResNetV2
from keras import models
from keras import layers
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from shapely.geometry import Polygon
import math

# Global Variables
BASE_DIR = '/media/baxter/DataDisk/Cornell Grasps Dataset/rgb_aug'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
TARGET_IMAGE_WIDTH = 224
SAMPLE_NUMBR = 9000  # Number of sub-samples to select from the data: image, label


# Import keras and print version
def import_keras():
    keras.__version__
    K.clear_session() # Init resource clean-up & refresh


# Function for reading CGD data into network variables
def read_cgd_data(sample_number):
    x_train = []
    y_train = []
    print(time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime()) + ': Reading the CGD data...')

    for i in range(0, 49359, int(49359 / sample_number)):
        img_filename = os.path.join(TRAIN_DIR, "img" + str(i) + "." + 'png')
        img = Image.open(img_filename)
        # resized_img = img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH), resample=Image.NEAREST)

        img_array = np.asarray(img, dtype='float32')
        x_train.append(img_array)

        grasp_filename = os.path.join(TRAIN_DIR, "grasp" + str(i) + "." + 'txt')
        grasp = np.loadtxt(grasp_filename, dtype='float32')
        y_train.append(grasp)
        print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Datapoint # loaded:' + str(i))

    x_train = np.asarray(x_train, dtype='float32')
    y_train = np.asarray(y_train, dtype='float32')
    return x_train, y_train


# --Define custom metric for jaccard index
def grasp_to_bbox(grasp):
    # Order of input: x, y, theta, h, w
    bbox = []
    x = grasp[0]
    y = grasp[1]
    theta = grasp[2]
    h = grasp[3]
    w = grasp[4]

    bbox.append(x + ((-w / 2) * math.cos(theta)) - ((-h / 2) * math.sin(theta)))
    bbox.append(y + ((-h / 2) * math.cos(theta)) + ((-w / 2) * math.sin(theta)))
    bbox.append(x + ((-w / 2) * math.cos(theta)) - ((h / 2) * math.sin(theta)))
    bbox.append(y + ((h / 2) * math.cos(theta)) + ((-w / 2) * math.sin(theta)))
    bbox.append(x + ((w / 2) * math.cos(theta)) - ((h / 2) * math.sin(theta)))
    bbox.append(y + ((h / 2) * math.cos(theta)) + ((w / 2) * math.sin(theta)))
    bbox.append(x + ((w / 2) * math.cos(theta)) - ((-h / 2) * math.sin(theta)))
    bbox.append(y + ((-h / 2) * math.cos(theta)) + ((w / 2) * math.sin(theta)))

    bbox = np.asarray(bbox, dtype='float32')

    return bbox


def jaccrad(y_true, y_pred):
    rec_true = grasp_to_bbox(y_true)
    rec_pred = grasp_to_bbox(y_pred)

    poly_true = Polygon([(rec_true[0:2]),
                         (rec_true[2:4]),
                         (rec_true[4:6]),
                         (rec_true[6:8])])
    poly_pred = Polygon([(rec_pred[0:2]),
                         (rec_pred[2:4]),
                         (rec_pred[4:6]),
                         (rec_pred[6:8])])

    iou = poly_true.intersection(poly_pred).area / (
                poly_true.area + poly_pred.area - poly_true.intersection(poly_pred).area)

    return iou


def angle_diff(y_true, y_pred):
    return K.abs((y_pred[2] - y_true[2]) * 57.3)  # 57.3 deg == 1 rad


# Defining the network architecture
def network_model():
    # Import ResNet50 with imagenet weights
    conv_base = InceptionResNetV2(weights='imagenet',
                                  include_top=False,
                                  input_shape=(TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH, 3))

    # Initiate the sequential model
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='relu'))  # Last layer does not have an activation as this is a vector regression

    # Freezing the weights of the VGG16 on imagenet weights
    conv_base.trainable = False

    # Verifying the model architecture by inspecting its summary
    model.summary()

    # Compile with Adam optimizer
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss='mse', metrics=[angle_diff])

    return model


# Save the model with the timestamp
def save_model(model):
    print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Saving the model...')
    t_stamp = time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())
    model.save('/home/baxter/PycharmProjects/rgd-cnn_models/grasp_detector_nightly_' + t_stamp + '.h5')
    print(t_stamp + ': SAVED: /home/baxter/PycharmProjects/rgd-cnn_models/grasp_detector_nightly_' + t_stamp + '.h5')


# Training with realtime data augmentation with ImagePreprocessing class
def network_training():
    datagen = ImageDataGenerator(rotation_range=5,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 validation_split=0.2)

    # Read the data
    (x_train, y_train) = read_cgd_data(SAMPLE_NUMBR)

    # Model definition & raining params
    model = network_model()
    epochs = 100
    batch_size = 32

    # Create callbacks for Tensorboard
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir='logged_data',
            histogram_freq=1,
            embeddings_freq=1
        )
    ]

    # Fits the model on batches with real-time data augmentation
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size, subset="training"),
                                  steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                                  #callbacks=callbacks,
                                  verbose=1)

    return model


if __name__ == '__main__':
    import_keras()
    trained_model = network_training()
    save_model(trained_model)
