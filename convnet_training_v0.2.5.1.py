"""deep-graspDetection v0.2.5.1

A source code developed by Shehan Caldera (shehancaldera@gmail.com) for the purpose of training the robot grasp
detection convolutional neural networks (RGD-CNN). The training dataset is acquired from Cornell Grasp Dataset,
and it contains about 885 images and labelled grasps. The complete dataset will be divided into 3 sets:
850 for training and 35 for testing. The training images and grasps have the following format
==> (RG-D images, grasp rectangles with params : x, y, theta, h, w ).

Real-time data augmentation is employed using keras ImagePreprocessing class.

The model has a convolutional base with a similar architecture to that of ResNet-50.
"""

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
BASE_DIR = '/media/baxter/DataDisk/Cornell Grasps Dataset/rgd'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
TARGET_IMAGE_WIDTH = 224


# Function for opening images and resizing them appropriately
def open_image(instance_num, image_dir):
    if instance_num < 1000:
        base_filename = 'pcd0'
    else:
        base_filename = 'pcd'

    img_filename = os.path.join(image_dir, base_filename + str(instance_num) + "r" + "." + 'png')
    img = Image.open(img_filename)

    # Cropping image
    crop_img = img.crop(box=(80, 0, 560, 480))

    return crop_img


# Read bboxes
def open_bbox(instance_num, bbox_dir):
    if instance_num < 1000:
        base_filename = 'pcd0'
    else:
        base_filename = 'pcd'

    filename = os.path.join(bbox_dir, base_filename + str(instance_num) + "cpos" + "." + 'txt')
    with open(filename) as f:
        bbox = list(map(
            lambda coordinate: float(coordinate), f.read().strip().split()))

    return bbox[0:8]  # We only consider the first grasp to avoid averaging


# Function for converting bounding boxes to grasps
def bboxes_to_grasps(bbox):
    box = bbox
    # box = open_bbox(instance_num, bbox_dir)
    # box = float(box) / TARGET_IMAGE_WIDTH #Refract the pixel locations in unit distances
    # bbox => grasp (x, y, h, w, sin 2theta, cos 2)
    scale_x = 120
    scale_y = 80

    x = ((box[0] + box[2] + box[4] + box[6]) / 4 - 80) * 0.46
    y = ((box[1] + box[3] + box[5] + box[7]) / 4) * 0.46

    div_denominator = box[6] - box[0]  # Avoiding the ZeroDivision error in python
    if div_denominator == 0:
        tan = 0.0
    else:
        tan = (box[7] - box[1]) / (box[6] - box[0])

    '''Range of tan is confined between -89 deg : 89 deg as tan(90 deg) reaches infinity
    tan(89 deg) = 57.27 '''
    if tan < 0:
        tan = max(tan, -57.27)
    else:
        tan = min(tan, 57.27)

    theta = np.arctan(tan)
    tan = 10 * tan

    h = np.sqrt(np.power(box[2] - box[0], 2) + np.power(box[3] - box[1], 2)) * 0.46
    w = np.sqrt(np.power(box[6] - box[0], 2) + np.power(box[7] - box[1], 2)) * 0.46

    grasp = [x, y, theta, h, w]

    return grasp


def grasp_to_bbox(grasp):
    x = grasp[0]
    y = grasp[1]
    theta = grasp[2]
    h = grasp[3]
    w = grasp[4]

    bbox_x = []
    bbox_y = []

    bbox_x.append(x + ((-w / 2) * np.cos(theta)) - ((-h / 2) * np.sin(theta)))
    bbox_y.append(y + ((-h / 2) * np.cos(theta)) + ((-w / 2) * np.sin(theta)))

    bbox_x.append(x + ((-w / 2) * np.cos(theta)) - ((h / 2) * np.sin(theta)))
    bbox_y.append(y + ((h / 2) * np.cos(theta)) + ((-w / 2) * np.sin(theta)))

    bbox_x.append(x + ((w / 2) * np.cos(theta)) - ((h / 2) * np.sin(theta)))
    bbox_y.append(y + ((h / 2) * np.cos(theta)) + ((w / 2) * np.sin(theta)))

    bbox_x.append(x + ((w / 2) * np.cos(theta)) - ((-h / 2) * np.sin(theta)))
    bbox_y.append(y + ((-h / 2) * np.cos(theta)) + ((w / 2) * np.sin(theta)))

    bbox_x.append(x + ((-w / 2) * np.cos(theta)) - ((-h / 2) * np.sin(theta)))
    bbox_y.append(y + ((-h / 2) * np.cos(theta)) + ((-w / 2) * np.sin(theta)))

    bbox_x = np.asarray(bbox_x, dtype='float32')
    bbox_y = np.asarray(bbox_y, dtype='float32')

    return bbox_x, bbox_y


def rmse(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# Defining the network architecture
def network_model():
    # Import ResNet50 with imagenet weights
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH, 3))

    # Initiate the sequential model
    model = Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5))  # Last layer does not have an activation as this is a vector regression

    # Freezing the weights of the VGG16 on imagenet weights
    conv_base.trainable = False

    # Verifying the model architecture by inspecting its summary
    model.summary()

    # Compile with Adam optimizer
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mse', metrics=['acc'])

    return model


if __name__ == '__main__':
    print keras.__version__
    K.clear_session()

    # Reading the complete CGD set for training data+targets, images are center cropped on the grasp position
    train_data = []
    train_targets = []

    print(time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime()) + ': Reading the raw CGD data...')
    for i in range(100, 950):
        bboxes = open_bbox(i, TRAIN_DIR)

        for bbox_i in range(0, len(bboxes), 8):
            bbox = bboxes[bbox_i:8 + bbox_i]

            grasp = bboxes_to_grasps(bbox)
            img = open_image(i, TRAIN_DIR)

            train_targets.append(grasp)

            resized_img = img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH))
            img_array = np.asarray(resized_img, dtype='float32')
            train_data.append(img_array)
            print(time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime()) + ': Read data instance: ' + str(i) + 'x' + str(
                bbox_i / 8))

    for i in range(1000, 1035):
        bboxes = open_bbox(i, TEST_DIR)

        for bbox_i in range(0, len(bboxes), 8):
            bbox = bboxes[bbox_i:8 + bbox_i]

            grasp = bboxes_to_grasps(bbox)
            img = open_image(i, TEST_DIR)

            train_targets.append(grasp)

            resized_img = img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH))
            img_array = np.asarray(resized_img, dtype='float32')
            train_data.append(img_array)
            print(time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime()) + ': Read data instance: ' + str(i) + 'x' + str(
                bbox_i / 8))

    train_data = np.asarray(train_data, dtype='float32')
    train_targets = np.asarray(train_targets, dtype='float32')
    print(time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime()) + " train_data shape: " + str(train_data.shape))
    print(time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime()) + " train_targets shape: " + str(train_targets.shape))

    # Check 6 random samples from the data
    fig = plt.figure(figsize=(12, 7), dpi=100, facecolor='w', edgecolor='k')
    i_plt = 0
    for i_t in range(0, len(train_data), int(len(train_data) / 6)):
        test_img = Image.fromarray(np.uint8(train_data[i_t,]))
        i_plt = i_plt + 1
        if i_plt>6:
            i_plt = 6
        plt.subplot(2,3, i_plt)
        plt.imshow(test_img)
        (bx_pred, by_pred) = grasp_to_bbox(train_targets[i_t,])
        plt.plot(bx_pred[0:2], by_pred[0:2], 'r-', bx_pred[1:3], by_pred[1:3], 'g-', bx_pred[2:4], by_pred[2:4],
                 'r-', bx_pred[3:5], by_pred[3:5], 'g-')
    plt.show()

    # Reading training data into training, validation, and testing subsets

    print(time.strftime("%d.%m.%Y_%H.%M.%S",
                        time.localtime()) + ': Creating 3 subsets (train, valid, test) from the CGD data...')

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []

    x_valid = train_data[0:100]
    y_valid = train_targets[0:100]

    x_test = train_data[100:150]
    y_test = train_targets[100:150]

    x_train = train_data[150:len(train_data)]
    y_train = train_targets[150:len(train_data)]

    x_train = np.asarray(x_train, dtype='float32')
    y_train = np.asarray(y_train, dtype='float32')
    x_valid = np.asarray(x_valid, dtype='float32')
    y_valid = np.asarray(y_valid, dtype='float32')
    x_test = np.asarray(x_test, dtype='float32')
    y_test = np.asarray(y_test, dtype='float32')

    print('Confirm if x_train has NaN:' + str(np.isnan(x_train).any()),
          'Confirm if x_train is finite:' + str(np.isfinite(x_train).any()))
    if np.isnan(x_train).any() or np.isinf(x_train).any():
        x_train = np.nan_to_num(x_train)

    print('Confirm if y_train has NaN:' + str(np.isnan(y_train).any()),
          'Confirm if y_train is finite:' + str(np.isfinite(y_train).any()))
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        y_train = np.nan_to_num(y_train)

    print('Confirm if x_valid has NaN:' + str(np.isnan(x_valid).any()),
          'Confirm if x_valid is finite:' + str(np.isfinite(x_valid).any()))
    if np.isnan(x_valid).any() or np.isinf(x_valid).any():
        x_valid = np.nan_to_num(x_valid)

    print('Confirm if y_valid has NaN:' + str(np.isnan(y_valid).any()),
          'Confirm if y_valid is finite:' + str(np.isfinite(y_valid).any()))
    if np.isnan(y_valid).any() or np.isinf(y_valid).any():
        y_valid = np.nan_to_num(y_valid)

    print('Confirm if x_test has NaN:' + str(np.isnan(x_test).any()),
          'Confirm if x_test is finite:' + str(np.isfinite(x_test).any()))
    if np.isnan(x_test).any() or np.isinf(x_test).any():
        x_test = np.nan_to_num(x_test)

    print('Confirm if y_test has NaN:' + str(np.isnan(y_test).any()),
          'Confirm if y_test is finite:' + str(np.isfinite(y_test).any()))
    if np.isnan(y_test).any() or np.isinf(y_test).any():
        y_test = np.nan_to_num(y_test)

    print(time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime()) + ': Reading the CGD data: DONE')

    print('x_train shape: ' + str(x_train.shape), 'y_train shape: ' + str(y_train.shape))
    print('x_valid shape: ' + str(x_valid.shape), 'y_valid shape: ' + str(y_valid.shape))
    print('x_test shape: ' + str(x_test.shape), 'y_test shape: ' + str(y_test.shape))

    train_datagen = ImageDataGenerator(rotation_range=0.01,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2)

    valid_datagen = ImageDataGenerator()

    model = network_model()
    epochs = 30
    batch_size = 128

    # Create callbacks for Tensorboard
    callback_list = [keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=0.1,
        patience=2, ),

        keras.callbacks.TensorBoard(
            log_dir="logs/{}".format(time.time()))
    ]

    history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=3600,  # 13 * len(x_train)/batch_size,
                                  epochs=epochs,
                                  callbacks=callback_list,
                                  validation_data=valid_datagen.flow(x_valid, y_valid),
                                  verbose=1)

    model.save('/home/baxter/PycharmProjects/deep-graspDetection/trained_models/grasp_det_v0.2.5.1__26.07.2018.h5')



