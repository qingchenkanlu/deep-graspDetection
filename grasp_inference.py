"""A source code developed by Shehan Caldera (shehancaldera@gmail.com) for the purpose of predicting the robotic grasp
rectangles using the trained RGD-CNN on the augmented Cornell Grasp data. The inference output has the
following format ==> (RGB image, grasp rectangle params : x, y, theta, h, w ).

The model has a convolutional base with the architecture of Inception-ResNet"""


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
from cgd_data_processing import CgdDataProcessing


IM_FILE = '/media/baxter/DataDisk/Cornell Grasps Dataset/rgb_aug/train/img1.png'
MODEL_FILE = 'grasp_detector_nightly_2018-05-14 22.04.48.h5'
TARGET_IMAGE_WIDTH = 224


def angle_diff(y_true, y_pred):
    return K.abs((y_pred[2] - y_true[2]) * 57.3)  # 57.3 deg == 1 rad


def run_grasp_predict(image_file, model):
    img = Image.open(image_file)
    # Resize to 224x224
    resized_img = img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH))
    im_array = np.asarray(img, dtype='float32')
    plt.imshow(img)
    plt.show()

    loaded_model = models.load_model(model)


if __name__ == '__main__':
    run_grasp_predict(IM_FILE, MODEL_FILE)