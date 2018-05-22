"""A source code developed by Shehan Caldera (shehancaldera@gmail.com) for the purpose of image augmentation for
the Cornell Grasp Dataset. The augmented data will be useful for training robotic grasp detection convolutional
neural networks for detecting grasp rectangles from a given image. As of now the output data format is a RGB image
followed by the grasp rectangle params : x, y, theta, h, w .

This script will generate @NUMBER_AUGMENT_SAMPLES per original image+grasp instance and then their respective
x-shifted, y-shifted, and rotated instances as augmented samples."""


# Define the global variables & imports
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

import convnet_training


BASE_DIR = '/media/baxter/DataDisk/Cornell Grasps Dataset/original'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
BACKGROUND_IMAGE = '/media/baxter/DataDisk/Cornell Grasps Dataset/original'
TARGET_IMAGE_WIDTH = 224
TRAIN_START_INSTANCE = 100 #Initial CGD instance is pcd0100.png & pcd0100cpos.txt
TRAIN_INSTANCE_RANGE = 950 #First 850 instances are for training => (%100:%949)
TEST_START_INSTANCE = 1000 #Initial CGD instance for test data is pcd1000.png & pcd1000cpos.txt
TEST_INSTANCE_RANGE = 1035 #(1000:1034) instances are for testing

# Data augmentation params
NUMBER_AUGMENT_SAMPLES = 2 #Augmented samples per original image+label
DX = np.random.randint(-16, 16, size=NUMBER_AUGMENT_SAMPLES) #Variation of the augmentation

# Save data dir
SAVE_PATH = '/media/baxter/DataDisk/Cornell Grasps Dataset/rgb_aug_1'
TRAIN_SAVE = os.path.join(SAVE_PATH, 'train')
TEST_SAVE = os.path.join(SAVE_PATH, 'test') # the test set is not augmented


# Make dirs for saving
def make_save_dir():
    os.mkdir(SAVE_PATH)
    os.mkdir(TRAIN_SAVE)
    os.mkdir(TEST_SAVE)


# Function for opening images and resizing them appropriately
def open_image(instance_num, image_dir):
    SIZE_BEFORE_AUGMENT = 256
    CROP_IMG_WIDTH = 320
    if instance_num <1000:
        base_filename = 'pcd0'
    else:
        base_filename = 'pcd'
    
    img_filename = os.path.join(image_dir, base_filename + str(instance_num) + "r" + "." + 'png')
    img = Image.open(img_filename)
    
    # Cropping a center 270x270 and Resizing to the preferred size (256x256)
    crop_img = img.crop(box=(120, 80, 520, 480))
    resized_img = crop_img.resize((SIZE_BEFORE_AUGMENT, SIZE_BEFORE_AUGMENT))
    return resized_img


# Read bboxes
def open_bbox(instance_num, bbox_dir):
    if instance_num <1000:
        base_filename = 'pcd0'
    else:
        base_filename = 'pcd'
    
    filename = os.path.join(bbox_dir, base_filename + str(instance_num) + "cpos" + "." + 'txt')
    with open(filename) as f:
        bbox = list(map(
            lambda coordinate: float(coordinate), f.read().strip().split()))
       
    return bbox


# Function for converting bounding boxes to grasps
def bboxes_to_grasps(box):
    # box = float(box) / TARGET_IMAGE_WIDTH #Refract the pixel locations in unit distances
    # bbox => grasp (x, y, h, w, sin 2theta, cos 2)
    scale_x = 120
    scale_y = 80
    
    for i in range(0,8,2):
        box[i] = box[i] - scale_x
        
    for i in range(1,9,2):
        box[i] = box[i] - scale_y
    
    x = (box[0] + box[2] + box[4] + box[6]) / 4
    y = (box[1] + box[3] + box[5] + box[7]) / 4
    
    div_denominator = box[6] - box[0] #Avoiding the ZeroDivision error in python
    if div_denominator == 0:
        tan = 0.0
    else:
        tan = (box[7] - box[1])/(box[6] - box[0])
    
    '''Range of tan is confined between -89 deg : 89 deg as tan(90 deg) reaches infinity
    tan(89 deg) = 57.27 '''
    if tan < 0:
        tan = max(tan, -57)
    else:
        tan = min(tan, 57)
        
    theta = 100 * np.arctan(tan)
    tan = 10*tan
    
    h = np.sqrt(np.power(box[2]-box[0], 2) + np.power(box[3]-box[1], 2))
    w = np.sqrt(np.power(box[6]-box[0], 2) + np.power(box[7]-box[1], 2))
    
    grasp = [x, y, theta, h, w]
    
    return grasp


# Function for writing augmented images and grasps into the disks
def save_datapoint(img_file, grasp_array, save_instance, save_dir):
    save_img_filename = os.path.join(save_dir, "img" + str(save_instance) + "." + 'png')
    save_grasp_filename = os.path.join(save_dir, "grasp" + str(save_instance) + "." + 'txt')
    
    img_file.save(save_img_filename)
    np.savetxt(save_grasp_filename, grasp_array)


# Function for creating test images with much less augmentation and minor scaling
def save_test_data():
    print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Test data saving...')
    save_int = 0

    for instance_num in range(TEST_START_INSTANCE, TEST_INSTANCE_RANGE):
        bboxes = open_bbox(instance_num, TEST_DIR)

        for box_num in range(0, len(bboxes), 8):
            # No augmentation
            grasps = bboxes_to_grasps(bboxes[box_num:box_num + 8])
            img = open_image(instance_num, TEST_DIR)
            resized_img = img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH))

            # save grasps, img
            save_datapoint(resized_img, grasps, save_int, TEST_SAVE)
            print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Test datapoint # saved:' + str(save_int))
            save_int = save_int + 1


# Continuous data augmentation
def run_train_data_augmentation():
    print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Data augmentation...')
    save_int = 0

    for instance_num in range(TRAIN_START_INSTANCE, TRAIN_INSTANCE_RANGE):
        bboxes = open_bbox(instance_num, TRAIN_DIR)

        for box_num in range(0, len(bboxes), 8):
            # No augmentation
            grasps = bboxes_to_grasps(bboxes[box_num:box_num + 8])
            img = open_image(instance_num, TRAIN_DIR)
            resized_img = img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH))

            # save grasps, img
            save_datapoint(resized_img, grasps, save_int, TRAIN_SAVE)
            print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Train datapoint # saved:' + str(save_int))
            save_int = save_int + 1

            # X-trans augmentation
            for i in range(len(DX)):
                dx = DX[i]
                grasps[0] = grasps[0] + dx
                trans_img = img.crop(box=(16 + dx, 0, 240 + dx, 224))

                # save grasps, trans imgs
                save_datapoint(trans_img, grasps, save_int, TRAIN_SAVE)
                print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Train datapoint # saved:' + str(save_int))
                save_int = save_int + 1

            # Y-trans augmentation
            for i in range(len(DX)):
                dy = DX[i]
                grasps[1] = grasps[1] + dy
                trans_img = img.crop(box=(0, 16 + dy, 224, 240 + dy))

                # save grasps, trans imgs
                save_datapoint(trans_img, grasps, save_int, TRAIN_SAVE)
                print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Train datapoint # saved:' + str(save_int))
                save_int = save_int + 1

            # Rot augmentation
            for i in range(len(DX)):
                d_theta = DX[i]
                grasps[2] = grasps[1] + d_theta
                rot_img = img.rotate(d_theta)
                rot_resized_img = rot_img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_WIDTH), resample=Image.NEAREST)

                # save grasps, trans imgs
                save_datapoint(rot_resized_img, grasps, save_int, TRAIN_SAVE)
                print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Train datapoint # saved:' + str(save_int))
                save_int = save_int + 1

    print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Data augmentation: DONE')
    print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()) + ': Nmbr of train datapoints saved:' + str(save_int - 1))


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())+': Data Augmentation & Writing- by Shehan Caldera')
    make_save_dir()
    run_train_data_augmentation()
    save_test_data()
