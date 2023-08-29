'''
This file is to apply illumination change or affine transformationfor Identical Evaluation (IE)
'''''
import random
import imgaug.augmenters as iaa
import cv2
import numpy as np



def change_intensity_checkerboard(img1, params, intensity_rate=None):
    '''
    Apply checkerboard-like/patch-like intensity change, on identical image for identical evaluation
    :param img1: 3-channels or greyscale cv2 image: reference image
    :param params: the params variable to call parameters in params.json
    :param intensity_rate: intensity change (used for single_distiortion evaluation)
    :return: greyscale or 3-channels cv2 image: reference image with/without checkerboard-like intensity change
    '''''
    if intensity_rate is None:
        # if if_change_intensity_checkerboard==False, directly return
        if not params.if_change_intensity_checkerboard:
            return img1

    # Check if the image is grayscale, if not, convert
    if len(img1.shape) != 2:
        # Convert the image to grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if intensity_rate is None:
        # Define the intensity change
        if params.if_intensity_rate_rand:
            intensity_rate = random.randint(params.intensity_rate_rand_val[0], params.intensity_rate_rand_val[1])
        else:
            intensity_rate = params.intensity_rate_constant

    # Copy img1 for intensity changing
    img1_new = np.copy(img1).astype(np.int16)

    # Flag to decide if change intensity or not, so that T-F-T-F-T-...
    flag = True
    # Get stepsize
    ss = params.checkerboard_stepsize
    # Go through the whole image by stepsize
    for i in range(0, img1.shape[0]-ss, ss):
        for j in range(0, img1.shape[1]-ss, ss):
            if flag:
                # Apply the intensity change within a patch
                img1_new[i:i+ss, j:j+ss] = np.add(img1_new[i:i+ss, j:j+ss], intensity_rate)
            flag = not flag

    # Clip the values of img1 between 0 and 255. The datatype must be unit8, otherwise error occurs
    img1_new = np.clip(img1_new, 0, 255).astype(np.uint8)

    return img1_new


def change_intensity(img1, params, intensity_rate=None):
    '''
    Apply intensity change on the whole image, on identical image for identical evaluation
    :param img1: 3-channels or greyscale cv2 image: reference image
    :param params: Param: the params variable to call parameters in params.json
    :param intensity_rate: intensity change (used for single_distiortion evaluation)
    :return: greyscale or 3-channels cv2 image: reference image with/without intensity change
    '''''
    if intensity_rate is None:
        # if if_change_intensity==False, directly return
        if not params.if_change_intensity:
            return img1

    # Check if the image is grayscale, if not, convert it
    if len(img1.shape) != 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if intensity_rate is None:
        # Define the intensity change
        if params.if_intensity_rate_rand:
            intensity_rate = random.randint(params.intensity_rate_rand_val[0], params.intensity_rate_rand_val[1])
        else:
            intensity_rate = params.intensity_rate_constant

    # Apply intensity change
    img1 = np.add(img1, intensity_rate)

    # Clip the values of img1 between 0 and 255. The datatype must be unit8, otherwise error occurs
    img1 = np.clip(img1, 0, 255).astype(np.uint8)

    return img1


def random_distortion(img2, params, scale_rate=None, rotation_rate=None, translate_rate=None):
    '''
    Apply random distortion (linear or nonlinear) on identical image for identical evaluation
    :param img2: cv2 image: the image to be distorted
    :param params: Param: the params variable to call parameters in params.json
    :param scale_rate: scale rate (used for single_distiortion evaluation)
    :param rotation_rate: rotation rate (used for single_distiortion evaluation)
    :param translate_rate: translate rate (used for single_distiortion evaluation)
    :return: augmentation sequential pipeline, transformation np matrix, and greyscale cv2 image, 
    
    Operations:
    ElasticTransformation: Applies local deformations to the image. (non-linear)
    PiecewiseAffine: Distorts the image locally by moving small parts. (non-linear)
    PerspectiveTransform: Applies a random perspective transformation to the image.(non-linear)
    Affine: Performs affine transformations including scaling, rotation, and translation. (linear)
    BarrelDistortion: Applies a barrel distortion effect to the image. (non-linear)
    SigmoidContrast: Adjusts the contrast of the image using a sigmoid function. (non-linear)
    '''''
    # Initialization
    distort, H1, distorted_img2 = None, None, None
    linear_distort, nonlinear_distort = None, None
    # Define distortion rates
    if params.distort_mode[0] == "linear":
        if scale_rate is None:
            if params.if_scale_rate_rand:
                scale_rate = random.uniform(params.scale_rate_rand_val[0], params.scale_rate_rand_val[1])
            else:
                scale_rate = params.scale_rate_constant
        if rotation_rate is None:
            if params.if_rotation_rate_rand:
                rotation_rate = random.uniform(params.rotation_rate_rand_val[0], params.rotation_rate_rand_val[1])
            else:
                rotation_rate = params.rotation_rate_constant
        if translate_rate is None:
            if params.if_translate_rate_rand:
                translate_rate = random.uniform(params.translate_rate_rand_val[0], params.translate_rate_rand_val[1])
            else:
                translate_rate = params.translate_rate_constant
        # print the distortion rates
        print("distort_mode: linear ", "scale_rate:", scale_rate, "rotation_rate:", rotation_rate, "translate_rate:", translate_rate)

        # Define the distortion pipeline
        linear_distort = iaa.Sequential([
            iaa.Affine(scale={"x": scale_rate, "y": scale_rate},),
            iaa.Affine(rotate=rotation_rate),
            iaa.Affine(translate_percent={"x": translate_rate, "y": translate_rate})
        ])

    nonlinear_distort = iaa.Sequential([
        # iaa.PerspectiveTransform(scale=0.1, cval=0, mode="constant", keep_size=False, fit_output=True, polygon_recoverer=None)
        iaa.PiecewiseAffine(scale=0.1)
    ])

    # Convert the image from BGR to RGB
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if params.distort_mode[0] == "linear":
        distort = linear_distort
        # Apply distortion
        distorted_img2 = linear_distort.augment_image(img2)
        # Calculate the homography matrix H1 (Incorrect)
        H1 = np.array([
            [scale_rate * np.cos(np.radians(rotation_rate)), -scale_rate * np.sin(np.radians(rotation_rate)),  translate_rate/scale_rate],
            [scale_rate * np.sin(np.radians(rotation_rate)), scale_rate * np.cos(np.radians(rotation_rate)), translate_rate/scale_rate],
            [0, 0, 1]
        ])
    else:
        distort = nonlinear_distort
        distorted_img2 = nonlinear_distort.augment_image(img2)
    # Convert the augmented image back to BGR
    distorted_img2 = cv2.cvtColor(distorted_img2, cv2.COLOR_RGB2BGR)

    return distort, H1, distorted_img2