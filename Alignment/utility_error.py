'''
This file is to calculate the Mean Absolute Error
'''''
import cv2
from Alignment.trad_alignment import *
import imgaug as ia
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage
import statistics
from Alignment.utility_plot import show_heatmap



def calc_step_distance(params, img2_grey, distort, H2_inv, stepsize=50, heatmap_idx=None):
    '''
    Calculate the mean of pixel shifting/offset from the target transformation to the reference transformation (Mean Absolute Error, MAE). 
    The shifting/offset is the Euclidean distance between each pair of corresponding keypoints from the reference and the target.
    :param img2_grey: greyscale cv2 image: the target image
    :param distort: Sequential: the pipeline of pre-defined distortion
    :param H2_inv: Numpy: the inverse of the estimated homography matrix
    :param stepsize: int: the stepsize to construct uniform coordinate system
    :return: 
    '''''

    # Get the height and width of the target image
    h, w = img2_grey.shape

    # Initialize the coordinate list of keypoints
    coord_list = []
    h_max = 0
    w_max = 0
    # Construct uniform coordinate system of by the same step size as denseSIFT
    for i in range(0, h, stepsize):
        w_max = 0
        for j in range(0, w, stepsize):
            new_coord = [i,j]
            coord_list.append(new_coord)
            w_max += 1
        h_max += 1

    # Get temporary image used for further processes
    ia.seed(1)
    image = ia.quokka(size=(w, h))

    # Construct keypoints by converting Numpy list to Keypoint
    keypoints = [Keypoint(x=coord[0], y=coord[1]) for coord in coord_list]
    kps = KeypointsOnImage(keypoints, shape=img2_grey.shape)

    # Apply pre-defined distortion and get the transformed keypoints (Target Keypoints)
    image_aug, kps_aug = distort(image=image, keypoints=kps)

    # Initialize the coordinate list of Target keypoints
    coord_list_distort = []
    # Convert Keypoint variables to Numpy list
    for i in range(len(kps.keypoints)):
        before = kps.keypoints[i]
        after = kps_aug.keypoints[i]
        # Save the transformed keypoints in the same form as coord_list
        new_coord = [after.x, after.y] # use after.x_int and after.y_int to get rounded integer coordinates
        coord_list_distort.append(new_coord)
        # print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (i, before.x, before.y, after.x, after.y)
    # Reshape for further processes
    coord_list_distort = np.array(coord_list_distort).astype(np.float32).reshape((-1, 1, 2))

    # # Visualize image with keypoints before/after augmentation (shown below)
    # image_before = kps.draw_on_image(image, size=7)
    # image_after = kps_aug.draw_on_image(image_aug, size=7)
    # ia.imshow(image_before)
    # ia.imshow(image_after)

    # Apply the perspective transformation and get the transformed keypoints (Reference Keypoints)
    coord_list = np.array(coord_list).astype(np.float32).reshape((-1, 1, 2))
    H2_inv = np.array(H2_inv).astype(np.float32)
    coord_list_homo = cv2.perspectiveTransform(coord_list, H2_inv)

    # Round each element to the nearest integer
    coord_list_distort_int = np.round(coord_list_distort).astype(int)
    coord_list_homo_int = np.round(coord_list_homo).astype(int)

    # Reshape the lists for error calculating
    coord_list = coord_list.squeeze(1) # (14382, 2)
    coord_list_distort_int = coord_list_distort_int.squeeze(1) # (14382, 2)
    coord_list_homo_int = coord_list_homo_int.squeeze(1) # (14382, 2)
    coord_list_distort = coord_list_distort.squeeze(1)  # (14382, 2)
    coord_list_homo = coord_list_homo.squeeze(1)  # (14382, 2)

    # Initialize a list to save the offset/shifting of each keypoint between the reference and the target
    distance_list_int = []
    # Calculate MAE keypoint by keypoint
    for i in range(coord_list.shape[0]):
        # Retrieve a pair of corresponding keypoints from the reference and target images
        kp1 = coord_list_distort_int[i, :]
        kp2 = coord_list_homo_int[i, :]
        # Calculate the Euclidean distance as shifting/offset
        distance = np.linalg.norm(kp1 - kp2)
        distance_list_int.append(distance)
    # Calculate the mean of shifting to get MAE
    mae_error = np.mean(distance_list_int)
    # Calculate the std
    std = statistics.stdev(distance_list_int)
    min = np.min(distance_list_int)
    max = np.max(distance_list_int)

    # Initialize a list for plotting heatmap
    distance_list = []
    # Calculate MAE keypoint by keypoint using the original lists (non-int) for plotting
    for i in range(coord_list.shape[0]):
        # Retrieve a pair of corresponding keypoints from the reference and target images
        kp1 = coord_list_distort[i, :]
        kp2 = coord_list_homo[i, :]
        # Calculate the Euclidean distance as shifting/offset
        distance = np.linalg.norm(kp1 - kp2)
        distance_list.append(distance)
    # Plot heat map of distance list
    show_heatmap(params, distance_list, w_max, h_max, heatmap_idx)

    # # Calculate the mean of shifting to get MAE without rounding keypoints' coordinates to int
    # mae_error = np.mean(distance_list)
    # # Calculate the std
    # std = 0
    # min = np.min(distance_list)
    # max = np.max(distance_list)

    return mae_error, std, min, max

