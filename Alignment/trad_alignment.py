'''
This file contain the main function for Alignment
'''''
import PIL
from PIL import Image
from Alignment.utility_plot import *
import time
from Alignment.utility_params import Params, get_params_path
import os
import cv2
from Alignment.utility_error import *
from Alignment.utility_distort import *
import sys
import warnings



def downscale(img_path, rate, output_name, save_ori_png=False, save_ori_png_path=None):
    '''
    Downscale the image and save it to './input_temp/' for evaluation
    :param img_path: str: the path of the image
    :param rate: float: the rate to downscale the image.
    :param output_name: str: the output name of the downscaled image
    :param save_ori_png: bool: whether to save the original image as png
    :return: None
    '''''
    # Load the image
    img = Image.open(img_path)

    # Save the resized image
    if save_ori_png:
        try:
            img.save(save_ori_png_path, dpi=(1200, 1200), format='PNG')
            print("1")
            print("Image saved successfully.")
        except Exception as e:
            print("Error saving the image:", str(e))

    # Get the original width and height
    width, height = img.size

    # Calculate the new width and height
    new_width = round(width * rate)
    new_height = round(height * rate)

    # Resize the image
    resized_image = img.resize((new_width, new_height))

    # Save the resized image
    output_path = './input_temp/' + output_name + '.png'
    resized_image.save(output_path, dpi=(1200, 1200), format='PNG')
    return


def crop_by_coord(img_path, coord_list, output_name):
    '''
    Crop the image by four coordinates
    :param img_path: str: the path of the image
    :param coord_list: list: the 4 coordinates of the image [left, right, top, bottom]
    :param output_name: str: the output name of the cropped image
    :return: None
    '''''
    # Load the image
    img = Image.open(img_path)
    # Define the four points
    l, r, t, b = coord_list

    # N.B. In OpenCV (cv2), the coordinate system of an image starts from the top-left corner.
    # Construct the target coordinates of four corners
    tl = (l, t) # top left
    bl = (l, b) # bottom left
    tr = (r, t) # top right
    br = (r, b) # bottom right
    # Construct the target height and width
    w = r - l
    h = b - t

    # Create a mask of zeros with the same shape as the input image, to get the channel information
    mask = np.zeros_like(img)
    # Downscale the mask to the target size
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert the image to a NumPy array
    img = np.array(img)
    # Copy the target region from img to the mask and convert it to PIL image
    mask = Image.fromarray(img[t:b, l:r])

    # Save the cropped image
    output_path = './input_temp/' + output_name + '.png'
    mask.save(output_path, dpi=(1200, 1200), format='PNG')
    return


def denseSift_extractFeatures(img, step_size=5):
    '''
    Extract denseSIFT features
    :param img: greyscale cv2 image: the image to extract features
    :param step_size: int: the step size retrieve denseSIFT features
    :return: the keypoints and the descriptors
    '''''
    # Initialize an empty numpy array to store the SIFT descriptors
    descriptor_list = np.empty(shape=(0, 128))
    # Create a SIFT object
    sift = cv2.SIFT_create()
    # Get the dimensions of the input image
    r, c = img.shape

    # Generate a grid of keypoints using the specified step size
    keyp = [cv2.KeyPoint(x, y, step_size) for y in range(0, r, step_size) for x in range(0, c, step_size)]
    # Compute SIFT features on the grayscale image using the generated keypoints
    _, des = sift.compute(img, keyp)
    # Append the computed descriptors to the descriptor_list numpy array
    descriptor_list = np.append(descriptor_list, des, axis=0)
    return keyp, descriptor_list.astype(np.float32)


def normalize_contrast(image, target_contrast):
    '''
    Normalize the contrast/intensity of images
    :param image: The image to normalize
    :param target_contrast: The average intensity of the target and reference images
    :return: 
    '''''
    # Calculate the mean intensity of the image, which is used as a contrast metric
    contrast = np.mean(image)
    # Calculate the adjustment factor needed to achieve the target contrast
    contrast_adjustment = target_contrast / contrast
    # Apply the contrast adjustment to the image by scaling the intensities
    adjusted_image = (image) * contrast_adjustment
    # Adjusted the image to [0, 255]
    return adjusted_image.astype(np.uint8)


def align_SIFT_FLANN_RANSAC(img1, img2, params, eva_addition=None, img1_ori=None, img2_ori=None, main_func=False):
    '''
    The main function for image alignment
    :param img1: 3-channels or greyscale cv2 image: reference image 
    :param img2: 3-channels or greyscale cv2 image: target image 
    :param params: Param: the params variable to call parameters in params.json
    :param eva_addition: anytype: the variable used to pass the value of the variable that needs to be updated. Could be stepsize, match_threshold, etc.
    :param img1_ori: 3-channels or greyscale cv2 image in original size: reference image 
    :param img2_ori: 3-channels or greyscale cv2 image in original size: target image 
    :param main_func: bool: Ture if the full pipeline (main.py) is running
    :return: 
    '''''
    if not main_func:
        # Clean output_temp folder
        clear_folder_if_not_empty(params.diff_img_path + "/")

    # Disable all warnings
    warnings.filterwarnings("ignore")

    # Check if img1 is in grayscale
    if len(img1.shape) == 2:
        gray1 = img1
    else:
        # Convert the image to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # Check if img1 is in grayscale
    if len(img2.shape) == 2:
        gray2 = img2
    else:
        # Convert the image to grayscale
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize parameters
    eva_mode = params.eva_mode[0]
    feature_mode = params.feature_mode[0]
    match_mode = params.match_mode[0]
    match_threshold = params.match_threshold[0]
    match_threshold_adaptive_mode = params.match_threshold_adaptive_mode[0]
    homo_mode = params.homo_mode[0]
    step_size = params.step_size[0]

    # Update the variable if needed during evaluation
    if eva_mode == "scale":
        step_size = eva_addition
    elif eva_mode == "scale2":
        step_size = eva_addition
        # Check greyscale
        if len(img1_ori.shape) != 2:
            img1_ori = cv2.cvtColor(img1_ori, cv2.COLOR_BGR2GRAY)
        if len(img2_ori.shape) != 2:
            img2_ori = cv2.cvtColor(img2_ori, cv2.COLOR_BGR2GRAY)
        ref = img1_ori
        tar = img2_ori
    elif eva_mode == "stepsize":
        step_size = eva_addition
    elif eva_mode == "feature_comparison":
        feature_mode = eva_addition

    # Calculate target contrast as the average of the contrasts of the two images
    target_contrast = (np.mean(gray1) + np.mean(gray2)) / 2

    # Normalize contrast for both images
    gray1 = normalize_contrast(gray1, target_contrast)
    gray2 = normalize_contrast(gray2, target_contrast)
    ref = gray1
    tar = gray2
    cv2.imwrite('./output_temp/ref.png', ref)
    cv2.imwrite('./output_temp/tar.png', tar)

    # print parameter settings
    print("Modes: ", f'eva_mode: {eva_mode},  feature_mode: {feature_mode},  match_mode: {match_mode}, match_threshold_adaptive_mode: {match_threshold_adaptive_mode}, homo_mode: {homo_mode}')
    print("Parameters: ", f'step_size: {step_size},  match_threshold: {match_threshold}')
    print("Image Info: ", f'image1: {img1.shape},  image2: {img2.shape}')

    # ------------------------------------------------------------------------------------------------------------------
    t1 = time.time()
    if feature_mode == "SIFT":
        # Initialize the SIFT detector
        sift = cv2.SIFT_create()
        # Find the keypoints and descriptors for the images
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
    elif feature_mode == "denseSIFT":
        # Find the keypoints and descriptors for the images
        kp1, des1 = denseSift_extractFeatures(gray1, step_size=step_size)
        kp2, des2 = denseSift_extractFeatures(gray2, step_size=step_size)
    elif feature_mode == "ORB":
        orb = cv2.ORB_create()
        # Detect keypoints and compute descriptors
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        # Convert ORB descriptors to float32 for FLANN matcher
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
    elif feature_mode == "BRISK":
        # Create BRISK object: BRISK (Binary Robust Invariant Scalable Keypoints)
        brisk = cv2.BRISK_create()
        # Detect keypoints and compute descriptors
        kp1, des1 = brisk.detectAndCompute(gray1, None)
        kp2, des2 = brisk.detectAndCompute(gray2, None)
        # Convert ORB descriptors to float32 for FLANN matcher
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
    elif feature_mode == "AKAZE":
        # Create AKAZE object
        akaze = cv2.AKAZE_create()
        # Detect keypoints and compute descriptors
        kp1, des1 = akaze.detectAndCompute(gray1, None)
        kp2, des2 = akaze.detectAndCompute(gray2, None)
        # Convert ORB descriptors to float32 for FLANN matcher
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
    t2 = time.time()
    print("t2", t2 - t1)
    print("The descriptors from img 1 is: ", len(kp1), "  The descriptors from img 2 is: ", len(kp2))

    # knnMatching can't handle more than (2**18) descriptors. To avoid error and increses the efficiency, uniformly retrieve elements to generate new tuples to 2**17.
    # N.B. Can only uniformly retrieve descriptors, otherwise, the matching result is very bad
    if len(kp1) >= 2**18:
        # Desired length for the new tuple
        new_tuple_length = 2 ** 17
        # Calculate the step size to uniformly retrieve elements
        step_size = len(kp1) // new_tuple_length
        print("interval for sampling img 1", step_size)
        # Construct the new tuple using slicing
        kp1 = kp1[::step_size]
        des1 = des1[::step_size]

    if len(kp2) >= 2**18:
        # Desired length for the new tuple
        new_tuple_length = 2 ** 17
        # Calculate the step size to uniformly retrieve elements
        step_size = len(kp2) // new_tuple_length
        print("interval for sampling img 2", step_size)
        # Construct the new tuple using slicing
        kp2 = kp2[::step_size]
        des2 = des2[::step_size]
    print("The descriptors from img 1 after reduction is: ", len(kp1), "  The descriptors from img 2 after reduction is: ", len(kp2))

    # ------------------------------------------------------------------------------------------------------------------
    if match_mode == "FLANN":
        # Define FLANN parameters
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        # Create FLANN matcher object
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Find matched keypoints between the descriptors
        matches = flann.knnMatch(des1, des2, k=2)
    else:
        print("Brute Force")
        # Initialize the brute force matcher
        bf = cv2.BFMatcher()
        # Find matched keypoints between the descriptors
        matches = bf.knnMatch(des1, des2, k=2)
    t3 = time.time()
    print("t3", t3 - t2)

    # ------------------------------------------------------------------------------------------------------------------
    if eva_mode == "match_threshold":
        # Directly return the required variables for match_threshold evaluation and no longer executes trad_alignment.py
        return kp1, kp2, matches, homo_mode

    # A function to ensure there is enough matches for homography estimation
    while 1:
        good = []
        for m, n in matches:
            # Filter good matches
            if m.distance < match_threshold * n.distance:
                good.append(m)
        # The input arrays should have at least 4 corresponding point sets to calculate Homography in function 'findHomography'
        if len(good) > 4^3 or match_threshold>1.0:
            break
        # If the number of good matches is less than 4, add more matches to continue to the next iteration
        match_threshold = match_threshold + 0.5
    print("match_threshold: ", match_threshold)

    # calculate the total number of matches for identical_evaluation_feature_comparison.py
    total_match = len(matches)
    # calculate the percentage of filtered matches for identical_evaluation_feature_comparison.py
    if total_match==0:
        percentage_match = 0
    else:
        percentage_match = len(good) / total_match
    print("total_match: ", total_match, "percentage_match: ", percentage_match)
    # Draw the matches
    match_points_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    t4 = time.time()
    print("t4",t4-t3)

    # ------------------------------------------------------------------------------------------------------------------
    # queryIdx: The index of the query descriptor in the query descriptors list.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # trainIdx: The index of the train descriptor in the train descriptors list.
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    t5 = time.time()
    print("t5",t5-t4)

    # ------------------------------------------------------------------------------------------------------------------
    # Get the approach of homography estimation
    if homo_mode == "RANSAC":
        method = cv2.RANSAC
    else:
        method = 0
    # Estimate the homography matrix
    H2, mask = cv2.findHomography(src_pts, dst_pts,
                                  method=method,
                                  ransacReprojThreshold=params.homo_mode[1],
                                  maxIters=int(params.homo_mode[2]),
                                  confidence=params.homo_mode[3]) # The mask is a 1-dimensional NumPy array, where each element represents a point in the input sets. A value of 1 in the mask indicates that the corresponding point was considered in the estimation of the homography matrix, while a value of 0 indicates that the point was considered as an outlier and not used.
    print("Homography Matrix: ", H2)
    t6 = time.time()
    print("t6",t6-t5)

    # ------------------------------------------------------------------------------------------------------------------
    # Apply the transformation with the estimated homography matrix
    wrapped_img = cv2.warpPerspective(ref, H2, (tar.shape[1], tar.shape[0]))
    t7 = time.time()
    print("t7",t7-t6)

    # Calculate the difference between the aligned images and the target image
    diff_img_color = calc_color_diff(wrapped_img, tar)
    diff_img_abs = calc_abs_diff(wrapped_img, tar)

    # Visualize and save the results
    show_result(params, match_points_img=match_points_img, mask=mask, wrapped_img=wrapped_img, diff_img_color=diff_img_color, diff_img_abs=diff_img_abs, main_func=main_func)
    t8 = time.time()
    print("t8", t8 - t7)

    return H2, wrapped_img, diff_img_abs, total_match, percentage_match



if __name__ == '__main__':
    # Set the maximum of processing
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 63).__str__()
    # Initialize params variable to call parameters in params.json
    params = Params(get_params_path())

    # # Downscale the input image for some evaluations
    # downscale(params.img1_path, 0.8, '1_small4', save_ori_png=False, save_ori_png_path=None)
    # downscale(params.img2_path, 0.8, '7_small4', save_ori_png=False, save_ori_png_path=None)

    # # Crop the input image for some evaluations. [left, right, top, bottom]
    # crop_by_coord(params.img2_path, [0, 1500, 0, 1500], 'img2_border1') # border with mark
    # crop_by_coord(params.img2_path, [3200, 4700, 3200, 4700], 'img2_border2') # border without mark
    # crop_by_coord(params.img2_path, [8200, 9700, 4500, 6000], 'img2_print') # prints

    # Disable all warnings
    warnings.filterwarnings("ignore")

    # Load the images
    img1 = cv2.imread(params.img1_path)  # reference image
    img2 = cv2.imread(params.img2_path)  # target image
    # img2 = cv2.resize(img2, (6938, 5013))

    # Implement image alignment
    align_SIFT_FLANN_RANSAC(img1, img2, params)








