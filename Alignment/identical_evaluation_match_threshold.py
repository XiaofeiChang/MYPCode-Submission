'''
Aim: Experiment on the influence of "filtering threshold (FT)". Referring to Experiment ID A9 in the dissertation.
'''''
from PIL import Image
import matplotlib.pyplot as plt
from Alignment.utility_error import *
from Alignment.utility_distort import *


def validate_match_threshold(kp1, kp2, matches, current_t, homo_mode, params):
    '''
    Check if the match_threshold is valid, 1 by 1.
    :param kp1: list[cv2.KeyPoint]: keypoints from image1
    :param kp2: list[cv2.KeyPoint]: keypoints from image2
    :param matches: matches between image1 and image2
    :param current_t: float: the match_threshold value
    :param homo_mode: str: the homography estimation method
    :return: 
    '''''
    # Initialize the variables to return
    H2, wrapped_img, diff_img_abs, total_match, percentage_match = None,None,None,None,None

    # Filter the matches by the current match_threshold value
    good = []
    for m, n in matches:
        # Filter good matches
        if m.distance < current_t * n.distance:
            good.append(m)

    # calculate the total number of matches
    total_match = len(matches)
    # calculate the percentage of filtered matches
    if total_match == 0:
        percentage_match = 0
    else:
        percentage_match = len(good) / total_match

    # queryIdx: The index of the query descriptor in the query descriptors list.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # trainIdx: The index of the train descriptor in the train descriptors list.
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

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
                                  confidence=params.homo_mode[
                                      3])  # The mask is a 1-dimensional NumPy array, where each element represents a point in the input sets. A value of 1 in the mask indicates that the corresponding point was considered in the estimation of the homography matrix, while a value of 0 indicates that the point was considered as an outlier and not used.

    return H2, wrapped_img, diff_img_abs, total_match, percentage_match



if __name__ == '__main__':
    # Set the maximum of processing
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 63).__str__()
    # Initialize params variable to call parameters in params.json
    params = Params(get_params_path())

    # Load the image
    img2 = cv2.imread(params.img2_path)  # target image
    # Convert to greyscale
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply random distortion
    linear_distort, H1, img1 = random_distortion(img2, params)

    # Apply intensity change
    img1 = change_intensity(img1, params)
    # Apply intensity change with patches
    img1 = change_intensity_checkerboard(img1, params)

    # Initialize the lists to save data for plotting
    match_threshold_list = []
    error_list = []
    std_list = []
    min_list = []
    max_list = []
    percentage_match_list = []

    # Initialize the starting value of match_threshold and get the update rate
    current_t = 1.0
    update_rate = params.eva_rate_match_threshold[0]
    stepsize = params.step_size[0]

    # Retrieve the variables required for the evaluation
    kp1, kp2, matches, homo_mode = align_SIFT_FLANN_RANSAC(img1, img2, params, eva_addition=None, img1_ori=None, img2_ori=None)

    num_itr = params.experiment_num[0]
    # Start from 1.0 and iterate for (num_itr) times with a step size of (-update_rate)
    for i in range(num_itr):
        # Implement image alignment
        H2, _, _, total_match, percentage_match = validate_match_threshold(kp1, kp2, matches, current_t, homo_mode, params)

        # If H2 is not found from the matches filtered by current_t, terminate it
        if H2 is None:
            break

        # Compute the inverse matrix H'
        H2_inv = np.linalg.inv(H2)

        # Calculate distance shifting error (MAE)
        error, std, min, max = calc_step_distance(params, img2_grey, linear_distort, H2_inv, stepsize=stepsize, heatmap_idx=current_t)

        # Save the data for plotting
        match_threshold_list.append(current_t)
        error_list.append(error)
        std_list.append(std)
        min_list.append(min)
        max_list.append(max)
        percentage_match_list.append(percentage_match)

        # Update variables
        current_t = current_t - update_rate
        # Terminate if match_threshold = 0
        if current_t == 0:
            break

    # Round every number to 4 decimal places for plotting
    match_threshold_list = np.round(np.array(match_threshold_list), decimals=4)
    error_list = np.round(np.array(error_list), decimals=4)
    percentage_match_list = np.round(np.array(percentage_match_list)*100, decimals=4)
    std_list = np.round(np.array(std_list), decimals=4)
    min_list = np.round(np.array(min_list), decimals=4)
    max_list = np.round(np.array(max_list), decimals=4)

    fig1 = plt.figure("error")
    plt.plot(match_threshold_list, error_list, 'b-')
    plt.scatter(match_threshold_list, error_list, color='blue', marker='o')
    # Add x, y values as text for each point
    for i, j in zip(match_threshold_list, error_list):
        plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
        plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel('Matching Threshold')
    plt.ylabel('Mean Absolute Error (px)')
    plt.title('Matching Threshold against Mean Absolute Error')
    fig1.savefig(params.identical_eva_match_threshold_error_figure_path)

    # Plot charts to show the results
    fig2 = plt.figure("percentage match")
    plt.plot(match_threshold_list, percentage_match_list, 'b-')
    plt.scatter(match_threshold_list, percentage_match_list, color='blue', marker='o')
    # Add x, y values as text for each point
    for i, j in zip(match_threshold_list, percentage_match_list):
        plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
        plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel('Matching Threshold')
    plt.ylabel('Percentage of Filtered Matches (%)')
    plt.title('Matching Threshold against Percentage of Filtered Matches')
    sub_title = "Total Number of Matches: " + str(total_match)
    plt.suptitle(sub_title)
    fig2.savefig(params.identical_eva_match_threshold_percentage_match_figure_path)

    fig3 = plt.figure("std")
    plt.plot(match_threshold_list, error_list, 'b-')
    plt.scatter(match_threshold_list, error_list, color='blue', marker='o')
    # Add x, y values as text for each point
    for i, j in zip(match_threshold_list, error_list):
        plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
        plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel('Matching Threshold')
    plt.ylabel('Standard Deviation (px)')
    plt.title('Matching Threshold against Standard Deviation')
    fig3.savefig(params.identical_eva_match_threshold_std_figure_path)

    show_std_mean_min_max(x_list=match_threshold_list, std_list=std_list, mean_list=error_list, min_list=min_list,
                          max_list=max_list,
                          title="Statistical Absolute Error Information", x_name='Matching Threshold', y_name="Absolute Error (px)",
                          save_path=params.identical_eva_match_threshold_statistical_figure_path)

    # Print the results
    print("match_threshold: ", (match_threshold_list)[::-1])
    print("error: ", (error_list)[::-1])
    print("percentage_match: ", (percentage_match_list)[::-1])
    print("std: ", (std_list)[::-1])
    print("min: ", (min_list)[::-1])
    print("max: ", (max_list)[::-1])
