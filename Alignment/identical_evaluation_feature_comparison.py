'''
Aim: Experiment on different feature matching algorithms. Referring to Experiment ID A1-A5 in the dissertation.

N.B. Could use small image as input for experiment to save processing time.

Changing Variables: [feature_mode]
Fixed Variables: [match_threshold, stepsize (for estimation and evaluation), img1.shape, img2.shape, img1_ori.shape, img2_ori.shape]
'''''
from PIL import Image
import matplotlib.pyplot as plt
from Alignment.utility_error import *
from Alignment.utility_distort import *



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
    t_list = []
    error_list = []
    std_list = []
    min_list = []
    max_list = []
    total_match_list = [] # total number of matched keypoints extracted by the extractor
    percentage_match_list = [] # <percentage = (num of filtered matched keypoints)/(num of all matched keypoints)> by the extractor

    # Get other parameters
    feature_mode_list = params.feature_comparison_experiment_modes[:-1]
    stepsize = params.step_size[0]

    cnt=0
    # record the number of iteration for plotting
    num_itr_list = []
    # Iterate over the feature mode list
    for feature_mode in feature_mode_list:
        cnt+=1
        # Start timing
        t_s = time.time()

        # Implement image alignment
        H2, wrapped_img, diff_img_abs, total_match, percentage_match = align_SIFT_FLANN_RANSAC(img1, img2, params, eva_addition=feature_mode)

        # End timing
        t_e = time.time()

        # Compute the inverse matrix H'
        H2_inv = np.linalg.inv(H2)

        # Calculate distance shifting error (MAE)
        error, std, min, max = calc_step_distance(params, img2_grey, linear_distort, H2_inv, stepsize=stepsize, heatmap_idx=feature_mode)

        # Save the data for plotting
        num_itr_list.append(cnt)
        t_list.append(t_e - t_s)
        error_list.append(error)
        std_list.append(std)
        min_list.append(min)
        max_list.append(max)
        total_match_list.append(total_match)
        percentage_match_list.append(percentage_match)

        # Print experiment information
        print("IE_Scale",
              f'Experiment No.: {cnt},  time: {t_e - t_s},  error: {error}, std: {std}, total_match: {total_match}, percentage_match: {percentage_match}, imageshape: {img2.shape},  fearure_mode: {feature_mode},  stepsize: {stepsize}, match_threshold: {params.match_threshold[0]}')

    # Round every number to 4 decimal places for plotting
    num_itr_list = np.array(num_itr_list, dtype=int)
    t_list = np.round(np.array(t_list), decimals=4)
    error_list = np.round(np.array(error_list), decimals=4)
    total_match_list = np.round(np.array(total_match_list), decimals=4)
    percentage_match_list = np.round(np.array(percentage_match_list)*100, decimals=4) # (unit: %)
    std_list = np.round(np.array(std_list), decimals=4)
    min_list = np.round(np.array(min_list), decimals=4)
    max_list = np.round(np.array(max_list), decimals=4)

    # Labels for each data point on the x-axis
    x_values = num_itr_list
    x_labels = feature_mode_list

    # Plot BAR charts to show the results
    fig1 = plt.figure("time")
    # Plot the bar chart
    plt.bar(x_values, t_list)
    # Set custom x-axis tick positions and labels
    plt.xticks(x_values, x_labels, rotation=0)
    # Show the y-values on top of the bars
    for index, value in enumerate(t_list):
        plt.text(index+1, value, str(value), ha='center', va='bottom')
    plt.xlabel('Feature Extractors')
    plt.ylabel('Time(s)')
    plt.title('Time Consuming Using Different Feature Extractors')
    fig1.savefig(params.identical_eva_feature_mode_time_figure_path)

    fig2 = plt.figure("error")
    # Plot the bar chart
    plt.bar(x_values, error_list)
    # Set custom x-axis tick positions and labels
    plt.xticks(x_values, x_labels, rotation=0)
    # Show the y-values on top of the bars
    for index, value in enumerate(error_list):
        plt.text(index + 1, value, str(value), ha='center', va='bottom')
    plt.xlabel('Feature Extractors')
    plt.ylabel('Mean Absolute Error (px)')
    plt.title('Mean Absolute Error Using Different Feature Extractors')
    fig2.savefig(params.identical_eva_feature_mode_error_figure_path)

    fig3 = plt.figure("total match")
    # Plot the bar chart
    plt.bar(x_values, total_match_list)
    # Set custom x-axis tick positions and labels
    plt.xticks(x_values, x_labels, rotation=0)
    # Show the y-values on top of the bars
    for index, value in enumerate(total_match_list):
        plt.text(index + 1, value, str(value), ha='center', va='bottom')
    plt.xlabel('Feature Extractors')
    plt.ylabel('Total Number of Matches')
    plt.title('Total Number of Matches Using Different Feature Extractors')
    fig3.savefig(params.identical_eva_feature_mode_total_match_figure_path)

    fig4 = plt.figure("percentage match")
    # Plot the bar chart
    plt.bar(x_values, percentage_match_list)
    # Set custom x-axis tick positions and labels
    plt.xticks(x_values, x_labels, rotation=0)
    # Show the y-values on top of the bars
    for index, value in enumerate(percentage_match_list):
        plt.text(index + 1, value, str(value), ha='center', va='bottom')
    plt.xlabel('Feature Extractors')
    plt.ylabel('Percentage of Filtered Matches (%)')
    plt.title('Percentage of Filtered Matches Using Different Feature Extractors')
    fig4.savefig(params.identical_eva_feature_mode_percentage_match_figure_path)

    fig5 = plt.figure("std")
    # Plot the bar chart
    plt.bar(x_values, std_list)
    # Set custom x-axis tick positions and labels
    plt.xticks(x_values, x_labels, rotation=0)
    # Show the y-values on top of the bars
    for index, value in enumerate(std_list):
        plt.text(index + 1, value, str(value), ha='center', va='bottom')
    plt.xlabel('Feature Extractors')
    plt.ylabel('Standard Deviation (px)')
    plt.title('Standard Deviations Using Different Feature Extractors')
    fig4.savefig(params.identical_eva_feature_mode_std_figure_path)

    show_std_mean_min_max(x_list=x_labels, std_list=std_list, mean_list=error_list, min_list=min_list,
                          max_list=max_list,
                          title="Statistical Absolute Error Information", x_name='Feature Extractors', y_name="Absolute Error (px)",
                          save_path=params.identical_eva_feature_mode_statistical_figure_path)

    # Print the results
    print("feature mode: ", feature_mode_list)
    print("running time: ", t_list)
    print("error: ", error_list)
    print("std: ", std_list)
    print("min: ", (min_list))
    print("max: ", (max_list))
    print("total match: ", total_match_list)
    print("percentage match: ", percentage_match_list)








