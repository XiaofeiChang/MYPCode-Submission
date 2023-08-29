'''
Aim: Experiment on the influence of "stepsize" of denseSIFT on "running time" and "MAE error". Referring to Experiment ID A6 in the dissertation.

N.B. could use small image as input

Changing Variables: [stepsize (for estimation)]
Fixed Variables: [stepsize (for evaluation), match_threshold, feature_mode, img1.shape, img2.shape, img1_ori.shape, img2_ori.shape]
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
    stepsize_list = []
    std_list = []
    min_list = []
    max_list = []
    total_match_list = []
    percentage_match_list = []

    # Initialize the starting rate and get the update rate
    current_rate = 1
    update_rate = params.eva_rate_stepsize[0]
    stepsize = params.step_size[0]
    # save a copy
    stepsize_ori = stepsize

    # Get the number of experiments
    num_itr = params.experiment_num[0]
    for i in range(num_itr):
        # Start timing
        t_s = time.time()

        # Implement image alignment
        H2, wrapped_img, diff_img_abs, total_match, percentage_match = align_SIFT_FLANN_RANSAC(img1, img2, params, eva_addition=stepsize)

        # End timing
        t_e = time.time()

        # Compute the inverse matrix H'
        H2_inv = np.linalg.inv(H2)

        # Calculate distance shifting error (MAE)
        error, std, min, max = calc_step_distance(params, img2_grey, linear_distort, H2_inv, stepsize=stepsize_ori, heatmap_idx=current_rate)

        # Save the data for plotting
        t_list.append(t_e - t_s)
        error_list.append(error)
        stepsize_list.append(stepsize)
        std_list.append(std)
        min_list.append(min)
        max_list.append(max)
        total_match_list.append(total_match)
        percentage_match_list.append(percentage_match)

        # Print experiment information
        print("IE_Scale",
              f'Experiment No.: {i+1},  time: {t_e - t_s},  error: {error}, std: {std}, total_match: {total_match}, percentage_match: {percentage_match},  imageshape: {img2.shape},  current_rate: {current_rate},  stepsize: {stepsize}')

        # Check if terminate
        if i==num_itr-1:
            break

        # Update variables
        current_rate = current_rate * update_rate
        stepsize = round(stepsize * update_rate)

    # Round every number to 4 decimal places for plotting
    scale_rate_list = np.round(np.array(stepsize_list), decimals=4)
    t_list = np.round(np.array(t_list), decimals=4)
    error_list = np.round(np.array(error_list), decimals=4)
    std_list = np.round(np.array(std_list), decimals=4)
    min_list = np.round(np.array(min_list), decimals=4)
    max_list = np.round(np.array(max_list), decimals=4)
    total_match_list = np.round(np.array(total_match_list), decimals=4)
    percentage_match_list = np.round(np.array(percentage_match_list), decimals=4)

    # Plot charts to show the results
    fig1 = plt.figure("time")
    plt.plot(stepsize_list, t_list, 'b-')
    plt.scatter(stepsize_list, t_list, color='blue', marker='o')
    # # Add x, y values as text for each point
    # for i, j in zip(stepsize_list, t_list):
    #     plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
    #     plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel('Step Size of DenseSIFT')
    plt.ylabel('Time(s)')
    plt.title('Stepsize against Time Consuming')
    fig1.savefig(params.identical_eva_stepsize_time_figure_path)

    fig2 = plt.figure("error")
    plt.plot(stepsize_list, error_list, 'b-')
    plt.scatter(stepsize_list, error_list, color='blue', marker='o')
    # # Add x, y values as text for each point
    # for i, j in zip(stepsize_list, error_list):
    #     plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
    #     plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel('Step Size of DenseSIFT')
    plt.ylabel('Mean Absolute Error')
    plt.title('Stepsize against Mean Absolute Error')
    fig2.savefig(params.identical_eva_stepsize_error_figure_path)

    fig3 = plt.figure("std")
    plt.plot(stepsize_list, std_list, 'b-')
    plt.scatter(stepsize_list, std_list, color='blue', marker='o')
    # # Add x, y values as text for each point
    # for i, j in zip(stepsize_list, std_list):
    #     plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
    #     plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel('Step size of DenseSIFT')
    plt.ylabel('Standard Deviation (px)')
    plt.title('Stepsize against Standard Deviation')
    fig3.savefig(params.identical_eva_stepsize_std_figure_path)

    show_std_mean_min_max(x_list=stepsize_list, std_list=std_list, mean_list=error_list, min_list=min_list,
                          max_list=max_list,
                          title="Statistical Absolute Error Information", x_name="Step Size", y_name="Absolute Error (px)",
                          save_path=params.identical_eva_stepsize_statistical_figure_path)

    # Print the results
    print("stepsize_rate: ", (stepsize_list)[::-1])
    print("running time: ", (t_list)[::-1])
    print("error: ", (error_list)[::-1])
    print("std: ", (std_list)[::-1])
    print("min: ", (min_list)[::-1])
    print("max: ", (max_list)[::-1])
    print("total match: ", (total_match_list)[::-1])
    print("percentage match: ", (percentage_match_list)[::-1])







