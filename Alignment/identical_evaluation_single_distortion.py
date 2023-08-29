'''
Aim: Experiment on the robustness against Illumination Change or Affine Transformation (Scale, Rotation, Translation). Referring to Experiment ID A10-A15 in the dissertation.
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

    # Initialize the lists to save data for plotting
    scale_rate_list = []
    t_list = []
    error_list = []
    std_list = []
    min_list = []
    max_list = []
    total_match_list = []
    percentage_match_list = []

    # Initialize the starting rate and get the update rate
    scale_rate = 1.0
    rotation_rate = 0
    translate_rate = 0
    intensity_rate = 0
    linear_distort, H1, img1 = random_distortion(img2, params, scale_rate=scale_rate, rotation_rate=rotation_rate, translate_rate=translate_rate)

    update_rate = params.eva_rate_single_distort[0]
    stepsize = params.step_size[0]
    if params.eva_mode[0] == "single_distort_scale":
        current_rate = scale_rate
    elif params.eva_mode[0] == "single_distort_rotate":
        current_rate = rotation_rate
    elif params.eva_mode[0] == "single_distort_translate":
        current_rate = translate_rate
    elif params.eva_mode[0] == "single_distort_intensity":
        current_rate = intensity_rate
    elif params.eva_mode[0] == "single_distort_intensity_checkerboard":
        current_rate = intensity_rate

    # Get the number of experiments
    num_itr = params.experiment_num[0]
    for i in range(num_itr):
        # Start timing
        t_s = time.time()

        # Implement image alignment
        H2, wrapped_img, diff_img_abs, total_match, percentage_match = align_SIFT_FLANN_RANSAC(img1, img2, params)

        # End timing
        t_e = time.time()

        # Compute the inverse matrix H'
        H2_inv = np.linalg.inv(H2)

        # Calculate distance shifting error (MAE)
        error, std, min, max = calc_step_distance(params, img2_grey, linear_distort, H2_inv, stepsize=stepsize, heatmap_idx=current_rate)
        error = round(error,4)
        # Save the data for plotting
        scale_rate_list.append(current_rate)
        t_list.append(t_e - t_s)
        error_list.append(error)
        std_list.append(std)
        min_list.append(min)
        max_list.append(max)
        total_match_list.append(total_match)
        percentage_match_list.append(percentage_match)

        # Print experiment information
        print("IE_Scale",
              f'Experiment No.: {i+1},  time: {t_e - t_s},  error: {error}, std: {std}, total_match: {total_match}, percentage_match: {percentage_match}, imageshape: {img2.shape},  current_rate: {current_rate},  stepsize: {stepsize}')

        # Check if terminate
        if i==num_itr-1:
            break

        # Update variables
        if params.eva_mode[0] == "single_distort_scale":
            current_rate = current_rate - update_rate
            scale_rate = current_rate
            linear_distort, H1, img1 = random_distortion(img2, params, scale_rate=scale_rate,
                                                         rotation_rate=rotation_rate,
                                                         translate_rate=translate_rate)
        elif params.eva_mode[0] == "single_distort_rotate":
            current_rate = current_rate + update_rate
            rotation_rate = current_rate
            linear_distort, H1, img1 = random_distortion(img2, params, scale_rate=scale_rate,
                                                         rotation_rate=rotation_rate,
                                                         translate_rate=translate_rate)
        elif params.eva_mode[0] == "single_distort_translate":
            current_rate = current_rate + update_rate
            translate_rate = current_rate
            linear_distort, H1, img1 = random_distortion(img2, params, scale_rate=scale_rate,
                                                         rotation_rate=rotation_rate,
                                                         translate_rate=translate_rate)
        elif params.eva_mode[0] == "single_distort_intensity":
            current_rate = current_rate - intensity_rate
            intensity_rate = current_rate
            # Apply intensity change
            img1 = change_intensity(img1, params)

        elif params.eva_mode[0] == "single_distort_intensity_checkerboard":
            current_rate = current_rate - intensity_rate
            intensity_rate = current_rate
            # Apply intensity change with patches
            img1 = change_intensity_checkerboard(img1, params)

    # Round every number to 4 decimal places for plotting
    scale_rate_list = np.round(np.array(scale_rate_list), decimals=4)
    t_list = np.round(np.array(t_list), decimals=4)
    error_list = np.round(np.array(error_list), decimals=4)
    std_list = np.round(np.array(std_list), decimals=4)
    min_list = np.round(np.array(min_list), decimals=4)
    max_list = np.round(np.array(max_list), decimals=4)
    total_match_list = np.round(np.array(total_match_list), decimals=4)
    percentage_match_list = np.round(np.array(percentage_match_list), decimals=4)

    # Assign Plotting Info
    if params.eva_mode[0] == "single_distort_scale":
        xlabel = "Scale"
    elif params.eva_mode[0] == "single_distort_rotate":
        xlabel = "Rotation Degree"
    elif params.eva_mode[0] == "single_distort_translate":
        xlabel = "Translation"
    elif params.eva_mode[0] == "single_distort_intensity":
        xlabel = "Intensity"
    elif params.eva_mode[0] == "single_distort_intensity_checkerboard":
        xlabel = "Intensity (Checkerboard)"

    # Plot charts to show the results
    fig1 = plt.figure("time")
    plt.plot(scale_rate_list, t_list, 'b-')
    plt.scatter(scale_rate_list, t_list, color='blue', marker='o')
    # Add x, y values as text for each point
    for i, j in zip(scale_rate_list, t_list):
        plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
        # plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('Time(s)')
    plt.title(xlabel + ' against Time Consuming')
    fig1.savefig(params.identical_eva_single_distort_time_figure_path)

    fig2 = plt.figure("error")
    plt.plot(scale_rate_list, error_list, 'b-')
    plt.scatter(scale_rate_list, error_list, color='blue', marker='o')
    # Add x, y values as text for each point
    for i, j in zip(scale_rate_list, error_list):
        plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
        # plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('Mean Absolute Error')
    plt.title(xlabel + ' against Mean Absolute Error')
    fig2.savefig(params.identical_eva_single_distort_error_figure_path)

    fig3 = plt.figure("std")
    plt.plot(scale_rate_list, std_list, 'b-')
    plt.scatter(scale_rate_list, std_list, color='blue', marker='o')
    # Add x, y values as text for each point
    for i, j in zip(scale_rate_list, std_list):
        plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
        # plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('Standard Deviation (px)')
    plt.title(xlabel + ' against Standard Deviation')
    fig3.savefig(params.identical_eva_single_distort_std_figure_path)

    fig4 = plt.figure("total match")
    plt.plot(scale_rate_list, total_match_list, 'b-')
    plt.scatter(scale_rate_list, total_match_list, color='blue', marker='o')
    # Show the y-values on top of the bars
    for i, j  in zip(scale_rate_list,total_match_list):
        plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
        # plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('Total Number of Matches')
    plt.title(xlabel + ' against Total Number of Matches')
    fig4.savefig(params.identical_eva_single_distort_total_match_figure_path)

    fig5 = plt.figure("percentage match")
    plt.plot(scale_rate_list, percentage_match_list, 'b-')
    plt.scatter(scale_rate_list, percentage_match_list, color='blue', marker='o')
    # Show the y-values on top of the bars
    for i, j  in zip(scale_rate_list,percentage_match_list):
        plt.text(i, j, f"x={i}", ha='center', va='bottom', color='green', alpha=0.5)
        # plt.text(i, j, f"y={j}", ha='center', va='top', color='blue', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('Match Utilization Rate (MUR) in Percentage')
    plt.title(xlabel + 'against Match Utilization Rate (MUR)')
    fig5.savefig(params.identical_eva_single_distort_percentage_match_figure_path)

    show_std_mean_min_max(x_list=scale_rate_list, std_list=std_list, mean_list=error_list, min_list=min_list, max_list=max_list,
                          title="Statistical Absolute Error Information", x_name=xlabel, y_name="Absolute Error (px)",
                          save_path=params.identical_eva_single_distort_statistical_figure_path)

    # Print the results
    print("scale_rate: ", (scale_rate_list)[::-1])
    print("running time: ", (t_list)[::-1])
    print("error: ", (error_list)[::-1])
    print("std: ", (std_list)[::-1])
    print("min: ", (min_list)[::-1])
    print("max: ", (max_list)[::-1])
    print("total match: ", (total_match_list)[::-1])
    print("percentage match: ", (percentage_match_list)[::-1])
