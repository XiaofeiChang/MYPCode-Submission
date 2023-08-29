'''
The utility functions for plotting images in trad_alignment.py, including the "Difference Image" in 3-channels or 1-channel.
'''''
import os
import sys
import cv2
import numpy as np
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)  # Used to print full numpy array
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import seaborn as sns
import matplotlib.pyplot as plt



def clear_folder_if_not_empty(folder_path):
    '''
    Clear the output_temp folder for evaluation
    :param folder_path: str: the path of the folder to be clean
    '''''
    try:
        # Check if the folder is empty
        if not os.listdir(folder_path):
            return
        # List all the files in the folder
        file_list = os.listdir(folder_path)

        # Loop through the files and remove each one
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def calc_color_diff(wrapped_img, img2):
    '''
    Use Red and Blue colors to represent the difference of reference image and 
    :param wrapped_img: greyscale cv2 image: the aligned reference image
    :param img2: greyscale cv2 image: the reference image
    :return: 4-channels cv2 image: the difference image in Red and Blue
    '''''
    # Check if the image is grayscale, if not, convert it
    if len(wrapped_img.shape) != 2:
        wrapped_img = cv2.cvtColor(wrapped_img, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) != 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # for conveniently display the difference image, we initialize it by using pil format
    h, w = img2.shape
    # Generate a mask to save results in Red and Blue
    # Note: mask.shape[3]=4 : RGBA
    mask = np.full((h, w, 4), 0, dtype=np.uint8)

    # Calculate the difference between the images
    # UTF-8 can have a value from 0 to 255
    # int8 can have a value from -128 to 127. As there are negative values when subtracting, convert it to int8
    difference = (img2.astype(np.int16) - wrapped_img.astype(np.int16)).astype(np.int16)

    # Situation1: For where the intensity value in img2 > img1
    indices_B = np.where(difference > 0)
    mask[indices_B[0], indices_B[1], 3] = abs(difference[indices_B])  # record the Situation1 using Blue color in B_mask's alpha channel
    mask[indices_B[0], indices_B[1], 2] = 255

    # Situation2: For where the intensity value in img2 < img1
    indices_R = np.where(difference < 0)
    mask[indices_R[0], indices_R[1], 3] = abs(difference[indices_R])  # record the Situation2 using Red color in R_mask's alpha channel
    mask[indices_R[0], indices_R[1], 0] = 255

    # Convert the mask to Image
    mask = Image.fromarray(mask)

    return mask


def calc_abs_diff(wrapped_img, img2):
    '''
    Calculate the difference between the aligned images and the target image
    :param wrapped_img: greyscale cv2 image: the aligned reference image
    :param img2: greyscale cv2 image: the reference image
    :return: greyscale cv2 image: the difference image
    '''''
    # Calculate the absolute difference
    diff_img_abs = cv2.absdiff(wrapped_img, img2)
    return diff_img_abs


def show_heatmap(params, distance_list, w, h, heatmap_idx=None):
    '''
    Visualize the distance matrix (MAE error) by heatmap
    :param distance_list: 1d list: the list of MAE error, with each element refer to the error at the predefined keypoint
    :param w: int: the number of keypoints corresponding to the reference image in columns
    :param h: int: the number of keypoints corresponding to the reference image in rows
    :param h: float or str: index for saving heatmap
    :return: None
    '''''
    # Convert the list to numpy array
    distance_list = np.asarray(distance_list)

    # Reshape the array to (h,w)
    new_shape = (h, w)
    distance_list = distance_list.reshape(new_shape)

    # As there are too many keypoints, we apply a mean kernel to reduce the amount and then plot
    # Define the pooling window size and stride
    pool_size = (2, 2)
    stride = (2, 2)
    # Apply mean pooling using uniform_filter
    distance_list_mean = uniform_filter(distance_list, size=pool_size, mode='constant')[::stride[0], ::stride[1]]
    # Round to 4 decimal
    distance_list_mean = np.round(distance_list_mean, decimals=4)

    fig = plt.figure(figsize=(12, 8), dpi=100)
    # Generate the labels for cells
    text_str = []
    for i in range(len(distance_list_mean)):
        x = []
        for j in range(len(distance_list_mean[0])):
            x.append(str(distance_list_mean[i][j]))
        text_str.append(x)

    # Generate values (positions) for x and y. (start, stop, stride)
    x_value = np.arange(0.5, 0.5 + 0.5 + (distance_list_mean.shape[1]-1) * 1.0, 1.0)
    y_value = np.arange(0.5, 0.5 + 0.5 + (distance_list_mean.shape[0]-1) * 1.0, 1.0)

    # Generate labels for x and y.
    x_label = np.arange(1, 1 + 1 + (distance_list_mean.shape[1] - 1) * 1, 1).astype(int).astype(str)
    y_label = np.arange(1, 1 + 1 + (distance_list_mean.shape[0] - 1) * 1, 1).astype(int).astype(str)

    # vmin- minimum colour range value, vmax - maximum colour range value
    # ax = sns.heatmap(distance_list_mean, cmap='PiYG', vmin=distance_list_mean.min(), vmax=distance_list_mean.max(), annot=text_str, fmt="")
    ax = sns.heatmap(distance_list_mean, cmap='coolwarm', vmin=distance_list_mean.min(), vmax=distance_list_mean.max(), fmt="")

    plt.xticks(x_value, x_label, rotation=45)
    plt.yticks(y_value, y_label, rotation=45)
    # Move x-axis ticks to the top
    ax.xaxis.tick_top()

    # We are imagining that the squares in the heat-map are function values corresponding to domain points, [20,40,60,80]x[100,200,300]
    plt.title("Heatmap of Absolute Error")
    plt.xlabel("Horizontal Sample")
    plt.ylabel("Vertical Sample")
    if heatmap_idx is None:
        fig.savefig(params.heatmap_folder_path + "heatmap.png")
    else:
        fig.savefig(params.heatmap_folder_path + "heatmap" + str(heatmap_idx) + ".png")
    return


def show_result(params, match_points_img=None, mask=None, wrapped_img=None, diff_img_color=None, diff_img_abs=None, main_func=False):
    '''
    Visualize and save the results
    :param match_points_img: 3-channels cv2 image: the reference and target images with the matching points
    :param wrapped_img: greyscale cv2 image: the aligned image
    :param diff_img_color: 4-channels cv2 image: the difference between the aligned image and the target image in Red and Blue
    :param diff_img_abs: greyscale cv2 image: the absolute difference between the aligned image and the target image
    :param main_func: bool: Ture if the full pipeline (main.py) is running, which changes the save_path
    :return: None
    '''''
    if main_func == True:
        folder = params.main_diff_img_path
    else:
        folder = params.diff_img_path

    # Display the image if it is not None
    if match_points_img is not None:
        cv2.imwrite(folder + '/matchpoints.png', match_points_img)

    if mask is not None:
        cv2.imwrite(folder + '/mask.png', mask)

    if wrapped_img is not None:
        cv2.imwrite(folder + '/wrapped.png', wrapped_img)

    if diff_img_color is not None:
        # Save the PIL image with the original size and specified DPI
        diff_img_color.save(folder + '/diff_color.png', dpi=(1200,1200), format='PNG', compress_level=0)

    if diff_img_abs is not None:
        # Save the PIL image with the original size and specified DPI
        diff_img_abs = Image.fromarray(diff_img_abs)
        diff_img_abs.save(folder + '/diff_abs.png', dpi=(1200,1200), format='PNG', compress_level=0)


def show_std_mean_min_max(x_list, std_list, mean_list, min_list, max_list, title, x_name, y_name, save_path=None):
    '''
    Visualize the std, mean, min and max by a boxplot. Used in most of the identical evaluation function
    :param x_list: list: x-axis, in other words, in "value of changing variable" in control-variable experiment
    :param std: list: standard deviation of error
    :param mean: list: mean absolute error
    :param min: list: min of the absolute error
    :param max: list: max of the absolute error
    :param title: str: the title of the figure
    :param x_name: str: the name of the x-axis of the figure
    :param y_name: str: the name of the y-axis of the figure
    :param save_path: str: the path to save the plot
    :return: 
    '''''
    # Initialization
    x_anchor = np.arange(len(x_list)) # Anchor just used for plotting values near dots
    x_label = np.asarray(x_list).astype(str)
    max_list = np.asarray(max_list)
    mean_list = np.asarray(mean_list)
    min_list = np.asarray(min_list)
    std_list = np.asarray(std_list)
    # Prepare the bar chart 'Mean+-STD'. N.B.:
    # The upper boundary of bar chart is: np.add(mean_minus_std_list, std_list2)
    # The lower boundary of bar chart is: mean_minus_std_list
    subcategories = ['', 'Mean+-STD']
    std_list2 = std_list * 2
    mean_minus_std_list = np.subtract(mean_list,std_list)
    bar_data = np.vstack((mean_minus_std_list, std_list2))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Plot the stacked bars of Mean+-STD
    cnt = 0
    bottoms = np.zeros(len(x_label))
    for i, subcat_data in enumerate(bar_data):
        if cnt == 0:
            # Hide the region under the lower boundary by setting alpha=0
            ax.bar(x_label, subcat_data, bottom=bottoms, label=subcategories[i], alpha=0)
        else:
            ax.bar(x_label, subcat_data, bottom=bottoms, label=subcategories[i], color='gold', alpha=0.5)
        bottoms += subcat_data
        cnt += 1

    # Add Max, Mean, and Min
    plt.plot(max_list, color='firebrick', label="Max", marker='v', markersize=10, linestyle='dotted')
    plt.plot(mean_list, color='darkorange', label="Mean", marker='d', markersize=10)
    plt.plot(min_list, color='royalblue', label="Min", marker='^', markersize=10, linestyle='dotted')

    # # Add labels to the dots
    # for i, label in enumerate(x_label):
    #     plt.text(x_anchor[i], max_list[i]+0.05, str(max_list[i]), fontsize=12, ha='center', va='top', color='firebrick')
    #     plt.text(x_anchor[i], mean_list[i]+0.07, str(mean_list[i]), fontsize=12, ha='center', va='top', color='darkorange')
    #     plt.text(x_anchor[i], min_list[i]+0.1, str(min_list[i]), fontsize=12, ha='center', va='top', color='royalblue')

    # Add labels and title
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.title(title)
    # Put a legend below current axis
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Display the plot
    plt.tight_layout()
    plt.savefig(save_path)

