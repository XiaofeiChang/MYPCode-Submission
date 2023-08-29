'''
The utility functions for preprocessing
'''''
import glob
import os
from PIL import Image
from os.path import exists



def getPath_auto_crop_preprocssing():
    '''
    Get the path of the DATASET directory.
                MYPCode
            /        \          \
       AutoCrop Alignment  Dataset
    '''''
    # Get the current directory (where the auto_crop_preprocess.py file is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the DATASET dir
    substring_to_remove = "/AutoCrop"
    proj_dir = current_dir.replace(substring_to_remove, "")
    all_data_dir = proj_dir + '/DATASET'

    # construct the output dir of data process, which is the input of auto_crop neural network
    output_dir = all_data_dir + '/AutoCrop' + '/Input/'
    # # Clean up the output folder each time implementing this func (not necessary)
    # clean_folder(output_dir)
    return all_data_dir, output_dir


def getPath_auto_crop_postprocssing():
    '''
    :param input_dir: The path of the input (greyscaled square) images
    :param output_dir: The path of the output (mask) from neural network
    :param output_floodfill_dir: The path of the binary-mask to guide cropping
    :param output_cropped_dir: The path of the output (cropped) after postprocessing
    :return: 
    '''''
    # Get the current directory (where the auto_crop_preprocess.py file is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the DATASET dir
    substring_to_remove = "/AutoCrop"
    proj_dir = current_dir.replace(substring_to_remove, "")
    all_data_dir = proj_dir + '/DATASET'

    # construct the input dir
    input_dir = all_data_dir + '/AutoCrop' + '/input_val/'
    # construct the output dir from neural network, which is the input of auto_crop postprocessing
    output_dir = all_data_dir + '/AutoCrop' + '/output_val/'
    # construct the postprocessing output dir for the binary rectangle mask to guide cropping
    output_floodfill_dir = all_data_dir + '/AutoCrop' + '/Output_Floodfill_Mask/'
    # construct the postprocessing output dir for the cropped images (cropping's final result)
    output_cropped_dir = all_data_dir + '/AutoCrop' + '/Output_Cropped/'
    # # Clean up the output folder each time implementing this func (not necessary)
    # clean_folder(output_cropped_dir)
    return input_dir, output_dir, output_floodfill_dir, output_cropped_dir


def getPath_denoising():
    '''
    Get the input and output path of texture denoising
    '''''
    # Get the current directory (where the auto_crop_preprocess.py file is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the DATASET dir
    substring_to_remove = "/AutoCrop"
    proj_dir = current_dir.replace(substring_to_remove, "")
    all_data_dir = proj_dir + '/DATASET'

    # construct the output dir of data process, which is the input of auto_crop neural network
    output_dir = all_data_dir + '/AutoCrop' + '/Denoised/'
    # # Clean up the output folder each time implementing this func (not necessary)
    # clean_folder(output_dir)
    return all_data_dir, output_dir


def getName(dir):
    '''
    Retrieve all file names within the folder
    '''''
    file_names = os.listdir(dir)

    # Save the file names to a list
    name_list = []
    for file_name in file_names:
        # Skip the .DS_Store files if any
        if '.DS_Store' in file_name:
            continue
        name_list.append(file_name)
    return name_list


def get_ratio(img, target_w, target_h):
    '''
    Calculate the horizontal and vertical ratio of downscale
    :param img: The input image
    :param target_h_ratio: Desired target Width
    :param target_v_ratio: Desired target height
    :return: The ratio of horizontal and vertical downscaled image scale compared with the original image scale
    '''''
    original_w, original_h = img.size
    w_ratio = original_w / target_w
    h_ratio = original_h / target_h
    return w_ratio, h_ratio


def clean_folder(out_dir):
    '''
    Delete all the files in a directory which will be used to save the output
    '''''
    if os.path.getsize(out_dir) != 0:
        imgFiles = glob.glob(out_dir + '*')
        for f in imgFiles:
            os.remove(f)
    return
