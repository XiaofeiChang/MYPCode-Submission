'''
Process both Turner's and JRL's data for auto cropping. This func is implemented only once and no longer used after implementing successfully.
In other words, this file should not be called by the other files.
1. Downscale to 500x500 and record the corresponding ratios with the image name/index
2. Greyscale the image
'''''
import os
import PIL
from PIL import Image
from AutoCrop import utility
from pathlib import Path



def rescale_and_grey(all_data_dir, output_dir, target_w=500, target_h=500):
    '''
    Rescales images from a directory to a target width and height and converts them to grayscale.
    :param all_data_dir: str: The directory containing the input images.
    :param output_dir: str: The directory where the processed images will be saved.
    :param target_w: int: The target width of the resized image.
    :param target_h: int: The target height of the resized image.
    :return: dst: np
    '''''
    # Intialize a dictionary to save the image path, horizontal and vertical ratio of downscale
    image_info = {}
    i = 0
    # Use os.walk() to traverse through all directories and subdirectories
    for root, dirs, files in os.walk(all_data_dir):
        # Iterate through all files in the current directory
        for file in files:
            # Skip the .DS_Store files if any
            if '.DS_Store' in file:
                continue
            # Construct the relative path to the file
            img_path = os.path.join(root, file)
            # Construct the absolute path to the file
            abs_img_path = os.path.abspath(img_path)

            img = Image.open(img_path)

            # Calculate the horizontal and vertical ratio of downscale
            w_ratio, h_ratio = utility.get_ratio(img, target_w, target_h)

            # implement downscale linearly
            resized_image = img.resize((target_w, target_h), Image.LANCZOS)
            #Greyscale the image
            grey_image = resized_image.convert('L')
            # Save the resized image to the output directory
            img_output_dir = output_dir + str(i) + ".png"
            grey_image.save(img_output_dir)
            # Save the information of the image in image_info {image_id: (img_path, w_ratio, h_ratio)}
            image_info[str(i)] = (abs_img_path, w_ratio, h_ratio)
            i += 1
    return image_info

if __name__ == '__main__':
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    all_data_dir, output_dir = utility.getPath_auto_crop_preprocssing()
    image_info = rescale_and_grey(all_data_dir, output_dir)
