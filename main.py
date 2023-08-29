'''
1. For each one image, implement auto crop by unet and then crop a rectangle (find the leftmost coordinate, rightmost, topmost, bottommost by scanlines)
2. Don't worry if the scanline tasks a little bit long because we finally only process two images each time.
3. If you want to speed up, downscale the image to 100x100
'''

import os

import PIL
import cv2
from PIL import Image
from main_utility import *
from AutoCrop.auto_crop_preprocess import rescale_and_grey
from AutoCrop.auto_crop_test_main import UNet_pipeline
from AutoCrop.auto_crop_postprocess import postprocessing_main
from Alignment.utility_params import Params, get_params_path
from Alignment.trad_alignment import *
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Set the maximum of processing
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 63).__str__()
    # Initialize params variable to call parameters in params.json
    params = Params(get_params_path())

    # The root path of all the image used in this program
    img_root_path = params.auto_crop_image_root_path

    # Clean all folders
    print("Processing: Image folder initialization")
    clean_all_subfolders(img_root_path + "/AutoCrop/")
    clean_all_subfolders(img_root_path + "/Alignment/")
    print("Done!")

    # Perform automatic cropping only
    if params.auto_crop_data_mode[0] == "Batch":
        # Get path of folder contain the batch data
        data_path = img_root_path + "/Original Input/"
        t0 = time.time()
        # AutoCrop
        # Save the information of the image in image_info {image_id: (img_path, w_ratio, h_ratio)}
        print("Processing: AutoCrop")
        image_info = rescale_and_grey(data_path, img_root_path +"/AutoCrop/Preprocess/")
        t1 = time.time()
        print("Time1 (AutoCrop)", t1 - t0)
        print("Done!")
        # print(image_info)

        # UNet pipeline
        print("Auto Cropping: UNet Pipeline")
        UNet_pipeline(params.auto_crop_UNet_path, img_root_path+"/AutoCrop/Preprocess/", img_root_path+"/AutoCrop/UNet/")
        t2 = time.time()
        print("Time2 (UNet)", t2 - t1)
        print("Done!")

        # Refinement + Coordinate Mapping
        print("Auto Cropping: Refinement + Coordinate Mapping")
        image_info = postprocessing_main(params, image_info=image_info, input_dir=img_root_path+"/AutoCrop/Preprocess/", output_dir=img_root_path+"/AutoCrop/UNet/", output_floodfill_dir=img_root_path+"/AutoCrop/FloodFill/", output_erode_dir=img_root_path+"/AutoCrop/Erode/", output_dilate_dir=img_root_path+"/AutoCrop/Dilate/", output_cropped_dir=img_root_path+"/AutoCrop/Cropped/")
        t3 = time.time()
        print("Time3 (Refinement)", t3 - t2)
        print("Done!")

        # Save the information of the image in image_info {image_id: (img_path, w_ratio, h_ratio, cropped_w, cropped_h)}
        file_path = './image_info.txt'
        with open(file_path, 'w') as file:
            for key, value in image_info.items():
                file.write(f"{key}: {value}\n")

    #  Default full pipeline
    elif params.auto_crop_data_mode[0] == "Defalut":
        # Load the images
        img1 = cv2.imread(params.main_img1_path)
        img2 = cv2.imread(params.main_img2_path)

        # Save the images to input path for following processes
        cv2.imwrite(img_root_path+"/AutoCrop/Input/" + "0.png", img1)
        cv2.imwrite(img_root_path+"/AutoCrop/Input/" + "1.png", img2)

        t0 = time.time()
        # AutoCrop
        # Save the information of the image in image_info {image_id: (img_path, w_ratio, h_ratio)}
        print("Auto Cropping: AutoCrop")
        image_info = rescale_and_grey(img_root_path+"/AutoCrop/Input/", img_root_path + "/AutoCrop/Preprocess/")
        t1 = time.time()
        print("Time1 (AutoCrop)", t1 - t0)
        print("Done!")
        # print(image_info)

        # UNet pipeline
        print("Auto Cropping: UNet Pipeline")
        UNet_pipeline(params.auto_crop_UNet_path, img_root_path + "/AutoCrop/Preprocess/",
                      img_root_path + "/AutoCrop/UNet/")
        t2 = time.time()
        print("Time2 (UNet)", t2 - t1)
        print("Done!")

        # Refinement + Coordinate Mapping
        print("Auto Cropping: Refinement + Coordinate Mapping")
        image_info = postprocessing_main(params, image_info=image_info,
                                         input_dir=img_root_path + "/AutoCrop/Preprocess/",
                                         output_dir=img_root_path + "/AutoCrop/UNet/",
                                         output_floodfill_dir=img_root_path + "/AutoCrop/FloodFill/",
                                         output_erode_dir=img_root_path + "/AutoCrop/Erode/",
                                         output_dilate_dir=img_root_path + "/AutoCrop/Dilate/",
                                         output_cropped_dir=img_root_path + "/AutoCrop/Cropped/")
        t3 = time.time()
        print("Time3 (Refinement)", t3 - t2)
        print("Done!")

        # Image Alignment
        print("Alignment")
        cropped_img1 = cv2.imread(img_root_path + "/AutoCrop/Cropped/" + "0.png")
        cropped_img2 = cv2.imread(img_root_path + "/AutoCrop/Cropped/" + "1.png")
        H2, wrapped_img, diff_img_abs, total_match, percentage_match = align_SIFT_FLANN_RANSAC(cropped_img1, cropped_img2, params, eva_addition=None, img1_ori=None, img2_ori=None, main_func=True)
        t4 = time.time()
        print("Time4 (Alignment)", t4 - t3)
        print("Done!")








