'''
This file contains refinement and coordinate mapping
'''''
from AutoCrop.utility import *
from AutoCrop.auto_crop_model_utility import *
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from Alignment.utility_params import Params, get_params_path



def floodFill(src, dst, w, h, x, y):
    '''
    Function for floodfill
    :param src: np :binary mask image outputted from the neural network [0, 255]
    :param dst: np :black image that will be used to save the result of this func [0, 255]
    :param w: int :width of the src/dst
    :param h: int :height of the src/dst
    :param x: int :coordinates of initial seed x point
    :param y: int :coordinates of initial seed y point
    :return: dst: np
    '''''
    if (not (0 <= x and x < w and 0 <= y and y < h)):
        return

    # if the seed has been 1 (lines), directly return
    if (dst[x, y] == 255):
        return

    stack = []
    stack.append([x, y])

    while (len(stack) > 0):
        [x, y] = stack.pop()
        x1 = x;
        while (x1 >= 0 and src[x1, y] == 255):
            x1 -= 1  # goes to the left-most of each line first
        x1 += 1  # it goes boyond at the while above so goes back 1 pixel
        spanAbove = spanBelow = False  # initialization

        while (x1 < w and src[x1, y] == 255):
            dst[x1, y] = 255  # set 1 means mark it as a seed field
            if ((not (spanAbove)) and y >= 0 and src[x1, y - 1] == 255 and (not (dst[x1, y - 1] == 255))):
                stack.append(
                    [x1, y - 1])  # we push xy here and the next loop will firstly consider this line (Horizontal)
                spanAbove = True

            elif (spanAbove and y >= 0 and (not (src[x1, y - 1] == 255))):
                spanAbove = False;  # meet a boundary at src so does not span more

            if ((not (spanBelow)) and y < h - 1 and src[x1, y + 1] == 255 and (not (dst[x1, y + 1] == 255))):
                stack.append(
                    [x1, y + 1])  # we push xy here and the next loop will firstly consider this line (Horizontal)
                spanBelow = True;

            elif (spanBelow and y < h - 1 and (not (src[x1, y + 1] == 255))):
                spanBelow = False

            x1 += 1  # depth-first
    return dst


def dilate(width, height, src, dst, output_dilate_dir, file_name, radius):
    '''
    Dilation. The radius is half of the user input gap tolerance. The larger the radius, the greater the degree of dilation.
    :param src: np
    :param dst: np
    :param radius:int
    :return: dst - np
    '''''
    # let dst = src
    for x in range(0, width):
        for y in range(0, height):
            dst[x, y] = src[x, y]

    rr = radius * radius

    for y in range(0, height - 1):
        for x in range(0, width - 1):

            isEdge = src[x,y] == 255 and not(src[x-1, y] == 255 and src[x+1, y] == 255 and src[x, y-1] == 255 and src[x, y+1] == 255)

            if (isEdge):
                # a filter with size(2 * radius) x (2 * radius)
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):

                        if (dx * dx + dy * dy < rr) :
                            x1 = x + dx
                            y1 = y + dy

                            if (0 <= x1 and x1 < width and 0 <= y1 and y1 < height) :
                                dst[x1, y1] = 255
    if output_dilate_dir is not None:
        # Convert the NumPy matrix to a PIL image
        image_pil = Image.fromarray(dst)
        # Save the PIL image to a file
        image_pil = image_pil.convert("RGB")
        image_pil.save(output_dilate_dir + file_name, dpi=(1200, 1200), format='PNG', compress_level=0)
    return dst


def erode(width, height, src, dst, output_erode_dir, file_name, radius):
    '''
    Erosion. The radius is half of the user input gap tolerance. The larger the radius, the greater the degree of erosion.
    :param src: np
    :param dst: np
    :param radius:int
    :return: dst - np
    '''''
    # let dst = src
    for x in range(0, width):
        for y in range(0, height):
            dst[x, y] = src[x, y]

    rr = radius * radius

    for y in range(0, height-1):
        for x in range(0, width-1):

            isEdge = src[x, y] == 0 and (src[x-1, y] == 255 or src[x+1, y] == 255 or src[x,y-1] == 255 or src[x,y+1] == 255)

            if (isEdge):

                # a filter with size(2 * radius) x (2 * radius)
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):

                        if (dx * dx + dy * dy < rr) :
                            x1 = x + dx
                            y1 = y + dy

                            if (0 <= x1 and x1 < width and 0 <= y1 and y1 < height) :
                                dst[x1, y1] = 0
    if output_erode_dir is not None:
        # Convert the NumPy matrix to a PIL image
        image_pil = Image.fromarray(dst)
        # Save the PIL image to a file
        image_pil = image_pil.convert("RGB")
        image_pil.save(output_erode_dir + file_name, dpi=(1200, 1200), format='PNG', compress_level=0)
    return dst


def findCorners(src, w, h):
    '''
    By utilizing scanline horizontally and vertically, this func finds the leftmost, rightmost, topmost, bottommost 
    coordinates of the marked area in the binary mask. As the target foreground region is marked by 1, and the background region is marked by 0,
    just find the index that results in non-zero row/column value.
    :param src: np: binary mask image from floodfill func
    :param dst: np: black image that will be used to save the result of this func
    :param w: int: width of the image/mask
    :param h: int: height of the image/mask
    :return: int: leftmost, rightmost, topmost, bottommost coordinates
    '''''
    #Initialization
    left, top= 0, 0
    right, bottom = w-1, h-1
    # find top and bottom by summing the columns to get a row vector
    row_vec = np.sum(src, axis=1, keepdims=True)
    # find top by checking the indices in increasing order
    for i in range(h):
        if (row_vec[i,0] != 0):
            top = i
            break
    # find bottom by checking the indices in decreasing order
    for i in range(h-1, 0, -1):
        if (row_vec[i,0] != 0):
            bottom = i
            break

    # find left and right by summing the rows to get a column vector
    col_vec = np.sum(src, axis=0, keepdims=True)
    # find left by checking the indices in increasing order
    for i in range(w):
        if (col_vec[0,i] != 0):
            left = i
            break
    # find right by checking the indices in decreasing order
    for i in range(w-1, 0, -1):
        if (col_vec[0,i] != 0):
            right = i
            break
    return left, right, top, bottom


def rectangularize(left, right, top, bottom, dst, output_floodfill_dir, file_name):
    '''
    :param left: int: left boundary coordinate
    :param right: int: right boundary coordinate
    :param top: int: top boundary coordinate
    :param bottom: int: bottom boundary coordinate
    :param dst: np array: initialized binary mask
    :param output_floodfill_dir: str: path to save rectangularized refinement output
    :param file_name: str: file name of the image
    :return:
    '''''
    # Define the coordinates of the upper left and bottom right corners of the rectangle
    x1, y1 = left, top  # Upper left corner
    x2, y2 = right, bottom  # Bottom right corner

    # Draw the rectangle on the matrix
    dst[y1:y2 + 1, x1:x2 + 1] = 255  # Set the rectangle area to white

    # Convert the NumPy matrix to a PIL image
    image_pil = Image.fromarray(dst)

    # Save the PIL image to a file
    image_pil = image_pil.convert("RGB")
    image_pil.save(output_floodfill_dir+file_name, dpi=(1200,1200), format='PNG', compress_level=0)
    return


def coordinate_mapping(w_ratio, h_ratio, left, right, top, bottom):
    '''
    Map the coordinate to the original image according to the ratio.
                 original width 
    w_ratio = --------------------     ==>   original width = rescaled width * w_ratio
                 rescaled width
    '''''
    left = left * w_ratio
    right = right * w_ratio
    top = top * h_ratio
    bottom = bottom * h_ratio
    return left, right, top, bottom


def crop(dst, left, right, top, bottom, output_cropped_dir, file_name):
    '''
    :param dst: np array: original image to be cropeped
    :param left: int: left boundary coordinate
    :param right: int: right boundary coordinate
    :param top: int: top boundary coordinate
    :param bottom: int: bottom boundary coordinate
    :param output_cropped_dir: str: path to save final cropping output
    :param file_name: str: file name of the image
    :return:
    '''''
    # Define the coordinates of the upper left and bottom right corners of the rectangle
    x1, y1 = left, top  # Upper left corner
    x2, y2 = right, bottom  # Bottom right corner

    # Crop the image based on the coordinates
    cropped_image = dst.crop((x1, y1, x2, y2))

    # Get new width and height
    cropped_width, cropped_height = cropped_image.size

    # Save or display the cropped image
    cropped_image.save(output_cropped_dir+file_name, dpi=(1200,1200), format='PNG', compress_level=0)
    return cropped_width, cropped_height


def postprocessing_main(params=None, image_info=None, input_dir=None, output_dir=None, output_floodfill_dir=None, output_erode_dir=None, output_dilate_dir=None, output_cropped_dir=None):
    '''
    Refinement + Coordinate Mapping
    :param params: 
    :param image_info: dict: the dictionary containing image information
    :param input_dir: str: input image path
    :param output_dir: str: output image path
    :param output_floodfill_dir: str: floodfill mask path
    :param output_erode_dir: str: erosion mask path
    :param output_dilate_dir: str: dilation mask path
    :param output_cropped_dir: str: final cropping output path
    :return: 
    '''''
    # Retrieve the name of files for saving
    name_list = getName(input_dir)
    i = 0  # used to name the file

    # Use os.walk() to traverse through all directories and subdirectories
    for root, dirs, files in os.walk(output_dir):
        # Iterate through all files in the current directory
        for file in files:
            # Skip the .DS_Store files if any
            if '.DS_Store' in file:
                continue
            # Construct the absolute path to the file
            img_path = os.path.join(root, file)

            # Load the image
            img = Image.open(img_path)
            w, h = img.size

            # Convert the PIL image to a NumPy array
            img_np = np.array(img)

            # Apply Erosion
            radius = params.auto_crop_morphology_radius[0]
            img_erode_dst = np.zeros((w, h), dtype=np.uint8)
            img_erode = erode(w, h, img_np, img_erode_dst, output_erode_dir, name_list[i], radius=radius)
            # Apply Dilation
            img_dilite_dst = np.zeros((w, h), dtype=np.uint8)
            img_dilite = dilate(w, h, img_erode, img_dilite_dst, output_dilate_dir, name_list[i], radius=radius)

            # Floodfill algorithm to find target region
            x, y = int(w / 2), int(h / 2)  # initial seed points
            img_floodfill_dst = np.zeros(img.size)
            # Apply FloodFill
            img_floodfill = floodFill(img_dilite, img_floodfill_dst, w, h, x, y)


            # find Corners algorithm to find target rectangular region
            left, right, top, bottom = findCorners(img_floodfill, w, h)

            # Construct and save the rectangular mask (just for visualization)
            img_rect_dst = np.zeros(img.size)
            rectangularize(left, right, top, bottom, img_rect_dst, output_floodfill_dir, name_list[i])

            # Coordinate mapping
            if image_info is not None:
                # Pair the image_id of the current file and the image_id (key) in image_info to retrieve the info (value)
                file_name = name_list[i]   # e.g. '4.png'
                # Convert the result into an integer
                idx = file_name.replace('.png', '')   # e.g. '4'

                (ori_img_path, w_ratio, h_ratio) = image_info[idx]
                # Update the left, right, top, bottom by mapping them to original image
                left, right, top, bottom = coordinate_mapping(w_ratio, h_ratio, left, right, top, bottom)
                # Crop and save the image
                img_crop_path = ori_img_path
                img_crop = Image.open(img_crop_path)
                cropped_width, cropped_height = crop(img_crop, left, right, top, bottom, output_cropped_dir, name_list[i])
                # Update the value for i-th key and add more elements, (cropped_width and cropped_height), to its value
                image_info[idx] = image_info[idx] + (cropped_width, cropped_height)

            else:
                # Crop and save the image
                img_crop_path = os.path.join(input_dir, file)
                img_crop = Image.open(img_crop_path)
                crop(img_crop, left, right, top, bottom, output_cropped_dir, name_list[i])
            i+=1
    return image_info


if __name__ == '__main__':
    # Set the maximum of processing
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 63).__str__()
    # Initialize params variable to call parameters in params.json
    params = Params(get_params_path())

    input_dir, output_dir, output_floodfill_dir, output_cropped_dir = getPath_auto_crop_postprocssing()
    # The image_info is none when we don't apply coordinate mapping
    postprocessing_main(params, None, input_dir, output_dir, output_floodfill_dir, None, None, output_cropped_dir)