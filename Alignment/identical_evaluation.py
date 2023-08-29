'''
Single Identical Evaluation experiment on illumination change, affine transformation, perspective transformation, or local distortion.
'''''
from PIL import Image
from Alignment.utility_error import *
from Alignment.utility_distort import *
from Alignment.utility_params import Params, get_params_path



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

    cv2.imwrite('./output_temp/img1.png', img1)
    print("eva1")

    # Implement image alignment
    H2, wrapped_img, diff_img_abs, _, _ = align_SIFT_FLANN_RANSAC(img1, img2, params)

    # Just for displaying
    cv2.imwrite('./output_temp/evaluation_wrapped_img.png', wrapped_img)
    cv2.imwrite('./output_temp/evaluation_diff_abs.png', diff_img_abs)
    print("eva2")

    # Compute the inverse matrix H2
    H2_inv = np.linalg.inv(H2)

    # Just for displaying
    img2_inv = cv2.warpPerspective(img2_grey, H2_inv, (img2_grey.shape[1], img2_grey.shape[0]))
    cv2.imwrite('./output_temp/evaluation_img2_inv.png', img2_inv)

    # Calculate distance shifting error (MAE)
    error, std, min, max = calc_step_distance(params, img2_grey, linear_distort, H2_inv, stepsize=params.step_size[0], heatmap_idx=None)
    print("The average distance shift error is: ", error, "px, and the standard deviation is: ", std, "px", "min: ", min, "max: ", max)



