import numpy as np
import lcm
import holoocean
import os
import sys
import cv2
import matplotlib.pyplot as plt
import torch 
import torchvision
from dataloader.demo_superpoint import SuperPointFrontend 


path = "/home/rplcmu/Documents/AUV_research/sonar_1_data/GTSAM_log/"




def load_data(sonar_path1, sonar_path2, pose_path1=None, pose_path2=None, mode = "sonar"):
    #check if file is npy or png
    if sonar_path1.endswith(".npy"):
        s1 = np.load(sonar_path1)
        s2 = np.load(sonar_path2)
        min_val = np.min(s1)
        max_val = np.max(s1)
        s1 = ((s1 - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        min_val = np.min(s2)
        max_val = np.max(s2)
        s2 = ((s2 - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        cv2.imwrite("s1.png", s1)
        cv2.imwrite("s2.png", s2)
        s1 = cv2.imread("s1.png", cv2.IMREAD_GRAYSCALE)
        s2 = cv2.imread("s2.png", cv2.IMREAD_GRAYSCALE)
        #convert to CV_8U
        # s1 = cv2.normalize(s1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        # s2 = cv2.normalize(s2, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        
    else:
        s1 = cv2.imread(sonar_path1, cv2.IMREAD_GRAYSCALE)
        s2 = cv2.imread(sonar_path2, cv2.IMREAD_GRAYSCALE)
    if mode == "sonar":
        s1 = np.flip(s1)
        s2 = np.flip(s2)

    # load corresponding poses which are in npy format
    if pose_path1 is not None and pose_path2 is not None:
        p1 = np.load(pose_path1)
        p2 = np.load(pose_path2)
        return s1, s2, p1, p2
    else:
        return s1, s2
    


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=3.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def rescale_keypoints(keypoints, size):
    """ Rescale keypoints to fit original image size.
    Inputs
      keypoints: Nx2 numpy array of keypoints.
      size: (H, W) tuple specifying original image size.
    Returns
      rescaled_keypoints: Nx2 numpy array of rescaled keypoints.
    """
    H, W = size 
    rescaled_keypoints = keypoints.copy()
    rescaled_keypoints[:, 0] = keypoints[:, 0] * W / 640
    rescaled_keypoints[:, 1] = keypoints[:, 1] * H / 480
    return rescaled_keypoints

def preprocess_image(image, size, mode=None):
    """ Preprocess image before generating keypoints for both akaze (norm_image) and superpoint(grayim)"""
    if mode == "sonar":
        norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        norm_image = anisotropic_diffusion(image,iterations=2, delta_t=0.08, kappa=10)
        norm_image = cv2.fastNlMeansDenoising(image, None, 5, 7, 21)
        norm_image = unsharp_mask(norm_image, kernel_size=(5, 5), sigma=0.5, amount=2.0, threshold=0.0)
        norm_image = norm_image.astype(np.uint8)
    else:
        norm_image = image

    # set up image for superpoint
    interp= cv2.INTER_AREA
    grayim = cv2.resize(norm_image, (640,480), interpolation=interp)
    grayim = grayim.astype(np.float32) / 255.

    return norm_image, grayim

def anisotropic_diffusion(image, iterations=10, delta_t=0.25, kappa=10):
    """
    Apply anisotropic diffusion to a grayscale image.

    Args:
        image (numpy.ndarray): The grayscale input image.
        iterations (int): Number of iterations for diffusion (default: 10).
        delta_t (float): Time step (default: 0.25).
        kappa (float): Diffusion coefficient (default: 10).

    Returns:
        numpy.ndarray: The filtered image.
    """
    # Convert the input image to a floating-point array
    filtered_image = image.astype(np.float64)

    # Define the gradient function using central differences
    def gradient(image):
        dx = np.gradient(image, axis=1)
        dy = np.gradient(image, axis=0)
        return dx, dy

    for _ in range(iterations):
        # Calculate the gradient components
        dx, dy = gradient(filtered_image)

        # Compute the diffusion coefficients
        c_x = 1 / (1 + (dx / kappa)**2)
        c_y = 1 / (1 + (dy / kappa)**2)

        # Update the image using the diffusion equation
        filtered_image += delta_t * (
            c_x * np.roll(filtered_image, shift=-1, axis=1) +
            c_y * np.roll(filtered_image, shift=-1, axis=0) -
            (c_x + np.roll(c_x, shift=1, axis=1) + c_y + np.roll(c_y, shift=1, axis=0)) * filtered_image
        )

    # Clip the values to ensure they stay within [0, 255] range
    filtered_image = np.clip(filtered_image, 0, 255)

    # Convert the result back to uint8
    filtered_image = filtered_image.astype(np.uint8)

    return filtered_image


def generate_query_kpts(img, mode=None, num_pts=None, akaze_pts=None, akaze_superpoint_pts= None, h=None, w=None):

    #initialize superpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    superpoint = SuperPointFrontend(weights_path='/home/akshay/Research/sonar_slam_eval2/src/two_view/superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=torch.cuda.is_available())
    #initialize akaze
    a = cv2.AKAZE_create()

    # preprocess image
    norm_image, grayim = preprocess_image(img, (h,w), mode)

    kpa, dsca = a.detectAndCompute(norm_image, None)
    kps,dscs, _ = superpoint.run(grayim)
    rescaled_kpts = rescale_keypoints(kps.T, (h,w))
    kpsp = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in rescaled_kpts]

    len_kpa = len(kpa)
    len_kpsp = len(kpsp)
    total = len_kpa + len_kpsp

    coord_a = np.array([[kp.pt[0], kp.pt[1]] for kp in kpa])
    coord_s = np.array([[kp.pt[0], kp.pt[1]] for kp in kpsp])
    #check if any of the two are empty, if so return the non empty one or return the vstack

    if len_kpa == 0:
        coord = coord_s
    elif len_kpsp == 0:
        coord = coord_a
    else:
        coord = np.vstack((coord_a,coord_s))
    
    return coord, coord_a, coord_s, dsca, dscs


def main():
        
    # load data
    s1, s2, p1, p2 = load_data(path + "sonar/SH_0.png", path + "sonar/SH_402.png", path + "sonar_pose/P_0.npy", path + "sonar_pose/P_402.npy", mode = "sonar")
    c1, c2, cp1, cp2 = load_data(path + "cam/cam_0.png", path + "cam/cam_402.png", path + "cam_pose/P_0.npy", path + "cam_pose/P_402.npy", mode = "cam")
    # generate query keypoints for s1 and s2
    s1_kpts, s1_kpts_a, s1_kpts_s, s1_dsc_a, s1_dsc_s = generate_query_kpts(s1, mode="sonar", h=512, w=512)
    s2_kpts, s2_kpts_a, s2_kpts_s, s2_dsc_a, s2_dsc_s = generate_query_kpts(s2, mode="sonar", h=512, w=512)
    # generate query keypoints for c1 and c2
    c1_kpts, c1_kpts_a, c1_kpts_s, c1_dsc_a, c1_dsc_s  = generate_query_kpts(c1, mode="cam", h=256, w=256)
    c2_kpts, c2_kpts_a, c2_kpts_s, c2_dsc_a, c2_dsc_s = generate_query_kpts(c2, mode="cam", h=256, w=256)

    #display the keypoints
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(s1)
    axs[0,0].scatter(s1_kpts[:,0], s1_kpts[:,1], s=1, c='r')
    axs[0,0].set_title("Sonar 1")
    axs[0,1].imshow(s2)
    axs[0,1].scatter(s2_kpts[:,0], s2_kpts[:,1], s=1, c='r')
    axs[0,1].set_title("Sonar 2")
    axs[1,0].imshow(c1)
    axs[1,0].scatter(c1_kpts[:,0], c1_kpts[:,1], s=1, c='r')
    axs[1,0].set_title("Camera 1")
    axs[1,1].imshow(c2)
    axs[1,1].scatter(c2_kpts[:,0], c2_kpts[:,1], s=1, c='r')
    axs[1,1].set_title("Camera 2")
    plt.show()

if __name__ == '__main__':
    main()


    