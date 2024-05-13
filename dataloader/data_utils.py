import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib
from .demo_superpoint import SuperPointFrontend 
import torch

'''Minor modifications from CAPS'''

def skew(x):
    '''
    converts it into a skew symmetric form, which can be used for crossproduct
    '''
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def rotateImage(image, angle):
    '''rotate the image'''
    #TODO: check if this is required
    # rotaion might not be necessary
    h, w = image.shape[:2]
    angle_radius = np.abs(angle / 180. * np.pi)
    cos = np.cos(angle_radius)
    sin = np.sin(angle_radius)
    tan = np.tan(angle_radius)
    scale_h = (h / cos + (w - h * tan) * sin) / h
    scale_w = (h / sin + (w - h / tan) * cos) / w
    scale = max(scale_h, scale_w)
    image_center = tuple(np.array(image.shape[1::-1]) / 2.)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    rotation = np.eye(4)
    rotation[:2, :2] = rot_mat[:2, :2]
    return result, rotation

def perspective_transform(img, param=0.001):
    '''do a warp perspective'''
    #TODO: Again I don't think this is necessary
    h, w = img.shape[:2]
    random_state = np.random.RandomState(None)
    M = np.array([[1 - param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand()],
                  [-param + 2 * param * random_state.rand(),
                   1 - param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand()],
                  [-param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand(),
                   1 - param + 2 * param * random_state.rand()]])

    dst = cv2.warpPerspective(img, M, (w, h))
    return dst, M


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

def preprocess_image(image, size):
    """ Preprocess image before generating keypoints for both akaze (norm_image) and superpoint(grayim)"""
    norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    norm_image = unsharp_mask(norm_image, kernel_size=(3, 3), sigma=0.6, amount=2.0, threshold=0)
    norm_image = norm_image.astype(np.uint8)

    # set up image for superpoint
    interp= cv2.INTER_AREA
    grayim = cv2.resize(norm_image, (640,480), interpolation=interp)
    grayim = grayim.astype(np.float32) / 255.

    return norm_image, grayim


def generate_query_kpts(img, mode=None, num_pts=None, akaze_pts=None, akaze_superpoint_pts= None, h=None, w=None):

    #initialize superpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        superpoint = SuperPointFrontend(weights_path='dataloader/superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=torch.cuda.is_available())
    except:
        superpoint = SuperPointFrontend(weights_path=parent_dir+'/dataloader/superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=False)
    #initialize akaze
    a = cv2.AKAZE_create()

    # preprocess image
    norm_image, grayim = preprocess_image(img, (h,w))

    kpa = a.detect(norm_image, None)
    kps, _, _ = superpoint.run(grayim)
    rescaled_kpts = rescale_keypoints(kps.T, (h,w))
    kpsp = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in rescaled_kpts]

    len_kpa = len(kpa)
    len_kpsp = len(kpsp)
    total = len_kpa + len_kpsp


    if(len_kpa<akaze_superpoint_pts or len_kpsp<akaze_superpoint_pts):
        return np.zeros((1,2))
    coord_a = np.array([[kp.pt[0], kp.pt[1]] for kp in kpa])
    coord_s = np.array([[kp.pt[0], kp.pt[1]] for kp in kpsp])
    coord = np.vstack((coord_a,coord_s))
    gen_kpts = []
    if total > num_pts:
        # randomly select num_pts from coord
        coord = coord[np.random.choice(coord.shape[0], num_pts, replace=False), :]
    else:
        # perform nearest neighbours to fill up the rest
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(coord)
        dist, idx = nbrs.kneighbors(coord)
        points = np.random.randint(low=0,high=len(coord)-1,size=num_pts-total)
        for i in points:
            gen_kpts.append([np.mean(coord[idx[i],0]),np.mean(coord[idx[i],1])])
        
        shuffle(gen_kpts)

        coord = np.vstack((coord, gen_kpts[:(num_pts-total)]))

    return coord



def prune_kpts(coord1, F_gt, im2_size, intrinsic1, intrinsic2, pose, d_min, d_max):
    '''Pruning some of the points'''
    '''Not used for SONICModel'''
    # compute the epipolar lines corresponding to coord1
    # convert to homogeneous
    coord1_h = np.concatenate([coord1, np.ones_like(coord1[:, [0]])], axis=1).T  # 3xn
    # get the epipolar line
    epipolar_line = F_gt.dot(coord1_h)  # 3xn
    # divide by the L2 norm
    epipolar_line /= np.clip(np.linalg.norm(epipolar_line[:2], axis=0), a_min=1e-10, a_max=None)  # 3xn

    # determine whether the epipolar lines intersect with the second image
    h2, w2 = im2_size
    # get the corners of the recangle
    corners = np.array([[0, 0, 1], [0, h2 - 1, 1], [w2 - 1, 0, 1], [w2 - 1, h2 - 1, 1]])  # 4x3
    # get the distance of the line from each of the corners
    dists = np.abs(corners.dot(epipolar_line))
    # if the epipolar line is far away from any image corners than sqrt(h^2+w^2) (diagonal)
    # it doesn't intersect with the image
    # if any line is farther than the diagonal distance from any of the corners
    # it doesn't intersect
    non_intersect = (dists > np.sqrt(w2 ** 2 + h2 ** 2)).any(axis=0)

    # determine if points in coord1 is likely to have correspondence in the other image by the rough depth range
    # basically reproject each point to the other's image ang check which ones are below and above some threshold
    # 1. Generate 3d points by first making the 2d points homogeneous and then multiplying by dmin and dmax to get
    # rough range
    # 2. Convert into 3d homogeneous points (basically 2d -> 2d homog -> 3d -> 3d homog)
    # Now reproject into the second image. And reject the ones which our outside the image bounds

    intrinsic1_4x4 = np.eye(4)
    intrinsic1_4x4[:3, :3] = intrinsic1
    intrinsic2_4x4 = np.eye(4)
    intrinsic2_4x4[:3, :3] = intrinsic2
    coord1_h_min = np.concatenate([d_min * coord1,
                                   d_min * np.ones_like(coord1[:, [0]]),
                                   np.ones_like(coord1[:, [0]])], axis=1).T
    coord1_h_max = np.concatenate([d_max * coord1,
                                   d_max * np.ones_like(coord1[:, [0]]),
                                   np.ones_like(coord1[:, [0]])], axis=1).T
    coord2_h_min = intrinsic2_4x4.dot(pose).dot(np.linalg.inv(intrinsic1_4x4)).dot(coord1_h_min)
    coord2_h_max = intrinsic2_4x4.dot(pose).dot(np.linalg.inv(intrinsic1_4x4)).dot(coord1_h_max)
    coord2_min = coord2_h_min[:2] / (coord1_h_min[2] + 1e-10)
    coord2_max = coord2_h_max[:2] / (coord1_h_max[2] + 1e-10)
    out_range = ((coord2_min[0] < 0) & (coord2_max[0] < 0)) | \
                ((coord2_min[1] < 0) & (coord2_max[1] < 0)) | \
                ((coord2_min[0] > w2 - 1) & (coord2_max[0] > w2 - 1)) | \
                ((coord2_min[1] > h2 - 1) & (coord2_max[1] > h2 - 1))

    ind_intersect = ~(non_intersect | out_range)
    return ind_intersect

def make_matching_figure(
    img0, img1, mkpts0, mkpts1, kpts0=None, kpts1=None, text=[], dpi=96, path=None
):
    # draw image pair
    assert (
        mkpts0.shape[0] == mkpts1.shape[0]
    ), f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    brightness = 2.1 
    # Adjusts the contrast by scaling the pixel values by 2.3
    contrast = 1.3
    img0 = cv2.addWeighted(img0, contrast, np.zeros(img0.shape, img0.dtype), 0, brightness)
    img1 = cv2.addWeighted(img1, contrast, np.zeros(img1.shape, img1.dtype), 0, brightness)

    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=0.8)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="yellow", edgecolor="g", s=1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="yellow", edgecolor="g", s=1)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c="lime",
                linewidth=0.5,
                alpha=0.5,
            )
            for i in range(len(mkpts0))
        ]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c="lime", s=7)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c="lime", s=7)

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig