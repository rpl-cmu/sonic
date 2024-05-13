import detection as det
import torch
import numpy as np
import cv2
from dataloader.demo_superpoint import PointTracker as pt
import matplotlib.pyplot as plt
import matplotlib
import config
from SONIC import sonic_model
from scipy.spatial.distance import cdist
from scipy import stats
import csv
import os
from scipy.spatial.transform import Rotation as R

path = "/home/akshay/Research/training/logs/GTSAM_log/"

# 1. Convert the pixels to range and bearing format
def pix_to_polar(point, bearing_center=256, range_max=10, range_min=0.1, image_width=512, image_height=512, bearing_max = 130):
    bearing = ((image_width-point[0])*np.deg2rad(bearing_max)/image_width) - np.deg2rad(bearing_max)/2
    range = ((range_max-range_min)/image_height)*(image_height-point[1]) + range_min
    return bearing, range

def pix_to_polar_batch(point, bearing_center=256, range_max=10, range_min=0.1, image_width=512, image_height=512, bearing_max = 130):
    bearing = ((image_width-point[:,0])*np.deg2rad(bearing_max)/image_width) - np.deg2rad(bearing_max)/2
    range = ((range_max-range_min)/image_height)*(image_height-point[:,1]) + range_min
    return bearing, range

# 2. While converting from range bearing to cartesian, sample the phi values.
def polar_to_cartesian_sampled_points(range, bearing, phi, sample_points=1):
    sampled_phi = np.deg2rad(np.linspace(-phi/2,phi/2, sample_points))
    X = range*np.cos(bearing)*np.cos(sampled_phi)
    Y = -range*np.sin(bearing)*np.cos(sampled_phi)
    Z = range*np.sin(sampled_phi)
    return np.array([X,Y,Z])

# 3. The sampled points are in Pose0, move them to pose1.
def pose_correction(pose):
    corr_pose = np.eye(4)
    Rot = pose[:3,:3].T
    corr_pose[:3,:3] = Rot
    trans = -Rot@pose[:3,3]
    corr_pose[:3,3]=trans
    return corr_pose

def transform_points(points, pose0, pose1):
    relPose = pose1 @ np.linalg.inv(pose0)
    rot = relPose[:3,:3]
    t = relPose[:3,3]
    # t= np.zeros(3)
    return np.array([np.dot(rot[0],points) + t[0], np.dot(rot[1],points) + t[1],np.dot(rot[2],points) + t[2]])

# 4. Convert the points back into range bearing format.
def cart_to_polar(points):
    range = np.linalg.norm(points, axis=0)
    bearing = np.arctan2(points[1,:], points[0,:])
    return bearing, range

def cart_to_polar_batch(points):
    range = np.linalg.norm(points, axis=1)
    bearing = np.arctan2(points[:,1,:], points[:,0,:])
    return bearing, range

# 5. Convert range, bearing to pixels.
def polar_to_pix(point,image_width=512, image_height=512, bearing_max = 130, range_max=10, range_min=0.1):
    # print(point.shape)
    range = point[1]
    bearing = point[0]
    v = image_height - ((range-range_min)*image_height)/(range_max-range_min)
    u = image_width - (bearing + np.deg2rad(bearing_max)/2)*image_width/np.deg2rad(bearing_max)
    return u,v

def polar_to_pix_batch(point,image_width=512, image_height=512, bearing_max = 130, range_max=10, range_min=0.1):
    range = point[1]
    bearing = point[0]
    v = image_height - ((range-range_min)*image_height)/(range_max-range_min)
    u = image_width - (bearing + np.deg2rad(bearing_max)/2)*image_width/np.deg2rad(bearing_max)
    return u,v


def make_matching_figure(
    img0, img1, mkpts0, mkpts1, kpts0=None, kpts1=None, text=[], dpi=150, path=None
):
    # draw image pair
    assert (
        mkpts0.shape[0] == mkpts1.shape[0]
    ), f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
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


def expectation_matching(img1_path, img2_path,pose1_path,pose2_path, model_path, match_path, th=0.1):
    # poses here are inconsquential because we only care about the matches.
    # Will use the matches later
    s1, s2, p1, p2 = det.load_data(
        img1_path,
        img2_path,
        pose1_path,
        pose2_path,
        mode="sonar",
    )
    # generate query keypoints for s1 and s2
    s1_kpts, s1_kpts_a, s1_kpts_s, s1_dsc_a, s1_dsc_s = det.generate_query_kpts(
        s1, mode="sonar", h=512, w=512
    )
    s2_kpts, s2_kpts_a, s2_kpts_s, s2_dsc_a, s2_dsc_s = det.generate_query_kpts(
        s2, mode="sonar", h=512, w=512
    )
    s1_clean = det.preprocess_image(s1, (512, 512), mode="sonar")
    s2_clean = det.preprocess_image(s2, (512, 512), mode="sonar")




######ONLY FOR REAL WORLD DATA##################
    # min_u, max_u = 0, 512
    # min_v, max_v = 220, 400

    # s1_kpts_np = np.array(s1_kpts)

    # filtered_kpts = s1_kpts_np[
    #     (s1_kpts_np[:, 0] >= min_u)
    #     & (s1_kpts_np[:, 0] <= max_u)
    #     & (s1_kpts_np[:, 1] >= min_v)
    #     & (s1_kpts_np[:, 1] <= max_v)
    # ]

    # # Convert the filtered keypoints back to a list if needed
    # # filtered_kpts_list = filtered_kpts.tolist()

    # s1_kpts = filtered_kpts
################################

    args = config.get_args()
    caps = sonic_model.SONICModel(args)
    
    caps.load_model(model_path)
    s1 = torch.from_numpy(s1.copy()).unsqueeze(0).cuda()
    s2 = torch.from_numpy(s2.copy()).unsqueeze(0).cuda()
    s1_kpts = torch.from_numpy(s1_kpts.copy()).int().unsqueeze(0).cuda()
    s2_kpts = torch.from_numpy(s2_kpts.copy()).int().unsqueeze(0).cuda()
    min_kpts = min(s1_kpts.shape[1], s2_kpts.shape[1])
    s1_kpts = s1_kpts[:, :min_kpts, :]
    s2_kpts = s2_kpts[:, :min_kpts, :]
    out_dict = caps.model.forward(s1, s2, s1_kpts)
    std_f = out_dict["std_f"].squeeze(0).detach().cpu().numpy()
    std_c = out_dict["std_c"].squeeze(0).detach().cpu().numpy()
    std_lc = out_dict["std_lc"].squeeze(0).detach().cpu().numpy()
    std_lf = out_dict["std_lf"].squeeze(0).detach().cpu().numpy()
    # argsort of std_f and sort it
    sorted_indices = np.argsort(std_f)
    std_f = np.sort(std_f)

    match_fine_points = out_dict["coord2_ef"].squeeze(0).detach().cpu().numpy()
    # sorted match_fine_points based on std_f
    match_fine_points = match_fine_points[sorted_indices] 
    match_coarse_points = out_dict["coord2_ec"].squeeze(0).detach().cpu().numpy()
    mkpts0 = s1_kpts.squeeze(0).detach().cpu().numpy()
    # sorted mkpts0 based on std_f
    mkpts0 = mkpts0[sorted_indices]
    mask = std_f < th
    ind_sel = np.where(mask)[0]
    mkpts0 = mkpts0[ind_sel]
    match_fine_points = match_fine_points[ind_sel]
    match_coarse_points = match_coarse_points[ind_sel]

    # make_matching_figure(s1.squeeze(0).detach().cpu().numpy(), s2.squeeze(0).detach().cpu().numpy(),
    #  mkpts0, mkpts1=match_fine_points, path =match_path)
    # make_matching_figure(
    #     s1.squeeze(0).detach().cpu().numpy(),
    #     s2.squeeze(0).detach().cpu().numpy(),
    #     mkpts0,
    #     mkpts1=match_fine_points,
    #     path=match_path,
    # )
    return mkpts0, match_fine_points 

def add_translation_noise(pose_matrix, sigma_x=0.0138, sigma_y=0.0138, sigma_z=0.0001, n_factor=2):
    """
    Add Gaussian noise to the translation part of the pose matrix.
    Parameters:
    - pose_matrix: 4x4 numpy array representing the original pose.
    - sigma_x, sigma_y, sigma_z: Standard deviations for the noise in x, y, z directions.
    Returns:
    - pose_with_noise: 4x4 numpy array representing the noisy pose.
    """
    # Create a deep copy of the original matrix to prevent modifying the input matrix
    pose_with_noise = np.copy(pose_matrix)
    # Generate translation noise
    x_noise = np.random.normal(0, sigma_x* n_factor) 
    y_noise = np.random.normal(0, sigma_y* n_factor) 
    z_noise = np.random.normal(0, sigma_z* n_factor) 
    # print(x_noise, y_noise, z_noise)
    # Add the noise to the pose matrix
    pose_with_noise[0, 3] += x_noise
    pose_with_noise[1, 3] += y_noise
    pose_with_noise[2, 3] += z_noise
    return pose_with_noise

import numpy as np

def euler_noise_to_rotation(pose_matrix, roll_sigma=1e-5, pitch_sigma=1e-6, yaw_sigma=1e-4, n_factor=2):
    # Convert Euler noise to axis-angle representation
    roll_noise = np.random.normal(0, roll_sigma* n_factor) 
    pitch_noise = np.random.normal(0, pitch_sigma* n_factor) 
    yaw_noise = np.random.normal(0, yaw_sigma* n_factor) 
    # Axis angle
    n = np.array([roll_noise, pitch_noise, yaw_noise])
    theta = np.linalg.norm(n)
    
    if theta == 0:
        return np.eye(3)

    n_unit = n / theta

    # Skew-symmetric matrix
    n_cross = np.array([
        [0, -n_unit[2], n_unit[1]],
        [n_unit[2], 0, -n_unit[0]],
        [-n_unit[1], n_unit[0], 0]
    ])

    # Compute the rotation matrix using Rodrigues' formula
    R_noise = np.eye(3) + np.sin(theta) * n_cross + (1 - np.cos(theta)) * np.dot(n_cross, n_cross)
    # print(R_noise, " noise")
    # print(pose_matrix, " pose matrix")
    temp = np.dot(pose_matrix[:3, :3], R_noise)
    # print(np.dot(pose_matrix[:3, :3].copy(), R_noise), " dot")
    # print(temp, " temp")
    pose_matrix[:3, :3] = temp.copy()
    # print(pose_matrix, "after")
    return pose_matrix

def get_projected_pix(mkpts0, pose1_path, pose2_path):
    bearings, ranges = pix_to_polar_batch(mkpts0)
    points_cartesian = []
    for i in range(len(bearings)):
        points_cartesian.append(polar_to_cartesian_sampled_points(ranges[i], bearings[i], 1, 1))
    points_cartesian = np.array(points_cartesian)
    pose0 = np.load(pose1_path)
    pose1 = np.load(pose2_path)
    pose1 = pose_correction(pose1)
    pose0 = pose_correction(pose0)
    relPose = pose1@ np.linalg.inv(pose0)
    rot = relPose[:3,:3]
    t = relPose[:3,3]
    if np.abs(t[0])<0.5 and np.abs(t[1])<0.5:
        noise_factor = 6
    elif np.abs(t[0])<1.5 and np.abs(t[1])<1.5:
        noise_factor = 10
    else:
        noise_factor = 12

    noise_factor =0.01

    pose0_t_noise = add_translation_noise(pose0, n_factor=noise_factor)
    pose0 = euler_noise_to_rotation(pose0_t_noise, n_factor=noise_factor)

    pose1_t_noise = add_translation_noise(pose1)
    pose1 = euler_noise_to_rotation(pose1_t_noise)
    
    transformed_points = []
    for i in range(len(points_cartesian)):
        transformed_points.append(transform_points(points_cartesian[i], pose0, pose1))
    transformed_points = np.array(transformed_points)

    new_bearings, new_ranges = cart_to_polar_batch(transformed_points)
    # print(new_bearings.shape)

    pix = polar_to_pix_batch(np.array([new_bearings, new_ranges]))
    pix = np.asarray(pix)
    pix = pix.reshape(2,-1).T
    return pix

def outlier_mask(gt_points, pred_points, z_th=1.0):
    np.random.seed(0)
    distances = np.min(cdist(pred_points, gt_points), axis=1)
    
    z_scores = stats.zscore(distances)
    inliers_under_5 = np.abs(distances)<20
    inlier_mask = np.abs(z_scores) < z_th
    inlier_indices = np.where(inlier_mask)[0]
    inlier_mkpts1 = pred_points[inlier_indices]
    inliers_under_5_indices = np.where(inliers_under_5)[0]
    std = np.std(distances)
    mean = np.mean(distances)
    print("Number of inliers:", len(inlier_indices))
    print("percetage inliers: ", len(inliers_under_5_indices)/len(gt_points)*100)
    per = len(inliers_under_5_indices)/len(gt_points)*100
    print(pred_points.shape)
    # print("Inlier indices:", inlier_indices)
    return inlier_indices, per, std, mean


def write_br_to_csv(pr_mkpts0, pr_mkpts1, filename):
    data = []
    bearings1, ranges1 = pix_to_polar_batch(pr_mkpts1)
    bearings0, ranges0 = pix_to_polar_batch(pr_mkpts0)
    bearings1 = np.rad2deg(bearings1)
    bearings0 = np.rad2deg(bearings0)

    # Assuming len(ranges0) == len(bearings0) == len(ranges1) == len(bearings1)
    for i in range(len(ranges0)):
        data.append([ranges0[i],bearings0[i],ranges1[i],bearings1[i]])
    # Specify the CSV file path
    csv_file = filename

    # Write the data to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["ranges0","bearings0","ranges1","bearings1"])
        # Write the data rows
        writer.writerows(data)

    print(f"Data has been saved to {csv_file}")

def write_pose_diff_xyz_ypr(pose0_path, pose1_path):
    pose0 = np.load(pose0_path)
    pose1 = np.load(pose1_path)
    pose1_c = pose_correction(pose1)
    pose0_c = pose_correction(pose0)
    relPose = pose1_c @ np.linalg.inv(pose0_c)
    rot = relPose[:3,:3]
    t = relPose[:3,3]
    if np.abs(t[0])<0.5 and np.abs(t[1])<0.5:
        noise_factor = 6
    elif np.abs(t[0])<1.5 and np.abs(t[1])<1.5:
        noise_factor = 10
    else:
        noise_factor = 12

    noise_factor=1

    pose0_c_noisy = add_translation_noise(pose0_c,n_factor=noise_factor)
    pose0_c_noisy = euler_noise_to_rotation(pose0_c_noisy,n_factor=noise_factor)
    pose1_c_noisy = add_translation_noise(pose1_c,n_factor=noise_factor)
    pose1_c_noisy = euler_noise_to_rotation(pose1_c_noisy,n_factor=noise_factor)


    # relPose = pose1_c @ np.linalg.inv(pose0_c)
    relPose_inv = pose0_c@np.linalg.inv(pose1_c)
    relPose_inv_noisy = pose0_c_noisy@np.linalg.inv(pose1_c_noisy)
    # rot = relPose[:3,:3]
    # t = relPose[:3,3]
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1]**2 + rot[2, 2]**2))
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    return t,yaw,pitch,roll, relPose, relPose_inv, relPose_inv_noisy

def matrix_to_xyz_quat(matrix, csv_file="output.csv"):
    # Extract translation
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    # Extract rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(rotation_matrix)
    quat = rotation.as_quat()
    qx, qy, qz, qw = quat  # Note: scipy returns [qx, qy, qz, qw]

    data = [[x, y, z, qw, qx, qy, qz]]

    # Write the data to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the header row
        writer.writerow(["x", "y", "z", "qw", "qx", "qy", "qz"])
    
        # Write the data row
        writer.writerows(data)

    print(f"Data has been saved to {csv_file}")

    return x, y, z, qw, qx, qy, qz

def matrix_to_xyz_ypr(matrix, csv_file="output.csv"):
    # Extract translation
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    # Extract rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert rotation matrix to Euler angles
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    data = [[x, y, z, yaw, pitch, roll]]

    # Write the data to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the header row
        writer.writerow(["x", "y", "z", "yaw", "pitch", "roll"])
    
        # Write the data row
        writer.writerows(data)

    print(f"Data has been saved to {csv_file}")

    return x, y, z, yaw, pitch, roll



if __name__ == "__main__":
    save_path = "/home/akshay/Research/sonar_slam_eval2/Eval/Easy/SONIC/"
    model_path = "/home/akshay/Research/sonar_slam_eval2/resnet34_64dim/130000.pth"
    pairs_path = "/home/akshay/Research/training/"

    imf1s_ = []
    imf2s_ = []
    pos1s_= []
    pos2s_= []

    img_folder = os.path.join(pairs_path,'logs/pairs_eval_easy_fixed.txt')
    pose_folder = os.path.join(pairs_path,'logs/pairs_pos_eval_easy_fixed.txt')

    if os.path.exists(img_folder):
        f = open(img_folder, 'r')

    if os.path.exists(img_folder):
        f_same = open(img_folder, 'r')

    if os.path.exists(pose_folder):
        p_val = open(pose_folder,'r')
    perc = []
    mean_ = []
    std_ = []
    for i,line in enumerate(f_same):
        im_line = f.readline()
        imf1, imf2 = im_line.strip().split(' ')
        pose_line = p_val.readline()
        pos1, pos2 = pose_line.strip().split(' ')

        sonar_image1_path = os.path.join(pairs_path,imf1)
        sonar_image2_path = os.path.join(pairs_path,imf2)
        pose1_path = os.path.join(pairs_path,pos1)
        pose2_path = os.path.join(pairs_path,pos2)

        match_path= save_path+"match_"+str(i)+".png"

        image1, image2 = det.load_data(
            sonar_image1_path, sonar_image2_path, None, None, mode="sonar"
        )

        mkpts0,mkpts1=expectation_matching(
            sonar_image1_path,
            sonar_image2_path,
            pose1_path,
            pose2_path,
            model_path,
            match_path,
            th=0.09,
        )
        pix = get_projected_pix(mkpts0, pose1_path, pose2_path)
        masked_idx,per, std, mean = outlier_mask(pix, mkpts1, z_th=2)
        perc.append(per)
        mean_.append(mean)
        std_.append(std)

        make_matching_figure(
            image1,
            image2,
            mkpts0[masked_idx],
            mkpts1[masked_idx],
            path=match_path
        )


    print("mean:", np.mean(perc))
    