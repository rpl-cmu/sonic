''' Derived from the original codebase of CAPSNet'''
from cmath import nan
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# u,v ; x, y; bearing, range
# for the point


class CtoFCriterion(nn.Module):
    def __init__(self, args):
        super(CtoFCriterion, self).__init__()
        self.args = args
        self.w_ec = args.w_epipolar_coarse
        self.w_ef = args.w_epipolar_fine
        self.w_cc = args.w_cycle_coarse
        self.w_cf = args.w_cycle_coarse
        self.w_std = args.w_std

    @staticmethod
    def pix_to_polar(point, bearing_center=256, range_max=10.0, range_min=0.1, image_width=512, image_height=512, bearing_max = 130.0):
        bearing = ((image_width-point[:,:,0])*np.deg2rad(bearing_max)/image_width) - np.deg2rad(bearing_max)/2
        range = ((range_max-range_min)/image_height)*(image_height-point[:,:,1]) + range_min
        return bearing, range
    @staticmethod
    def polar_to_pix(point,image_width=512, image_height=512, bearing_max = 130, range_max=10, range_min=0.1):
        range = point[1]
        bearing = point[0]
        v = image_height - ((range-range_min)*image_height)/(range_max-range_min)
        u = image_width - (bearing + np.deg2rad(bearing_max)/2)*image_width/np.deg2rad(bearing_max)
        return u,v
    
    @staticmethod
    def get_rel_pose(pose0, pose1):
        # relPose = pose1 @ torch.linalg.inv(pose0)
        relPose = torch.matmul(pose1, torch.linalg.inv(pose0)).float()
        rot = relPose[:,:3,:3]
        t = relPose[:,:3,3]
        return rot, t
    
    @staticmethod
    def convert_to_arc(range_1, bearing_1,sampled_phi):
        x = torch.einsum("i,jk->jki",torch.cos(torch.deg2rad(sampled_phi)),torch.cos(bearing_1))
        y = torch.einsum("i,jk->jki",torch.cos(torch.deg2rad(sampled_phi)),torch.sin(bearing_1))
        z = torch.ones_like(bearing_1)
        z = -torch.einsum("i,jk->jki",torch.sin(torch.deg2rad(sampled_phi)),z)
        
        P_sampled_collective = torch.stack((x,y,z),axis  = 3)

        P_sampled_collective =torch.einsum("ij,ijkl->ijkl",range_1,P_sampled_collective)
        
        return P_sampled_collective
    
    @staticmethod
    def transform_arc_points(P_sampled_collective, R, t):
        P_rotated_collective = torch.einsum("iml,ijkl->ijkm",R,P_sampled_collective)
        
        P_transformed_collective = P_rotated_collective + t.unsqueeze(1).unsqueeze(1)

        return P_transformed_collective
    
    @staticmethod
    def cartesian_arc_to_polar(P_transformed_collective):
        range_transformed   = torch.linalg.norm(P_transformed_collective, axis=3)
        bearing_transformed = torch.arctan2(P_transformed_collective[:,:,:,1], P_transformed_collective[:,:,:,0])
        return bearing_transformed, range_transformed

    def p2bearing(self,x,w):
        horizontal_fov = self.args.horizontal_fov
        return ((x-w/2)/w)*horizontal_fov

    def p2range(self, y,h):
        range_end = self.args.range_end
        range_start = self.args.range_start
        return range_start + y*range_end/h

    def bearing2p(self, b,w):
        horizontal_fov = self.args.horizontal_fov
        return w*(b/horizontal_fov)+ w/2

    def range2h(self, r,h):
        range_start = self.args.range_start
        range_end = self.args.range_end
        return (r-range_start)*h/range_end


    def calculate_distance(self, bearing_transformed, range_transformed, bearing2, range2):
      
        rep_range = range2.unsqueeze(2)
        rep_range = rep_range.repeat(1,1,self.args.num_samples)

        rep_bearing = bearing2.unsqueeze(2)
        rep_bearing = rep_bearing.repeat(1,1,self.args.num_samples)
        # Calculate the cosine of the difference in bearings
        cos_diff_bearing = torch.cos(rep_bearing - bearing_transformed)

        # Calculate the square of the distance using the formula
        dist_squared = range_transformed**2 + rep_range**2 - 2 * range_transformed * rep_range * cos_diff_bearing

        return dist_squared
    
    def mask_invalid_points(self, euc_dist, u, v,debug=False):
        # Define the mask for valid pixel values
        u_valid_mask = (u >= 0) & (u < 512)
        v_valid_mask = (v >= 0) & (v < 512)

        valid_mask = u_valid_mask & v_valid_mask
        final_mask = torch.all(valid_mask,dim=2) 
        # Apply the mask to euc_dist, u, and v
        valid_euc_dist = euc_dist[final_mask]
        if debug:
            return valid_euc_dist, u_valid_mask, v_valid_mask, valid_mask, final_mask
        
        return valid_euc_dist


    def set_weight(self, std, mask=None, regularizer=0.0):
        # getting weight using standard deviation and mask
        if self.args.std:
            inverse_std = 1. / torch.clamp(std+regularizer, min=1e-10)
            weight = inverse_std / torch.mean(inverse_std)
            weight = weight.detach()  # Bxn
        else:
            weight = torch.ones_like(std)

        if mask is not None:
            weight *= mask.float()
            weight /= (torch.mean(weight) + 1e-8)
        return weight



    def sonar_cycle_consistency_loss(self, coord1, coord1_loop, weight, im1, im2, loss_type="fine", th=40):
        '''
        compute the cycle consistency loss
        :param coord1: [batch_size, n_pts, 2]
        :param coord1_loop: the predicted location  [batch_size, n_pts, 2]
        :param weight: the weight [batch_size, n_pts]
        :param th: the threshold, only consider distances under this threshold
        :return: the cycle consistency loss value
        '''
        #TODO: Convert to SONAR
        h = 0; w= 0
        if(loss_type=="fine"):
            h, w = im1.size()[2:]
        if(loss_type=="coarse"):
            h, w = im1.size()[2:]
      
        bearing_1, dist_1 = self.pix_to_polar(coord1,
                                              bearing_center = w/2,
                                              range_max = self.args.range_max,
                                              range_min = self.args.range_start,
                                              image_width=w,
                                              image_height=h,
                                              bearing_max=self.args.horizontal_fov,
                                              )

        bearing_2, dist_2 = self.pix_to_polar(coord1_loop,
                                              bearing_center=w/2,
                                              range_max=self.args.range_max,
                                              range_min = self.args.range_start,
                                              image_width=w,
                                              image_height=h,
                                              bearing_max=self.args.horizontal_fov,
                                              )        

        euc_distance = torch.square(dist_1) + torch.square(dist_2) - 2 * dist_1 * dist_2 * (torch.cos(torch.deg2rad(bearing_2) - torch.deg2rad(bearing_1)))

        euc_distance = torch.mean(euc_distance)

        return euc_distance

    #TODO: remove images adding now for debugging
    def sonar_epipolar_cost(self, coord1, coord2, T1, T2, im1, im2, loss_type="fine"):
        '''
        compute sonar epipolar cost
        Coordinates come in, in (x,y) format
        :param coord1: query point for which we have to find the epipolar arc [batch_size, n_pts, 2]
        :param coord2: the predicted location  [batch_size, n_pts, 2]
        :param weight: the weight [batch_size, n_pts]
        :param th: the threshold, only consider distances under this threshold
        '''
        # get height, width based on the type
        h = 0; w= 0
        if(loss_type=="fine"):
            h, w = im1.size()[2:]
        if(loss_type=="coarse"):
            h, w = im1.size()[2:]
        
        # get distance and bearing based on the coordinates

        bearing_1, dist_1 = self.pix_to_polar(coord1,
                                              bearing_center = w/2,
                                              range_max = self.args.range_max,
                                              range_min = self.args.range_start,
                                              image_width=w,
                                              image_height=h,
                                              bearing_max=self.args.horizontal_fov,
                                              )

        bearing_2, dist_2 = self.pix_to_polar(coord2,
                                              bearing_center=w/2,
                                              range_max=self.args.range_max,
                                              range_min = self.args.range_start,
                                              image_width=w,
                                              image_height=h,
                                              bearing_max=self.args.horizontal_fov,
                                              )        

        # Get relative pose, projecting from T1 to T2
        rot, t = self.get_rel_pose(T1, T2)
        # Sampling points along the vertical field of view
        sampled_phi = torch.linspace(-self.args.vertical_fov/2,self.args.vertical_fov/2,self.args.num_samples, device=dist_1.device)
        # Getting the points along the arc of elevation ambiguity
        P_sampled_collective = self.convert_to_arc(dist_1, bearing_1, sampled_phi)
        # Transforming the sampled points into the second pose using rel_pose
        P_transformed_collective = self.transform_arc_points(P_sampled_collective, rot, t)
        # Converting the arc from cartesian to polar coordinates
        bearings_transformed, ranges_transformed = self.cartesian_arc_to_polar(P_transformed_collective)
        # Converting the bearings and ranges to pixels
        u, v = self.polar_to_pix((bearings_transformed,ranges_transformed))
        # Calculate distance between transformed ranges and bearings
        euc_dist = self.calculate_distance(bearings_transformed, ranges_transformed, bearing_2, dist_2)        
        # Find the loss by finding the minimum
        euc_loss_min = torch.min(euc_dist, axis=2)
        # getting the values from the min
        euc_loss_values = euc_loss_min.values
        # choosing euc_dist for which the epipolar contours are completely inside the image
        valid_euc_loss = self.mask_invalid_points(euc_loss_values, u, v)
        # return the loss values, not the indices
        return valid_euc_loss


    # adding image1 and image2 to forward for debugging, TODO: remove later    
    def forward(self, coord1, data, T1, T2, pose, im_size, im1, im2):
        # getting the coordinates for 1 & 2 in coarse and fine
        coord2_ec = data['coord2_ec']
        coord2_ef = data['coord2_ef']
        coord1_lc = data['coord1_lc']
        coord1_lf = data['coord1_lf']
        self.coarse_h =data['coarse_h']
        self.coarse_w = data['coarse_w']
        self.fine_h = data['fine_h']
        self.fine_w = data['fine_w']

        max_dist = self.args.range_max
        max_dist = max_dist**2

        # Passing images to the epipolar cost for debugging TODO: remove later
        epipolar_cost_c = self.sonar_epipolar_cost(coord1, coord2_ec, T1, T2, im1, im2, loss_type="coarse") # Bxn
        # only add fine level loss if the coarse level prediction is close enough to gt epipolar line
        mask_ctof = (epipolar_cost_c < max_dist)

        mask_epip_c = (epipolar_cost_c < max_dist * self.args.th_epipolar)
        mask_cycle_c = (epipolar_cost_c < max_dist * self.args.th_cycle)

        epipolar_cost_f = self.sonar_epipolar_cost(coord1, coord2_ef, T1, T2, im1, im2, loss_type="fine")
        # only add cycle consistency loss if the fine level prediction is close enough to gt epipolar line
        mask_epip_f = (epipolar_cost_f < max_dist * self.args.th_epipolar)
        mask_cycle_f = (epipolar_cost_f < max_dist * self.args.th_cycle)

        weight_c = 1
        weight_f = 1
        # weight_c = self.set_weight(std_c)
        # weight_f = self.set_weight(std_f)

        
        max_range = max_dist
        eloss_c = torch.mean(epipolar_cost_c * weight_c) / max_range
        eloss_f = torch.mean(epipolar_cost_f * weight_f) / max_range
        
        # Not using mask
        closs_c = self.sonar_cycle_consistency_loss(coord1,coord1_lc,None, im1, im2, "coarse") /max_range
        closs_f = self.sonar_cycle_consistency_loss(coord1,coord1_lf,None, im1, im2, "fine") /max_range

        # add the epipolar coarse, epi_fine, cyclic_fine, cyclic_coarse
        loss = self.w_ec * eloss_c + self.w_ef * eloss_f + self.w_cc * closs_c + self.w_cf * closs_f

        std_loss = 1
        if(torch.isnan(loss)):
            print(data)

        return loss, eloss_c, eloss_f, closs_c, closs_f, std_loss