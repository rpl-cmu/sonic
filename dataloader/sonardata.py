import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import skimage.io as io
import torchvision.transforms as transforms
import utils
import collections
from tqdm import tqdm
import dataloader.data_utils as data_utils
import config


rand = np.random.RandomState(234)

class SonarDataLoader():
    
    def __init__(self, args):
        self.args = args 
        # initialising the dataset class
        if args.phase == 'train':
            self.dataset = SonarData(args)
        else:
            self.dataset = SonarData(args)
        # inittialising the pytorch dataloader using args
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers, collate_fn=self.my_collate)
        if args.phase!= 'train':
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.workers, collate_fn=self.my_collate)
            
    def my_collate(self, batch):
        ''' Puts each data field into a tensor with outer dimension batch size '''
        # The filter() function extracts elements from an iterable (list, tuple etc.) for which a function returns True.
        batch = list(filter(lambda b: b is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def load_data(self):
        return self.data_loader
    
    def name(self):
        return 'SonarDataLoader'

    def __len__(self):
        return len(self.dataset)


# The dataset class for SonarData will have to change it for Sonar
class SonarData(Dataset):
    def __init__(self, args):
        self.args = args
        if args.phase == 'train':
            self.transform = transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485),
                                                                      std=(0.229)),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485),
                                                                      std=(0.229)),
                                                 ])
        # train or test
        self.phase = args.phase
        
        self.root = args.datadir
        if args.phase == "val":
            self.root = args.val_data_dir
        self.imf1s, self.imf2s, self.pos1s, self.pos2s = self.read_pairs()
        print('total number of image pairs loaded: {}'.format(len(self.imf1s)))
        # shuffle data
        index = np.arange(len(self.imf1s))
        rand.shuffle(index)
        self.imf1s = list(np.array(self.imf1s)[index])
        self.imf2s = list(np.array(self.imf2s)[index])
        self.pos1s = list(np.array(self.pos1s)[index])
        self.pos2s = list(np.array(self.pos2s)[index])
    

    def read_pairs(self):
        imf1s, imf2s = [], []
        pos1s, pos2s = [], []
        
        print('reading image pairs from {}...'.format(self.root))
        imf1s_ = []
        imf2s_ = []
        pos1s_= []
        pos2s_= []
        img_folder = os.path.join(self.root,'logs/test_pairs_SHIP.txt')
        pose_folder = os.path.join(self.root,'logs/test_pairs_pos_SHIP.txt')
        if(self.args.phase=='val'):
            img_folder = os.path.join(self.root,'logs/test_pairs_val_SHIP.txt')
            pose_folder = os.path.join(self.root,'logs/test_pairs_pos_val_SHIP.txt')
        
        if os.path.exists(img_folder):
            f = open(img_folder, 'r')
        for line in f:
            imf1, imf2 = line.strip().split(' ')
            imf1s_.append(os.path.join(self.root,imf1))
            imf2s_.append(os.path.join(self.root,imf2))

        if os.path.exists(pose_folder):
            f = open(pose_folder, 'r')
        for line in f:
            pos1, pos2 = line.strip().split(' ')
            pos1s_.append(os.path.join(self.root,pos1))
            pos2s_.append(os.path.join(self.root,pos2))

        imf1s.extend(imf1s_)
        imf2s.extend(imf2s_)
        pos1s.extend(pos1s_)
        pos2s.extend(pos2s_)

        return imf1s, imf2s, pos1s, pos2s 
    
    def read_default_pose(self, im_meta):
        '''return the default pose'''
        default_pose = np.asarray(np.load(im_meta))
        return default_pose

    @staticmethod
    def get_extrinsics(default_pose):
        '''change tranlation vector to be homogeneous -rcw as per Foundations of CV
        and the rotation matrix to have the axis direcitons as rows
        '''
        extrinsic = np.eye(4,4)
        extrinsic[:3,:3] = default_pose[:3,:3].T
        extrinsic[:3,3] = -default_pose[:3,:3].T @ default_pose[:3,3]
        return extrinsic
    @staticmethod
    def get_relative_pose(pose2, pose1):
        '''get the relative rotation and translation between two poses'''
        relPose = pose2 @ np.linalg.inv(pose1)
        # rot = relPose[:3,:3]
        # t = relPose[:3,3]
        return relPose

    
    def __getitem__(self, item):
        '''get item'''
        # gets the image path for the item/index
        # this comes from self.read_pairs, item is the index
        imf1 = self.imf1s[item]
        imf2 = self.imf2s[item]
        # gets the pose information
        # for the given path
        im1_pos = self.pos1s[item]
        im2_pos = self.pos2s[item]
        # reading images using numpy
        im1 = np.load(imf1)
        im2 = np.load(imf2)
        # flip images since the image has sensor origin at 
        # top of the image center, make it bottom center instead
        im1 = np.flip(im1)
        im2 = np.flip(im2)
        im1 = im1.copy()
        im2 = im2.copy()
        # get the shapes
        w, h = im1.shape[:2] # bearing, range, x,y, u,v
        extrinsic1 = self.get_extrinsics(self.read_default_pose(im1_pos))
        extrinsic2 = self.get_extrinsics(self.read_default_pose(im2_pos))

        # using the poses for each 
        # And then add the test
        relative = self.get_relative_pose(extrinsic2,extrinsic1)
        R = relative[:3, :3]
        T = relative[:3,3]
        
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
        
        if theta > 80 and self.phase == 'train':
            print("theta greater than 80")
            # return None

        # generate candidate query points
        coord1 = data_utils.generate_query_kpts(im1, self.args.train_kp, self.args.num_pts, self.args.akaze_pts, self.args.akaze_superpoint_pts, h, w)

        # if no keypoints are detected
        if (len(coord1)<self.args.num_pts):
            print("coord1 has points {} but needed {}".format(len(coord1),self.args.num_pts))

        # prune query keypoints that are not likely to have correspondence in the other image
        if self.args.prune_kp:
            ind_intersect = data_utils.prune_kpts(coord1, im2.shape[:2],
                                                  relative, d_min=4, d_max=400)
            if np.sum(ind_intersect) == 0:
                print("prune_kp")
            coord1 = coord1[ind_intersect]

        coord1 = torch.from_numpy(coord1).float()
        
        im1_ori, im2_ori = torch.from_numpy(im1), torch.from_numpy(im2)

        pose = torch.from_numpy(relative).float()
        im1_tensor = self.transform(im1)
        im2_tensor = self.transform(im2)
        
        #return a dict with all this information
        out = {'im1': im1_tensor,
               'im2': im2_tensor,
               'im1_ori': im1_ori,
               'im2_ori': im2_ori,
               'pose': pose,
               'T1': extrinsic1,
               'T2': extrinsic2,
               'coord1': coord1}

        for di in out.keys():
            if(out[di] is None):
                print(di)
        return out

    def __len__(self):
        '''returns the length '''
        return len(self.imf1s)
    
    def collate_fn(batch):
        batch = list(filter(lambda b: b is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


