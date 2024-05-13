''' Derived from the original codebase of CAPSNet'''

import os
import cv2
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import utils
import torchvision.utils as vutils
from SONIC.criterion import CtoFCriterion
from SONIC.network import SONICNet
import matplotlib.pyplot as plt
import wandb
import os
from utils import make_matching_figure
class SONICModel():

    def name(self):
        return 'CAPS Model'

    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # init model, optimizer, scheduler
        self.model = SONICNet(args, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        # reloading from checkpoints
        self.start_step = self.load_from_ckpt()


        #TODO: Confirm this working
        # define loss function
        self.criterion = CtoFCriterion(args).to(self.device)
    
    def set_input(self, data):
        # the two images
        self.im1 = Variable(data['im1'].to(self.device))
        self.im2 = Variable(data['im2'].to(self.device))
        
        # The coordinates for image 1 -> likely query points
        # fmatrix
        # camera pose
        # intrinsics
        # most of these needed for the loss function
        self.coord1 = Variable(data['coord1'].to(self.device))
        # self.fmatrix = data['F'].cuda()
        self.T1 = Variable(data['T1']).to(self.device)
        self.T2 = Variable(data['T2']).to(self.device)
        self.pose = Variable(data['pose'].to(self.device))
        # self.intrinsic1 = data['intrinsic1'].to(self.device)
        # self.intrinsic2 = data['intrinsic2'].to(self.device)

        # image orgins needed for what - check
        self.im1_ori = data['im1_ori']
        self.im2_ori = data['im2_ori']
        # self.batch_size = len(self.im1)
        self.imsize = self.im1.size()[2:]
    
    def forward(self):
        self.out = self.model.forward(self.im1, self.im2, self.coord1)
    
    def val_forward(self):
        self.out = self.model.forward(self.im1, self.im2, self.coord1)
        loss = self.criterion(self.coord1, self.out, self.T1, self.T2, self.pose, self.imsize, self.im1, self.im2)
        self.j_loss, self.eloss_c, self.eloss_f, self.closs_c, self.closs_f, self.std_loss = loss

    def backward_net(self):
        # need to fix the forward function for the loss function
        # OLD LOSS FORWARD CALL
        # loss = self.criterion(self.coord1, self.out, self.fmatrix, self.pose, self.imsize)
        # NEW LOSS FORWARD CALL
        loss = self.criterion(self.coord1, self.out, self.T1, self.T2, self.pose, self.imsize, self.im1, self.im2)
        # You can have multiple losses. When you do backward it will calculate gradients
        # based on those losses and then you can do optimizer.step() to update the weigths.
        self.j_loss, self.eloss_c, self.eloss_f, self.closs_c, self.closs_f, self.std_loss = loss
        self.j_loss.backward()
        # Her j_loss is the total loss and the other losses are the epipolara and cyclinc 
        # losses for the coarse and fine images
    
    def optimize_parameters(self):
        # This function does the training loop step by calling the individual functions
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        self.forward()
        self.backward_net()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.args.clip)
        self.optimizer.step()
        self.scheduler.step()

    def test(self):
        '''This function returns the coordinates of the descriptors given the
        2 images and the coordinates for the first image
        '''
        self.model.eval()
        with torch.no_grad():
            coord2_e, std = self.model.test(self.im1, self.im2, self.coord1)
        return coord2_e, std
    
    def extract_features(self, im, coord):
        '''
        Returns the coarse and fine feature descriptors for the image given the 
        coordinates
        '''
        self.model.eval()
        with torch.no_grad():
            feat_c, feat_f = self.model.extract_features(im, coord)
        return feat_c, feat_f
    
    def write_summary(self, n_iter,pbar = None, val_set=False, remove=False):
        
        description = f"{self.args.exp_name} | Step: {n_iter}, Loss: {self.j_loss.item():2.5f}"
        if val_set:
            description = f"{self.args.exp_name} | Val Step: {n_iter}, Val Loss: {self.j_loss.item():2.5f}"
            pbar.set_description(description)
            pbar.update(1)
            # print("%s | Val Step: %d, Val Loss: %2.5f" % (self.args.exp_name, n_iter, self.j_loss.item()))
            if n_iter % self.args.log_scalar_interval == 0:
                wandb.log({
                    'val_n_iter': n_iter,
                    'val_total_loss': self.j_loss.item(),
                    'val_epipolar_loss_coarse': self.eloss_c.item(),
                    'val_epipolar_loss_fine': self.eloss_f.item(),
                    'val_cycle_loss_coarse': self.closs_c.item(),
                    'val_cycle_loss_fine': self.closs_f.item(),
                })
            # write image
            if n_iter % self.args.log_img_interval == 0:
                num_kpts_display = self.args.num_kpts_shown
                if(num_kpts_display>self.coord1[0].shape[0]):
                    num_kpts_display = self.coord1[0].shape[0]
                im1_o = self.im1_ori[0].numpy()
                wandb_im1_o = wandb.Image(im1_o, caption=f"image 1 at {n_iter}")
                im2_o = self.im2_ori[0].numpy()
                wandb_im2_o = wandb.Image(im2_o, caption = f"image 2 at {n_iter}")
                kpt1 = self.coord1.cpu().numpy()[0][:num_kpts_display, :]
                # predicted correspondence
                correspondence = self.out['coord2_ef']
                kpt2 = correspondence.detach().cpu().numpy()[0][:num_kpts_display, :]
                # wandb.log({'val_kpt1_{}'.format(n_iter):kpt1, 'val_kpt2_{}'.format(n_iter):kpt2})
                # matching images given in utils
                local_path = 'test/debug_val_' + str(n_iter) + '.png' 
                make_matching_figure(im1_o,im2_o,kpt1,kpt2,path=local_path)
                # fig = make_matching_figure(im1_o,im2_o,kpt1,kpt2)
                wandb.log({'val_correspondence_image_{}'.format(n_iter): wandb.Image(local_path),
                        'val_image1_original_{}'.format(n_iter):wandb_im1_o,
                        'val_image2_original_{}'.format(n_iter):wandb_im2_o 
                        })
                if(remove):
                    os.remove(local_path)
            return
        # write scalar
        # print("%s | Step: %d, Loss: %2.5f" % (self.args.exp_name, n_iter, self.j_loss.item()))
        pbar.set_description(description)
        pbar.update(1)
        if n_iter % self.args.log_scalar_interval == 0:
            wandb.log({
                'n_iter': n_iter,
                'Total_loss': self.j_loss.item(),
                'epipolar_loss_coarse': self.eloss_c.item(),
                'epipolar_loss_fine': self.eloss_f.item(),
                'cycle_loss_coarse': self.closs_c.item(),
                'cycle_loss_fine': self.closs_f.item(),
            })

        # write image
        if n_iter % self.args.log_img_interval == 0:
            # this visualization shows a number of query points in the first image,
            # and their predicted correspondences in the second image,
            # the groundtruth epipolar lines for the query points are plotted in the second image
            num_kpts_display = self.args.num_kpts_shown
            if(num_kpts_display>self.coord1[0].shape[0]):
                num_kpts_display = self.coord1[0].shape[0]
            im1_o = self.im1_ori[0].numpy()
            wandb_im1_o = wandb.Image(im1_o, caption=f"image 1 at {n_iter}")
            im2_o = self.im2_ori[0].numpy()
            wandb_im2_o = wandb.Image(im2_o, caption = f"image 2 at {n_iter}")
            kpt1 = self.coord1.cpu().numpy()[0][:num_kpts_display, :]
            # predicted correspondence
            correspondence = self.out['coord2_ef']
            kpt2 = correspondence.detach().cpu().numpy()[0][:num_kpts_display, :]
            # wandb.log({'kpt1_{}'.format(n_iter):kpt1, 'kpt2_{}'.format(n_iter):kpt2})
            # matching images given in utils
            local_path = 'test/debug' + str(n_iter) + '.png' 
            make_matching_figure(im1_o,im2_o,kpt1,kpt2,path=local_path)
            # fig = make_matching_figure(im1_o,im2_o,kpt1,kpt2)
            wandb.log({'correspondence_image_{}'.format(n_iter): wandb.Image(local_path),
                      'image1_original_{}'.format(n_iter):wandb_im1_o,
                      'image2_original_{}'.format(n_iter):wandb_im2_o 
                     })
            if(remove):
                os.remove(local_path)
            # lines2 = cv2.computeCorrespondEpilines(kpt1.reshape(-1, 1, 2), 1, self.fmatrix[0].cpu().numpy())
            # lines2 = lines2.reshape(-1, 3)
            # im2_o, im1_o = utils.drawlines(im2_o, im1_o, lines2, kpt2, kpt1)
            # vis = np.concatenate((im1_o, im2_o), 1)
            # vis = torch.from_numpy(vis.transpose(2, 0, 1)).float().unsqueeze(0)
            # x = vutils.make_grid(vis, normalize=True)
            # writer.add_image('Image', x, n_iter)
    
    def load_model(self, filename):
        to_load = torch.load(filename)
        self.model.load_state_dict(to_load['state_dict'])
        if 'optimizer' in to_load.keys():
            self.optimizer.load_state_dict(to_load['optimizer'])
        if 'scheduler' in to_load.keys():
            self.scheduler.load_state_dict(to_load['scheduler'])
        return to_load['step']
    
    def load_from_ckpt(self):
        '''
        load model from existing checkpoints and return the current step
        :param ckpt_dir: the directory that stores ckpts
        :return: the current starting step
        '''

        # load from the specified ckpt path
        if self.args.ckpt_path != "":
            print("Reloading from {}".format(self.args.ckpt_path))
            if os.path.isfile(self.args.ckpt_path):
                step = self.load_model(self.args.ckpt_path)
            else:
                raise Exception('no checkpoint found in the following path:{}'.format(self.args.ckpt_path))

        else:
            ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
            os.makedirs(ckpt_folder, exist_ok=True)
            # load from the most recent ckpt from all existing ckpts
            ckpts = [os.path.join(ckpt_folder, f) for f in sorted(os.listdir(ckpt_folder)) if f.endswith('.pth')]
            if len(ckpts) > 0:
                fpath = ckpts[-1]
                step = self.load_model(fpath)
                print('Reloading from {}, starting at step={}'.format(fpath, step))
            else:
                print('No ckpts found, training from scratch...')
                step = 0

        return step

    def save_model(self, step):
        ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
        os.makedirs(ckpt_folder, exist_ok=True)

        save_path = os.path.join(ckpt_folder, "{:06d}.pth".format(step))
        print('saving ckpts {}...'.format(save_path))
        torch.save({'step': step,
                    'state_dict': self.model.state_dict(),
                    'optimizer':  self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    },
                   save_path)