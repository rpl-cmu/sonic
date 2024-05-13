''' Derived from the original codebase of CAPSNet'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

class SONICNet(nn.Module):
    def __init__(self, args, device):
        super(SONICNet, self).__init__()
        self.args = args
        self.device = device
        self.net = ResUNet(pretrained=args.pretrained,
                            encoder=args.backbone,
                            coarse_out_ch=args.coarse_feat_dim,
                            fine_out_ch=args.fine_feat_dim).to(self.device)
    
    @staticmethod
    def normalize(coord, h, w):
        '''
        turn the coordinates from pixel indices to the range of [-1, 1]
        :param coord: [..., 2]
        :param h: the image height
        :param w: the image width
        :return: the normalized coordinates [..., 2]
        '''
        
        # create a tensor {width_mean, height_mean}
        c = torch.Tensor([(w-1)/2., (h-1)/2.]).to(coord.device).float()
        coord_norm = (coord - c) / c
        return coord_norm
    
    @staticmethod
    def denormalize(coord_norm, h, w):
        '''
        turn the coordinates from normalized value ([-1, 1]) to actual pixel indices
        :param coord_norm: [..., 2]
        :param h: the image height
        :param w: the image width
        :return: actual pixel coordinates
        '''
        c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord_norm.device)
        coord = coord_norm * c + c
        return coord
    
    def ind2coord(self, ind, width):
        # converting index to coordinates
        # using the old divide by width and remainder trick
        # again this is just the coord
        ind = ind.unsqueeze(-1)
        x = ind%width
        # y = ind // width
        y = torch.div(ind, width, rounding_mode='trunc')
        coord = torch.cat((x,y),-1).float()
        return coord
    
    def gen_grid(self, h_min, h_max, w_min, w_max, len_h, len_w):
        # Creates a grid b/w h_min, h_max, w_min, w_max
        # with num_samples along each side len_h, len_w
        x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w), torch.linspace(h_min, h_max, len_h)])
        grid = torch.stack((x, y), -1).transpose(0, 1).reshape(-1, 2).float().to(self.device)
        return grid
    
    def sample_feat_by_coord(self,x, coord_n, norm = False):
        '''
        sample from normalized coordinates
        :param x: feature map [batch_size, n_dim, h, w]
        :param coord_n: normalized coordinates, [batch_size, n_pts, 2]
        :param norm: if l2 normalize features
        :return: the extracted features, [batch_size, n_pts, n_dim]
        '''
        # Interpolation here
        # For the coordinate locations find the features from feature_map x 
        feat = F.grid_sample(x, coord_n.unsqueeze(2)).squeeze(-1)
        if norm:
            feat = F.normalize(feat)
        feat = feat.transpose(1,2)
        return feat
    
    def compute_prob(self,feat1, feat2):
        '''
        compute probability
        :param feat1: query features, [batch_size, m, n_dim]
        :param feat2: reference features, [batch_size, n, n_dim]
        :return: probability, [batch_size, m, n]
        '''
        # For every query decriptor in Img1 we find the closeness across Img2
        # p(x|x1,M1,M2) = exp(M_1(x_1).T @ M_2(x)) / sum_y (exp(M_1(x_1).T @ M_2(y)))
        assert self.args.prob_from in ['correlation', 'distance']
        if self.args.prob_from == 'correlation':
            # similarity
            sim = feat1.bmm(feat2.transpose(1, 2))
            # final probability
            prob = F.softmax(sim, dim=-1)  # Bxmxn
        else:
            dist = torch.sum(feat1**2, dim=-1, keepdim=True) + \
                   torch.sum(feat2**2, dim=-1, keepdim=True).transpose(1, 2) - \
                   2 * feat1.bmm(feat2.transpose(1, 2))
            prob = F.softmax(-dist, dim=-1)  # Bxmxn
        return prob
    
    def get_1nn_coord(self, feat1, featmap2):
        '''
        find the coordinates of nearest neighbor match
        :param feat1: query features, [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the other image
        :return: normalized correspondence locations [batch_size, n_pts, 2]
        '''
        # This function finds the closest using max similarity
        # or the least distance
        batch_size, d, h, w = featmap2.shape
        feat2_flatten = featmap2.reshape(batch_size, d, h*w).transpose(1, 2)  # Bx(hw)xd

        assert self.args.prob_from in ['correlation', 'distance']
        if self.args.prob_from == 'correlation':
            sim = feat1.bmm(feat2_flatten.transpose(1, 2))
            ind2_1nn = torch.max(sim, dim=-1)[1]
        else:
            dist = torch.sum(feat1**2, dim=-1, keepdim=True) + \
                   torch.sum(feat2_flatten**2, dim=-1, keepdim=True).transpose(1, 2) - \
                   2 * feat1.bmm(feat2_flatten.transpose(1, 2))
            ind2_1nn = torch.min(dist, dim=-1)[1]

        coord2 = self.ind2coord(ind2_1nn, w)
        coord2_n = self.normalize(coord2, h, w)
        return coord2_n
    
    def get_expected_correspondence_locs(self, feat1, featmap2, with_std=False):
        '''
        compute the expected correspondence locations
        :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
        :param with_std: if return the standard deviation
        :return: the normalized expected correspondence locations[batch_size, n_pts, 2]
        '''
    
        B, d, h2, w2 = featmap2.size()
        # create a grid b/w -1 to 1 in x & y of dims h2, w2
        grid_n = self.gen_grid(-1, 1, -1, 1, h2, w2)
        # We convert the 2d into indices baseically each cell is an index
        # And the index corresponds to a feature descriptor of size d
        featmap2_flatten = featmap2.reshape(B,d,h2*w2).transpose(1,2) #Bx(hw)xd
        # Use that to calculate the probability for each query_point(total n_pts)
        # For earch query point we get h*w number of probabilities denoting the 
        # probability at each index
        prob = self.compute_prob(feat1, featmap2_flatten) # Bxnxhw

        grid_n = grid_n.unsqueeze(0).unsqueeze(0) # 1x1x(hw)x2
        # For each point we get the expected coordinates
        expected_coord_n = torch.sum(grid_n * prob.unsqueeze(-1), dim = 2) #Bxnx2

        if with_std:
            # convert to normalized scale [-1,1]
            var = torch.sum(grid_n**2 * prob.unsqueeze(-1), dim=2) - expected_coord_n**2  # Bxnx2
            std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
            return expected_coord_n, std
        else:
            return expected_coord_n

    def get_expected_correspondence_within_window(self,feat1, featmap2, coord2_n, with_std=False):
        '''
        :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
        :param coord2_n: normalized center loctions [batch_size, n_pts, 2]
        :param with_std: if return the standard deviation
        :return: the normalized expected correspondence locations, [batch_size, n_pts, 2], optionally with std
        '''
        batch_size, n_dim, h2, w2 = featmap2.shape
        # total number of query points
        n_pts = coord2_n.shape[1]
        # create a large grid
        grid_n = self.gen_grid(h_min=-self.args.window_size, h_max=self.args.window_size,
                               w_min=-self.args.window_size, w_max=self.args.window_size,
                               len_h=int(self.args.window_size*h2), len_w=int(self.args.window_size*w2))
        # repeat batch_size number of times
        grid_n_ = grid_n.repeat(batch_size, 1, 1, 1)  # Bx1xhwx2

        coord2_n_grid = coord2_n.unsqueeze(-2) + grid_n_  # Bxnxhwx2
        # Finds the feature descriptors for the query points in the window
        feat2_win = F.grid_sample(featmap2, coord2_n_grid, padding_mode='zeros').permute(0, 2, 3, 1)  # Bxnxhwxd
        feat1 = feat1.unsqueeze(-2)
        prob = self.compute_prob(feat1.reshape(batch_size*n_pts, -1, n_dim),
                                 feat2_win.reshape(batch_size*n_pts, -1, n_dim)).reshape(batch_size, n_pts, -1)

        expected_coord2_n = torch.sum(coord2_n_grid * prob.unsqueeze(-1), dim=2)  # Bxnx2

        if with_std:
            var = torch.sum(coord2_n_grid**2 * prob.unsqueeze(-1), dim=2) - expected_coord2_n**2  # Bxnx2
            std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
            return expected_coord2_n, std
        else:
            return expected_coord2_n

    def forward(self, im1, im2, coord1):
 
        xc1, xf1 = self.net(im1)
        xc2, xf2 = self.net(im2)

        #image width and height
        h1i, w1i = im1.size()[2:]
        h2i, w2i = im2.size()[2:]
        
        coord1_n = self.normalize(coord1, h1i, w1i)
        # find the feature descriptors for the coarse feature locations which
        # are determined by the normalized query points coord1_n
        feat1_coarse = self.sample_feat_by_coord(xc1, coord1_n) #Bxnxd
        # Get the expected correspondence points in img2 given the query point descriptors from img1
        coord2_ec_n, std_c = self.get_expected_correspondence_locs(feat1_coarse, xc2, with_std=True)

        # the center locations  of the local window for fine level computation
        # Based on the whether we use expected or nearest neighbor
        coord2_ec_n_ = self.get_1nn_coord(feat1_coarse, xc2) if self.args.use_nn else coord2_ec_n
        # Get the feature descriptors for the fine image using the normalized query coordinates
        feat1_fine = self.sample_feat_by_coord(xf1, coord1_n) #Bxnxd
        # coord2_ec_n_ used for the window
        coord2_ef_n, std_f = self.get_expected_correspondence_within_window(feat1_fine, xf2,
                                                                            coord2_ec_n_, with_std=True)
        
        # For the cyclic loss finding the correspondences of the predicted correspondences
        feat2_coarse = self.sample_feat_by_coord(xc2, coord2_ec_n_)
        coord1_lc_n, std_lc = self.get_expected_correspondence_locs(feat2_coarse, xc1, with_std=True)

        feat2_fine = self.sample_feat_by_coord(xf2, coord2_ef_n)  # Bxnxd
        coord1_lf_n, std_lf = self.get_expected_correspondence_within_window(feat2_fine, xf1,
                                                                             coord1_n, with_std=True)
        
        # Denormalizing the coordinates
        coord2_ec = self.denormalize(coord2_ec_n, h2i, w2i)
        coord2_ef = self.denormalize(coord2_ef_n, h2i, w2i)
        coord1_lc = self.denormalize(coord1_lc_n, h1i, w1i)
        coord1_lf = self.denormalize(coord1_lf_n, h1i, w1i)
        # need the (h,w) dimensions of coarse and fine features
        if(torch.isnan(coord1_lc.mean())):
            print(xc1) 

        return {'coord2_ec': coord2_ec, 'coord2_ef': coord2_ef,
                'coord1_lc': coord1_lc, 'coord1_lf': coord1_lf,
                'std_c': std_c, 'std_f': std_f,
                'std_lc': std_lc, 'std_lf': std_lf,
                'coarse_h':xc1.shape[2], 'coarse_w':xc2.shape[3],
                'fine_h':xf1.shape[2], 'fine_w':xf2.shape[3]
                }
    
    def extract_features(self, im, coord):
        '''
        extract coarse and fine level features given the input image and 2d locations
        :param im: [batch_size, 3, h, w]
        :param coord: [batch_size, n_pts, 2]
        :return: coarse features [batch_size, n_pts, coarse_feat_dim] and fine features [batch_size, n_pts, fine_feat_dim]
        '''
        xc, xf = self.net(im)
        hi, wi = im.size()[2:]
        coord_n = self.normalize(coord, hi, wi)
        feat_c = self.sample_feat_by_coord(xc, coord_n)
        feat_f = self.sample_feat_by_coord(xf, coord_n)
        return feat_c, feat_f
    
    def test(self, im1, im2, coord1):
        '''
        given a pair of images im1, im2, compute the coorrespondences for query points coord1.
        We performa full image search at coarse level and local search at fine level
        :param im1: [batch_size, 3, h, w]
        :param im2: [batch_size, 3, h, w]
        :param coord1: [batch_size, n_pts, 2]
        :return: the fine level correspondence location [batch_size, n_pts, 2]
        '''
        xc1, xf1 = self.net(im1)
        xc2, xf2 = self.net(im2)

        h1i, w1i = im1.shape[2:]
        h2i, w2i = im2.shape[2:]

        coord1_n = self.normalize(coord1, h1i, w1i)
        feat1_c = self.sample_feat_by_coord(xc1, coord1_n)
        _, std_c = self.get_expected_correspondence_locs(feat1_c, xc2, with_std=True)

        coord2_ec_n = self.get_1nn_coord(feat1_c, xc2)
        feat1_f = self.sample_feat_by_coord(xf1, coord1_n)
        _, std_f = self.get_expected_correspondence_within_window(feat1_f, xf2, coord2_ec_n, with_std=True)

        coord2_ef_n = self.get_1nn_coord(feat1_f, xf2)
        coord2_ef = self.denormalize(coord2_ef_n, h2i, w2i)
        std = (std_c + std_f)/2

        return coord2_ef, std


##################### ResUnet
# Main network
def class_for_name(module_name, class_name):
    #load the module, will raise ImportError if module can't be loaded
    # For loading the resnet encoders
    m = importlib.import_module(module_name)
    return getattr(m, class_name)

class conv(nn.Module):
    #conv layer that will be used to get coarse, fine
    # and after skip connect also used in upconv
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride) -> None:
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers, num_out_layers, kernel_size = kernel_size,
                            stride = stride, padding = (self.kernel_size-1) // 2)
        self.bn = nn.BatchNorm2d(num_out_layers)
    
    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)

class upconv(nn.Module):
    # upconv/upsample as part of the unet type architecture
    # use bilinear interpolation for upsampling given the scale
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale) -> None:
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        ## nn.interpolate here scales up the h,w
        x = nn.functional.interpolate(x, scale_factor = self.scale, align_corners = True, mode = 'bilinear')
        ## conv will convert from num_in_layers to num_out_layers
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self,
                encoder = 'resnet50',
                pretrained = True,
                # out channels for the coarse level
                coarse_out_ch = 128,
                # out channels for the final fine level
                fine_out_ch = 128
                ) -> None:
        super(ResUNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64,128,256,512]
        else:
            filters = [256, 512, 1024, 2048]
        # get pretrained resnet encoder    
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)
        
        self.firstconv = resnet.conv1 #H/2
        #batch_norm
        self.firstbn = resnet.bn1
        # relu
        self.firstrelu = resnet.relu
        # maxpool
        self.firstmaxpool = resnet.maxpool #H/4

        #encoder
        self.layer1 = resnet.layer1 #H/4
        self.layer2 = resnet.layer2 #H/8
        self.layer3 = resnet.layer3 #H/16

        #coarse-level conv
        self.conv_coarse = conv(filters[2], coarse_out_ch, 1 ,1)

        #decoder
        self.upconv3 = upconv(filters[2], 512, 3, 2)
        # Will be used after concat based skip-connect
        self.iconv3 = conv(filters[1]+512, 512, 3, 1)
        # up conv again after the previous skip connect and conv
        self.upconv2 = upconv(512,256,3,2)
        # after skip connect
        self.iconv2 = conv(filters[0]+256, 256, 3, 1)

        #fine-level conv
        self.conv_fine = conv(256, fine_out_ch, 1, 1)
    
    def skipconnect(self,x1, x2):
        # Find the diffence in H, W of x2 & x1
        # Pad that difference onto x1 so they
        # are of the same size
        # x1 is from down sampling, x2 from upsampling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX//2, diffX - diffX //2,
                        diffY//2, diffY - diffY//2))
        # why dim =1 ?
        x = torch.cat([x2, x1],dim = 1)
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x

    def forward(self, x):
        '''SONAR Images
        Input (id: dimension)     Layer                         Output (id: dimension)
        0: 512 × 512 × 1          7 × 7 Conv, 64, stride 2      1: 256 × 256 × 64
        1: 256 × 256 × 64         3 × 3 MaxPool, stride 2       2: 128 × 128 × 64
        2: 128 × 128 × 64         Residual Block 1              3: 128 × 128 × 256
        3: 128 × 128 × 256        Residual Block 2              4: 64 × 64 × 512
        4: 64 × 64 × 512          Residual Block 3              5: 32 × 32 × 1024
        5: 32 × 32 × 1024         1 × 1 Conv, 128               Coarse: 32 × 32 × 128
        5: 32 × 32 × 1024         3 × 3 Upconv, 512, factor 2   6: 64 × 64 × 512
        [4, 6]: 64 × 64 × 1024    3 × 3 Conv, 512               7: 64 × 64 × 512
        7: 64 × 64 × 512          3 × 3 Upconv, 256, factor 2   8: 128 × 128 × 256
        [3, 8]: 128 × 128 × 512   3 × 3 Conv, 256               9: 128 × 128 × 256
        9: 128 × 128 × 256        1 × 1 Conv, 128               Fine: 128 × 128 × 128
        '''
        # hacky way of copying to 3 channels because resnet has copied weights
        # need double but found float
        input_val = x
        x = x.repeat(1,3,1,1)
        x = x.to(torch.float)
    
        x = self.firstrelu(self.firstbn(self.firstconv(x)))
        x = self.firstmaxpool(x)
        if(not torch.isnan(x.mean())):
            self.firstconv_grad = self.firstconv.weight.grad
       
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_coarse = self.conv_coarse(x3)

        #starting upsampling
        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_fine = self.conv_fine(x)
        if(torch.isnan(x_fine.mean()) or torch.isnan(x_coarse.mean())):
            print(x_fine)

        return [x_coarse, x_fine]

