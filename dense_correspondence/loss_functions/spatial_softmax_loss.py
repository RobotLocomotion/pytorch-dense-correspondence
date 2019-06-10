import torch
import numpy as np
from torch.autograd import Variable
import math
from numpy import linalg



def bilinear_interpolate_torch(im, x, y, cuda=True):
    """
    im shape: (H,W,D) # or can just do (H,W)
    x shape: (1,num_samples)
    y shape: (1,num_samples)
    """
    
    if cuda:
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        dtype_long = torch.LongTensor
   
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    Ia = im[ y0, x0 ][0]
    Ib = im[ y1, x0 ][0]
    Ic = im[ y0, x1 ][0]
    Id = im[ y1, x1 ][0]
    
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)


class SpatialSoftmaxLoss(object):

    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.setup_pixel_maps()
        self.debug_counter = 0

    def get_loss(self, image_a_pred, image_b_pred, matches_a, matches_b, img_a, depth_a, depth_b):
        """
        image_a_pred: N, D, H, W
        image_b_pred: N, D, H, W
        matches_a: N, num_matches
        matches_b: N, num_matches

        img_a: just for debug, N, 3, H, W

        """
        N, D, H, W = image_a_pred.shape
        num_matches = len(matches_a[0])

        matches_x_a = (matches_a%640).float() / 8.0 # convert to x,y coordinates in the downsampled image
        matches_y_a = (matches_a/640).float() / 8.0

        matches_x_b = (matches_b%640).float() / 8.0 # convert to x,y coordinates in the downsampled image
        matches_y_b = (matches_b/640).float() / 8.0

        def sample_descriptors(image_pred, matches_x, matches_y):
            descriptors = torch.zeros(N, num_matches, D).cuda()

            # self.ref_descriptor_vec is Nref, D 
            for i in range(N):
                one_image = image_pred[i,:]
                one_image = one_image.permute(1,2,0)
                one_matches_x = matches_x[i,:].unsqueeze(0)
                #print one_matches_x.shape
                one_matches_y = matches_y[i,:].unsqueeze(0)
                #print one_matches_y.shape
                #print torch.min(one_matches_x)
                #print torch.min(one_matches_y)
                #print torch.max(one_matches_x)
                #print torch.max(one_matches_y)
                descriptors[i,:] = bilinear_interpolate_torch(one_image, one_matches_x, one_matches_y)
            
            return descriptors
            
        # a_descriptors is N, num_matches, D
        a_descriptors = sample_descriptors(image_a_pred, matches_x_a, matches_y_a)
        b_descriptors = sample_descriptors(image_b_pred, matches_x_b, matches_y_b)

        def compute_softmax_activations(image_pred, descriptors):
                                               # image_pred starts N, D, H, W
            image_pred = image_pred.permute(0,2,3,1)            # N, H, W, D
            image_pred = image_pred.unsqueeze(3)                # N, H, W, 1,  D
            image_pred = image_pred.expand(N,H,W,num_matches,D) # N, H, W, nm, D 

            # descriptors starts out N, nm, D
            
            softmax_activations = torch.zeros(N, num_matches, H, W).cuda()

            for i in range(N):
                one_image = image_pred[i,:]           # H, W, nm, D
                deltas = one_image - descriptors[i] # H, W, nm, D
                neg_squared_norm_diffs = -1.0*torch.sum(torch.pow(deltas,2), dim=3) # H, W, nm
                neg_squared_norm_diffs = neg_squared_norm_diffs.permute(2,0,1).unsqueeze(0) # 1, nm, H, W

                neg_squared_norm_diffs_flat = neg_squared_norm_diffs.view(1, num_matches, H*W) # 1, nm, H*W
                softmax = torch.nn.Softmax(dim=2)
                softmax_activations[i] = softmax(neg_squared_norm_diffs_flat).view(N, num_matches, H, W).squeeze(0) # 1, nm, H, W

            return softmax_activations


        def compute_expected_x_y(softmax_activations):

            expected_x = torch.zeros(N, num_matches).cuda()
            expected_y = torch.zeros(N, num_matches).cuda()

            for i in range(N):
                one_expected_x = torch.sum(softmax_activations[i].unsqueeze(0)*self.pos_x, dim=(2,3)) # 1, nm
                one_expected_y = torch.sum(softmax_activations[i].unsqueeze(0)*self.pos_y, dim=(2,3)) # 1, nm
                expected_x[i,:] = one_expected_x
                expected_y[i,:] = one_expected_y

            return expected_x, expected_y

            # THIS BELOW MAYBE WORKS FULLY VECTORIZED
            # deltas = image_pred - descriptors.unsqueeze(1).unsqueeze(1).expand(N,H,W,num_matches,D)
            # squared_norm_diffs = torch.sum(torch.pow(deltas,2), dim=4)

        def compute_expected_z(softmax_activations, depth):
            """
            softmax_activations: N, nm, H, W
            depth: N, C=1, H, W
            """

            expected_z = torch.zeros(N, num_matches).cuda()
            downsampled_depth = torch.nn.functional.interpolate(depth, scale_factor=1.0/8, mode='bilinear', align_corners=True)
            # N, C=1, H/8, W/8

            for i in range(N):
                one_expected_z = torch.sum((softmax_activations[i]*downsampled_depth[i,0]).unsqueeze(0), dim=(2,3)) #1, nm
                expected_z[i,:] = one_expected_z
            
            return expected_z

        def get_match_depth(matches, depth):
            """
            matches: N, num_matches
            depth: N, C=1, H, W
            """
            depth_flattened = depth.view(N, 1, depth.shape[2]*depth.shape[3])          # N, C=1, W*H
            depth_flattened = depth_flattened.permute(0, 2, 1) # N, W*H, C=1

            depths = torch.zeros(N, num_matches).cuda()
            
            for i in range(N):
                one_depths = torch.index_select(depth_flattened[i].unsqueeze(0), 1, matches[i]).squeeze(2)
                depths[i] = one_depths

            return depths


        def vectorized_norm_pdf_multivariate_torch(x,mu):
            """
            everything is torch.FloatTensors
            for N samples, dimension dim
            x  shape: N, dim
            mu shape: N, dim
            
            returns shape: N
            """
            # this is for sigma = eye(4)
            size = len(x[0])
            
            det = 1.0

            norm_const = 1.0 / ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
            x_minus_mu = (x - mu)
            exp_factor = -0.5 * torch.sum(x_minus_mu*x_minus_mu,dim=1)
            return norm_const*torch.exp(exp_factor)

        def compute_l2_gaussian_divergence_loss(softmax_activations, norm_matches_x, norm_matches_y, pos_samples):
            loss_sum = 0

            loss = torch.nn.MSELoss()

            for i in range(N):
                one_softmax_activations = softmax_activations[i] # nm,H,W
                
                repeated_matches_x = norm_matches_x[0].repeat(H*W)
                repeated_matches_y = norm_matches_y[0].repeat(H*W)

                mu_samples = torch.stack((repeated_matches_x, repeated_matches_y)).permute(1,0) # Nm*H*W, 2

                pdf_samples = vectorized_norm_pdf_multivariate_torch(pos_samples, mu_samples)
                pdf_samples_shaped = pdf_samples.view(num_matches, H, W)
                loss_sum += loss(pdf_samples_shaped, one_softmax_activations)

            magic_weight = 0.001
            return loss_sum*magic_weight



        def convert_pixel_coords_to_norm_coords(matches_x, matches_y):
            norm_matches_x = (matches_x/80.0*2.0-1.0)
            norm_matches_y = (matches_y/60.0*2.0-1.0)
            return norm_matches_x, norm_matches_y

        def convert_norm_coords_to_orig_pixel_coords(norm_matches_x, norm_matches_y, to_cpu=True):
            matches_x = ((norm_matches_x/2.0 + 0.5)*640.0).detach().cpu().numpy()
            matches_y = ((norm_matches_y/2.0 + 0.5)*480.0).detach().cpu().numpy()
            return matches_x, matches_y
        

        norm_matches_x_a, norm_matches_y_a = convert_pixel_coords_to_norm_coords(matches_x_a, matches_y_a)
        norm_matches_x_b, norm_matches_y_b = convert_pixel_coords_to_norm_coords(matches_x_b, matches_y_b)

        softmax_activations_a = compute_softmax_activations(image_a_pred, b_descriptors)
        softmax_activations_b = compute_softmax_activations(image_b_pred, a_descriptors)

        pos_x_repeated = (self.pos_x.unsqueeze(2).expand(60,80,num_matches).contiguous().view(-1)/2.0+0.5)*640.0
        pos_y_repeated = (self.pos_y.unsqueeze(2).expand(60,80,num_matches).contiguous().view(-1)/2.0+0.5)*480.0
        pos_samples = torch.stack((pos_x_repeated.view(-1),pos_y_repeated.view(-1))).permute(1,0) # Nm*H*W, 2

        divergence_loss_a = compute_l2_gaussian_divergence_loss(softmax_activations_a, matches_x_a, matches_y_a, pos_samples)
        divergence_loss_b = compute_l2_gaussian_divergence_loss(softmax_activations_b, matches_x_b, matches_y_b, pos_samples)

        expected_x_a, expected_y_a = compute_expected_x_y(softmax_activations_a)
        expected_x_b, expected_y_b = compute_expected_x_y(softmax_activations_b)

        expected_z_a = compute_expected_z(softmax_activations_a, depth_a)
        expected_z_b = compute_expected_z(softmax_activations_b, depth_b)

        z_stars_a = get_match_depth(matches_a, depth_a)
        z_stars_b = get_match_depth(matches_b, depth_b)

        l_1 = torch.norm(norm_matches_x_a - expected_x_a, p=1)
        l_2 = torch.norm(norm_matches_x_b - expected_x_b, p=1)
        l_3 = torch.norm(norm_matches_y_a - expected_y_a, p=1)
        l_4 = torch.norm(norm_matches_y_b - expected_y_b, p=1)

        l_5 = torch.norm(z_stars_a - expected_z_a, p=1)
        l_6 = torch.norm(z_stars_b - expected_z_b, p=1)

        ## DEBUG
        
        if self.debug_counter % 100 == 0:
            import matplotlib.pyplot as plt
            numpy_img = img_a.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            plt.imshow(numpy_img)
            pix_matches_x_a, pix_matches_y_a = convert_norm_coords_to_orig_pixel_coords(norm_matches_x_a, norm_matches_y_a)
            pix_expected_x_a, pix_expected_y_a = convert_norm_coords_to_orig_pixel_coords(expected_x_a, expected_y_a)
            
            plt.scatter(pix_matches_x_a[0,0], pix_matches_y_a[0,0], color='green')
            plt.scatter(pix_expected_x_a[0,0], pix_expected_y_a[0,0], color='red')

            plt.show()
            numpy_img_pred = image_a_pred.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            if numpy_img_pred.shape[2] == 3:
                plt.imshow(numpy_img_pred)
                plt.show()
        self.debug_counter+=1

        loss = l_1 + l_2 + l_3 + l_4 + divergence_loss_a + divergence_loss_b #+ l_5 + l_6
        
        return loss

    def setup_pixel_maps(self):
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.W),
            np.linspace(-1., 1., self.H)
        )

        self.pos_x = torch.from_numpy(pos_x).float().cuda() # H, W
        self.pos_y = torch.from_numpy(pos_y).float().cuda() # H, w