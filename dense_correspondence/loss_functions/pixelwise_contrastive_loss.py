import torch
from torch.autograd import Variable
class PixelwiseContrastiveLoss():

    def __init__(self):
    	self.type = "pixelwise_contrastive"
        self.image_width = 640
        self.image_height = 480 # maybe set this elsewhere

    def get_loss(self, image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b,
                 M_margin=0.5, non_match_loss_weight=1.0):
        """
        Computes the loss function

        DCN = Dense Correspondence Network
        num_images = number of images in this batch
        num_matches = number of matches
        num_non_matches = number of non-matches
        W = image width
        H = image height
        D = descriptor dimension


        match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2^2 )

        loss = match_loss + non_match_loss

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :type non_matches_b:
        :return: loss, match_loss, non_match_loss
        :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
        """

        num_matches = matches_a.size()[0]
        num_non_matches = non_matches_a.size()[0]

        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)


        match_loss = 1.0/num_matches * (matches_a_descriptors - matches_b_descriptors).pow(2).sum()

        # add loss via non_matches
        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)
        pixel_wise_loss = (non_matches_a_descriptors - non_matches_b_descriptors).pow(2).sum(dim=2)
        pixel_wise_loss = torch.add(torch.neg(pixel_wise_loss), M_margin)
        print pixel_wise_loss.shape, "is pixel_wise_loss.shape"
        pixel_wise_loss = pixel_wise_loss * self.l2_pixel_loss(matches_b, non_matches_b)
        print pixel_wise_loss.shape, "is pixel_wise_loss.shape"
        zeros_vec = torch.zeros_like(pixel_wise_loss)
        non_match_loss = non_match_loss_weight * 1.0/num_non_matches * torch.max(zeros_vec, pixel_wise_loss).sum()

        loss = match_loss + non_match_loss

        return loss, match_loss, non_match_loss

    def l2_pixel_loss(self, matches_b, non_matches_b):
        """
        Apply l2 loss in pixel space.

        This weights non-matches more if they are "far away" in pixel space.

        :param matches_b: A torch.LongTensor of shape (num_matches)
        :param non_matches_b: A torch.LongTensor of shape (num_non_matches)
        :return l2 loss per sample: A torch.FloatTensorof shape (num_non_matches,1)
        """
        print len(matches_b)
        print len(non_matches_b)

        num_non_matches_per_match = len(non_matches_b)/len(matches_b)

        ground_truth_pixels_for_non_matches_b = torch.t(matches_b.repeat(num_non_matches_per_match,1)).contiguous().view(-1,1)
        print "ground_truth_pixels_for_non_matches_b"
        print len(ground_truth_pixels_for_non_matches_b)
        print type(ground_truth_pixels_for_non_matches_b)

        print ground_truth_pixels_for_non_matches_b.shape, "is ground_truth_pixels_for_non_matches_b shape"
        print non_matches_b.shape, "is non_matches_b.shape"

        ground_truth_u_v_b = self.flattened_pixel_locations_to_u_v(ground_truth_pixels_for_non_matches_b)
        sampled_u_v_b      = self.flattened_pixel_locations_to_u_v(non_matches_b.unsqueeze(1))
        print ground_truth_u_v_b.shape, "is ground_truth_u_v_b.shape"
        print sampled_u_v_b.shape, "is sampled_u_v_b.shape"

        squared_l2_pixel_loss = (ground_truth_u_v_b - sampled_u_v_b).float().pow(2).sum(dim=1)
        print squared_l2_pixel_loss.shape
        return squared_l2_pixel_loss
        

    
    def flattened_pixel_locations_to_u_v(self, flat_pixel_locations):
        """
        :param flat_pixel_locations: A torch.LongTensor of shape (n,1) where each element is a flattened pixel index, i.e.
                                        some integer between 0 and 307,200 for a 640x480 image
        :type flat_pixel_locations: torch.LongTensor

        :return A torch.LongTensor of shape (n,2) where the first column is the u coordinates of the pixel
                                        and the second column is the v coordinate                                        

        """
        u_v_pixel_locations = flat_pixel_locations.repeat(1,2)
        u_v_pixel_locations[:,0] = u_v_pixel_locations[:,0]%self.image_width 
        u_v_pixel_locations[:,1] = u_v_pixel_locations[:,1]/self.image_width
        return u_v_pixel_locations