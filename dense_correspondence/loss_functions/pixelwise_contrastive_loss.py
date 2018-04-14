import torch
from torch.autograd import Variable
class PixelwiseContrastiveLoss():

    def __init__(self):
    	self.type = "pixelwise_contrastive"

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


        match_loss = 1/num_matches \sum_m ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_m max(0, M_margin - ||descriptor_a - descriptor_b||_2^2 )

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
        zeros_vec = torch.zeros_like(pixel_wise_loss)
        non_match_loss = non_match_loss_weight * 1.0/num_non_matches * torch.max(zeros_vec, pixel_wise_loss).sum()

        loss = match_loss + non_match_loss

        return loss, match_loss, non_match_loss