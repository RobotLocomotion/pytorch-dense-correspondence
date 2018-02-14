import torch

class PixelwiseContrastiveLoss():

    def __init__(self):
    	self.type = "pixelwise_contrastive"

    def get_loss(self, image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b):
    	loss = 0

    	# add loss via matches
        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)
        loss += (matches_a_descriptors - matches_b_descriptors).pow(2).sum()
        match_loss = 1.0*loss.data[0]
        
        # add loss via non_matches
        M_margin = 0.5 # margin parameter
        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)
        pixel_wise_loss = (non_matches_a_descriptors - non_matches_b_descriptors).pow(2).sum(dim=2)
        pixel_wise_loss = torch.add(torch.neg(pixel_wise_loss), M_margin)
        zeros_vec = torch.zeros_like(pixel_wise_loss)
        loss += torch.max(pixel_wise_loss, zeros_vec).sum()/100.0 # need to sync this later with num_non_matches
        non_match_loss = loss.data[0] - match_loss

        return loss, match_loss, non_match_loss