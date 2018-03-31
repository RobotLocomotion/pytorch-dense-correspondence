import torch
from PIL import Image
import numpy as np
import correspondence_plotter
import matplotlib.pyplot as plt

class PixelwiseContrastiveLoss():

    def __init__(self):
    	self.type = "pixelwise_contrastive"
        self.debug = True

    def get_loss(self, image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b):
    	loss = 0

        if self.debug:
            img_a_pred_numpy = self.convert_to_plottable_numpy(image_a_pred)
            img_b_pred_numpy = self.convert_to_plottable_numpy(image_b_pred)
            plt.imshow(img_a_pred_numpy)
            plt.show()
            plt.imshow(img_b_pred_numpy)
            plt.show()
            exit(0)

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
        loss += torch.max(zeros_vec, pixel_wise_loss).sum()/150.0 # need to sync this later with num_non_matches
        non_match_loss = loss.data[0] - match_loss

        return loss, match_loss, non_match_loss


    def convert_to_plottable_numpy(self, img_torch_variable):
        return img_torch_variable.data.squeeze(0).contiguous().view(480,640,3).cpu().numpy()