import torch

from PIL import Image
import numpy as np
import correspondence_plotter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation

class PixelwiseContrastiveLoss():

    def __init__(self):
    	self.type = "pixelwise_contrastive"
        self.debug = False
        self.counter = 0

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
        loss += torch.max(zeros_vec, pixel_wise_loss).sum()/150.0 # need to sync this later with num_non_matches
        non_match_loss = loss.data[0] - match_loss

        return loss, match_loss, non_match_loss

    def get_adversarial_loss(self, image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b):

        if (self.counter % 20 == 0):
            self.debug = True
        else:
            self.debug = False
        self.counter += 1

        if self.debug:
            img_a_pred_numpy = self.convert_to_plottable_numpy(image_a_pred)
            img_b_pred_numpy = self.convert_to_plottable_numpy(image_b_pred)
            fig, axes = DenseCorrespondenceEvaluation.plot_descriptor_colormaps(img_a_pred_numpy, img_b_pred_numpy)
            
            first_match_a = self.flattened_pixel_location_to_plottable_pixel_location(matches_a[0])
            circ = Circle(first_match_a, radius=10, facecolor='g', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
            axes[0].add_patch(circ)
            
            # don't need to add non match from a side, it's the same as match
            #first_non_match_a = self.flattened_pixel_location_to_plottable_pixel_location(non_matches_a[0])
            #circ = Circle(first_non_match_a, radius=10, facecolor='r', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
            #axes[0].add_patch(circ)


            first_match_b = self.flattened_pixel_location_to_plottable_pixel_location(matches_b[0])
            circ = Circle(first_match_b, radius=10, facecolor='g', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid', alpha=0.8)
            axes[1].add_patch(circ)
            first_non_match_b = self.flattened_pixel_location_to_plottable_pixel_location(non_matches_b[0])
            circ = Circle(first_non_match_b, radius=10, facecolor='r', edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
            axes[1].add_patch(circ)

            plt.show()

        return self.get_loss(image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b)


    def convert_to_plottable_numpy(self, img_torch_variable):
        return img_torch_variable.data.squeeze(0).contiguous().view(480,640,3).cpu().numpy()

    def flattened_pixel_location_to_plottable_pixel_location(self, flat_pixel_location):
        return (flat_pixel_location%640, flat_pixel_location/640)