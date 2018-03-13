import torch
from torch.autograd import Variable

class SemanticConsistencyLoss():

    def __init__(self):
    	self.type = "semantic_consistency"

    def get_loss(self, image_a_pred, image_b_pred, mask_a, mask_b):
        loss = 0

        # get the nonzero indices
        mask_a_indices_flat = torch.nonzero(mask_a)
        mask_b_indices_flat = torch.nonzero(mask_b)
        if len(mask_a_indices_flat) == 0:
            return Variable(torch.cuda.LongTensor([0]), requires_grad=True)
        if len(mask_b_indices_flat) == 0:
            return Variable(torch.cuda.LongTensor([0]), requires_grad=True)

        # take 5000 random pixel samples of the object, using the mask
        num_samples = 10000

        rand_numbers_a = (torch.rand(num_samples)*len(mask_a_indices_flat)).cuda()
        rand_indices_a = Variable(torch.floor(rand_numbers_a).type(torch.cuda.LongTensor), requires_grad=False)
        randomized_mask_a_indices_flat = torch.index_select(mask_a_indices_flat, 0, rand_indices_a).squeeze(1)

        rand_numbers_b = (torch.rand(num_samples)*len(mask_b_indices_flat)).cuda()
        rand_indices_b = Variable(torch.floor(rand_numbers_b).type(torch.cuda.LongTensor), requires_grad=False)
        randomized_mask_b_indices_flat = torch.index_select(mask_b_indices_flat, 0, rand_indices_b).squeeze(1)

        # index into the image and get descriptors
        M_margin = 0.5 # margin parameter
        random_img_a_object_descriptors = torch.index_select(image_a_pred, 1, randomized_mask_a_indices_flat)
        random_img_b_object_descriptors = torch.index_select(image_b_pred, 1, randomized_mask_b_indices_flat)
        pixel_wise_loss = (random_img_a_object_descriptors - random_img_b_object_descriptors).pow(2).sum(dim=2)
        pixel_wise_loss = torch.add(pixel_wise_loss, -2*M_margin)
        zeros_vec = torch.zeros_like(pixel_wise_loss)
        loss += torch.max(zeros_vec, pixel_wise_loss).sum()

        return loss