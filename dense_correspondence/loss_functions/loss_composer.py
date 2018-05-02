from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, SpartanDatasetDataType
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss

import torch
from torch.autograd import Variable

def get_loss(pixelwise_contrastive_loss, match_type, 
              image_a_pred, image_b_pred,
              matches_a,     matches_b,
              masked_non_matches_a, masked_non_matches_b,
              background_non_matches_a, background_non_matches_b,
              blind_non_matches_a, blind_non_matches_b):
    """
    This function serves the purpose of:
    - parsing the different types of SpartanDatasetDataType...
    - parsing different types of matches / non matches..
    - into different pixelwise contrastive loss functions

    :return args: loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss
    :rtypes: each pytorch Variables

    """
    if (match_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE).all():
        print "applying SINGLE_OBJECT_WITHIN_SCENE loss"
        return get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            matches_a,    matches_b,
                                            masked_non_matches_a, masked_non_matches_b,
                                            background_non_matches_a, background_non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)

    if (match_type == SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE).all():
        print "applying SINGLE_OBJECT_ACROSS_SCENE loss"
        return get_same_object_across_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            blind_non_matches_a, blind_non_matches_b)

    if (match_type == SpartanDatasetDataType.DIFFERENT_OBJECT).all():
        print "applying DIFFERENT_OBJECT loss"
        return get_different_object_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            blind_non_matches_a, blind_non_matches_b)


    if (match_type == SpartanDatasetDataType.MULTI_OBJECT).all():
        print "applying MULTI_OBJECT loss"
        return get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                            matches_a,    matches_b,
                                            masked_non_matches_a, masked_non_matches_b,
                                            background_non_matches_a, background_non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)

    else:
        raise ValueError("Should only have above scenes?")


def get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                        matches_a,    matches_b,
                                        masked_non_matches_a, masked_non_matches_b,
                                        background_non_matches_a, background_non_matches_b,
                                        blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """

    _, match_loss, masked_non_match_loss =\
        pixelwise_contrastive_loss.get_loss_matched_and_non_matched_with_l2(image_a_pred,         image_b_pred,
                                                                          matches_a,            matches_b,
                                                                          masked_non_matches_a, masked_non_matches_b)

    # background_non_match_loss =\
    #     pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
    #                                                             background_non_matches_a, background_non_matches_b,
    #                                                             M_descriptor=0.5)
    


    background_non_match_loss =\
        pixelwise_contrastive_loss.non_match_loss_with_l2_pixel_norm(image_a_pred, image_b_pred, matches_b, background_non_matches_a, background_non_matches_b)    
    

    num_masked_non_matches = len(masked_non_matches_a)
    num_background_non_matches = len(background_non_matches_a)
    total_num_non_matches = num_masked_non_matches + num_background_non_matches

    weight_masked = num_masked_non_matches*1.0/total_num_non_matches
    weight_background = num_background_non_matches*1.0/total_num_non_matches
        
    loss = match_loss + \
    pixelwise_contrastive_loss._config["non_match_loss_weight"] * \
    (weight_masked *  masked_non_match_loss + weight_background * background_non_match_loss)
    


    blind_non_match_loss = zero_loss()
    # if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
    #     blind_non_match_loss =\
    #         pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
    #                                                                 blind_non_matches_a, blind_non_matches_b,
    #                                                                 M_descriptor=0.5)
    #     loss += blind_non_match_loss

    return loss, match_loss, masked_non_match_loss, background_non_match_loss, blind_non_match_loss

def get_different_object_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                              blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    blind_non_match_loss = zero_loss()
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        blind_non_match_loss =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=0.5)
    loss = blind_non_match_loss
    return loss, zero_loss(), zero_loss(), zero_loss(), blind_non_match_loss

def get_same_object_across_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                              blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    blind_non_match_loss = zero_loss()
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        blind_non_match_loss =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=0.5, invert=True)
    loss = blind_non_match_loss
    return loss, zero_loss(), zero_loss(), zero_loss(), blind_non_match_loss

def zero_loss():
    return Variable(torch.FloatTensor([0]).cuda())


