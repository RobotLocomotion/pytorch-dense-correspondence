import torch
import dense_correspondence_manipulation.utils.utils as pdc_utils

# useful to checkout https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7

def match_loss_Lp(des_a, # [M, D]
                  des_b, # [M, D]
                  p=2):

    print("\n---match_loss---")
    print("des_a.shape", des_a.shape)

    loss = (des_a - des_b).abs().pow(p)
    return loss

def non_match_loss(des_a, # [M, D]
                   des_b, # [M, D]
                   margin, # float
                   eps=20, # number to add to denominator
                   verbose=False):

    norm_diff = torch.norm(des_a - des_b, dim=-1)

    y = margin - norm_diff
    y[y < 0] = 0 # clip negative values

    num_hard_negatives = torch.nonzero(y.detach()).shape[0]

    loss_unscaled = torch.sum(y.pow(2))
    loss = loss_unscaled * 1.0/(num_hard_negatives + eps)

    if verbose:
        print("\n---non-match_loss---")
        # print("norm_diff.shape", norm_diff.shape)
        # print("des_a.shape", des_a.shape)
        # print("y.shape", y.shape)
        print("num_hard_negatives", num_hard_negatives)


    return {'loss': loss,
            'loss_unscaled': loss_unscaled,
            'num_hard_negatives': num_hard_negatives,
            }

def match_loss_Huber(des_a,
                     des_b,
                     ):
    raise NotImplementedError
