import torch
import torch.nn as nn
import torchvision


def sphere_projection(x,
                      p, # use Lp norm
                      dim, # which dimension to normalize
                      ):

    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    x_out = torch.div(x, norm)
    return x_out

class SphereProjection(nn.Module):

    def __init__(self, p, dim):
        self.p = p
        self.dim = dim

    def forward(self,
                x, # [B, D, H, W]
                ): # torch.Tensor [B, D, H, W]

        # each x_norm[i,:, j, k, l] has norm 1

        return sphere_projection(x, p=self.p, dim=self.dim)

class DenseDescriptorNetwork(nn.Module):

    def __init__(self,
                 model,
                 normalize,
                 config=None,
                 ):
        nn.Module.__init__(self)
        """

        :param model: nn.Module. Input must be [B,3,H,W] --> [B, D, H, W]
        :param normalize:
        :type normalize:
        """
        self.model = model
        self.normalize = normalize
        self.config = config
        if self.normalize:
            self.sphere_projection = SphereProjection(p=2, dim=1)

    def forward(self,
                x, # torch.Tensor [B, 3, H, W]
                ):

        model_out = self.model.forward(x)
        descriptor_image = model_out['out']
        if self.normalize:
            descriptor_image = self.sphere_projection.forward(x)

        return {'descriptor_image': descriptor_image,
                'model_out': model_out}


def fcn_resnet(model_name,  # str [fcn_resnet50, fcn_resnet101]
               num_classes,  # int
               pretrained=True,  # share the backbone weights
               ):
    """
    Returns nn.Module
    :return:
    :rtype:
    """

    # note still need to deal with initialization of those layers . . .

    model_construction_func = getattr(torchvision.models.segmentation, model_name)
    model = model_construction_func(pretrained=False, progress=True,
                                    num_classes=num_classes, aux_loss=None)

    if pretrained:
        # can only load pretrained models with num_classes=21 since this is what
        # is in Pascal VOC
        model_pretrained = model_construction_func(pretrained=True, progress=True,
                                    num_classes=21, aux_loss=None)

        model.backbone.load_state_dict(model_pretrained.backbone.state_dict(), strict=True)

    return model
