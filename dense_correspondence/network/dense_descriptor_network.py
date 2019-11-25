import torch.nn as nn
import torchvision

class DenseDescriptorNetwork(nn.Module):

    def __init__(self,
                 backbone, # nn.Module
                 image_width,
                 image_height,
                 normalize):
        """

        :param backbone: nn.Module. Input must be [B,3,H,W] --> [B, D, H, W]
        :type backbone:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :param normalize:
        :type normalize:
        """
        pass

    def forward(self,
                x, # torch.Tensor [B, 3, H, W]
                ):
        pass


def fcn_resnet_backbone(model_name, # str [fcn_resnet50, fcn_resnet101]
                        num_classes, # int
                        pretrained=True, # share the backbone weights
                        ):
    """
    Returns nn.Module
    :return:
    :rtype:
    """

    model_construction_func = getattr(torchvision.models.segmentation, model_name)
    model = model_construction_func(pretrained=False, progress=True,
                                    num_classes=num_classes, aux_loss=None)

    if pretrained:
        model_pretrained = model_construction_func(pretrained=True, progress=True,
                                    num_classes=21, aux_loss=None)

        model.backbone.load_state_dict(model_pretrained.backbone.state_dict(), strict=True)

    return model
