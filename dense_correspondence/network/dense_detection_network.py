import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated


class DenseDetectionNetwork(nn.Module):
    def __init__(self):
        super(DenseDetectionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 12 * 17, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 12 * 17)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DenseDetectionResnet(nn.Module):

    def __init__(self):
        super(DenseDetectionResnet, self).__init__()
        num_classes = 2
        self.fcn = getattr(resnet_dilated, "Resnet18_8s")(num_classes=num_classes)
        self.fcn.resnet18_8s.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                                bias=False)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(num_classes*4*5,num_classes)

    def forward(self, x):
        x = self.fcn.forward(x, upsample=False)
        #print x.shape, "is resnet shape after fwd"
        x = self.avgpool(x)
        #print x.shape, "is resnet shape after avgpool"
        x = x.view(x.size(0), -1)
        #print x.shape, "is resnet shape after avgpool"
        x = self.fc(x)
        #print x.shape, "is final shape"
        return x

    @staticmethod
    def from_model_folder(model_folder, model_param_file=None, iteration=None):
        """
        """

        from_model_folder = False
        model_folder = utils.convert_to_absolute_path(model_folder)

        if model_param_file is None:
            model_param_file, _, _ = utils.get_model_param_file_from_directory(os.path.join(model_folder,"detection"), iteration=iteration)
            from_model_folder = True

        model_param_file = utils.convert_to_absolute_path(model_param_file)

        print model_param_file, "loading"

        detection_net = torch.load(model_param_file)

        return detection_net

        