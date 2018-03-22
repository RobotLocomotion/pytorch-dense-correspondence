## Set up

import sys, os
sys.path.insert(0, '../../pytorch-segmentation-detection/vision/')
sys.path.append('../../pytorch-segmentation-detection/')

# Use second GPU -pytorch-segmentation-detection- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated
import numpy as np
import glob
import sys; sys.path.append('../dataset')
sys.path.append('../correspondence_tools')
from spartan_dataset_masked import SpartanDataset


descriptor_dimensionality = 3
nets = sorted(glob.glob("../recipes/trained_models/train_only_10_drill_long_"+str(descriptor_dimensionality)+"d/dense_resnet*.pth"))
print "Networks:"
for net in nets:
    print "   - ", net

lf = SpartanDataset(mode="test")

last_net = nets[-1]

## Run 

valid_transform = transforms.Compose(
                [
                     transforms.ToTensor(),
                ])

fcn = resnet_dilated.Resnet34_8s(num_classes=descriptor_dimensionality)
fcn.load_state_dict(torch.load(net))
fcn.cuda()
fcn.eval()

def forward_on_img(fcn, img):
    img = valid_transform(img)
    img = img.unsqueeze(0)
    img = Variable(img.cuda())
    res = fcn(img)
    res = res.squeeze(0)
    res = res.permute(1,2,0)
    res = res.data.cpu().numpy().squeeze()
    return res

#res_a = forward_on_img(last_net[0], img_a_rgb)
#res_b = forward_on_img(last_net[0], img_b_rgb)

for i in range(100):
    data_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b = lf[i]

    img_a = Variable(img_a.cuda(), requires_grad=False)
    img_b = Variable(img_b.cuda(), requires_grad=False)
    
    W = 640
    H = 480
    N = 1
    
    if data_type == "matches":
        matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
        matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
        non_matches_a = Variable(non_matches_a.cuda().squeeze(0), requires_grad=False)
        non_matches_b = Variable(non_matches_b.cuda().squeeze(0), requires_grad=False)
       