import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np; np.random.seed(42)


import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import dense_correspondence_manipulation.utils.utils as utils
utils.set_cuda_visible_devices([1])
utils.add_dense_correspondence_to_python_path()

import dense_correspondence
from dense_correspondence.evaluation.evaluation import *
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
from dense_correspondence.dataset.dense_correspondence_dataset_masked import ImageType
from dense_correspondence.dataset.dynamic_time_contrast_dataset import DynamicTimeContrastDataset

# LOAD DATASET
dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset', 'composite',
                                       'dynamic.yaml')
dataset_config = utils.getDictFromYamlFilename(dataset_config_filename)
dataset = DynamicTimeContrastDataset(config=dataset_config)
logs = dataset.scene_generator()
logs_list = []
for log in logs:
    logs_list.append(log)

# LOAD DESCRIPTOR NETWORK
eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 
                               'dense_correspondence', 'evaluation', 'evaluation.yaml')
eval_config = utils.getDictFromYamlFilename(eval_config_filename)

utils.set_cuda_visible_devices([0])
dce = DenseCorrespondenceEvaluation(eval_config)
network_name = "sugar_closer_3"
dcn = dce.load_network_from_config(network_name)
dcn.cuda().eval()
print "loaded dcn"

# load tcn
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbeddingNetwork(nn.Module):
    def __init__(self, D):
        super(TimeEmbeddingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 3, 5)
        self.fc1 = nn.Linear(12768, 12768)
        self.fc2 = nn.Linear(12768, 12768)
        self.fc3 = nn.Linear(12768, D)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,12768)
        drop = nn.Dropout(0.1)
        x = F.relu(drop(self.fc1(x)))
        x = F.relu(drop(self.fc2(x)))
        x = self.fc3(x)
        return x

D_embedding = 32
model = TimeEmbeddingNetwork(D=D_embedding).cuda()
model.load_state_dict(torch.load("/home/peteflo/code/dense_correspondence/training/tensorboard_log_dir/2019-02-23-19-41-36/norm_net.pth"))
model.eval()
print "loaded tcn"

num_logs_used = 1
embeddings = None
colors = None
images = None

for i, log in enumerate(logs_list[0:num_logs_used]):
    print i

    scene_directory = dataset.get_full_path_for_scene(log)
    state_info_filename = os.path.join(scene_directory, "states.yaml")
    state_info_dict = utils.getDictFromYamlFilename(state_info_filename)
    image_idxs = state_info_dict.keys() # list of integers

    camera_num = 0

    new_embeddings = np.zeros((len(image_idxs),D_embedding))
    new_colors = np.zeros((len(image_idxs)))
    new_images = np.empty((len(image_idxs),480,640))
    new_embeddings.shape
    new_colors.shape

    

    #for i in image_idxs:
    for j in range(len(image_idxs)):
        rgb = dataset.get_rgb_image(dataset.get_image_filename(log, camera_num, j, ImageType.RGB))
        rgb_tensor = dataset.rgb_image_to_tensor(rgb)
        rgb_tensor = rgb_tensor.unsqueeze(0).cuda() #N, C, H, W
        descriptor_image = dcn.forward(rgb_tensor).detach() #N, D, H, W
        stacked = torch.cat([rgb_tensor[0], descriptor_image[0]]).unsqueeze(0)
        embedding = model(stacked)[0].detach().cpu().numpy()
        new_embeddings[j] = embedding
        new_colors[j] = float(i) / float(num_logs_used)
        rgb_numpy = np.asarray(rgb)
        print rgb_numpy.shape
        new_images[j,:,:] = rgb_numpy[:,:,0]

    if i == 0:
        embeddings = new_embeddings
        colors = new_colors
        images = new_images
    else:
        embeddings = np.vstack((embeddings, new_embeddings))
        images = np.vstack((images, new_images))
        colors = np.hstack((colors, new_colors))


print images.shape


### GET TSNE
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D
X, color = embeddings, colors
n_neighbors = 10
n_components = 2
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)


# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)
cmap=plt.cm.Spectral
colors_mapped = [cmap(c) for c in colors ]
line, = ax.plot(Y[:,0], Y[:,1], c="black", ls="", marker="o")

# create the annotations box
im = OffsetImage(images[0,:,:], zoom=0.5)
xybox=(50., 50.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

def hover(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:

        print line.contains(event)[1]["ind"]
        # find out the index within the array from the event
        try:
            ind = line.contains(event)[1]["ind"][0]
        except:
            ind = line.contains(event)[1]["ind"]

        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(Y[ind,0], Y[ind,1])
        # set the image corresponding to that point
        im.set_data(images[ind,:,:])
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)           
plt.show()