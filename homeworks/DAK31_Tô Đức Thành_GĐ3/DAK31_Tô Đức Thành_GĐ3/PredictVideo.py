#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import torch
from glob import glob
from utils.decode_pose import decode_pose


# In[2]:


cap = cv2.VideoCapture('./data/dance.mp4')
ret_ls = []
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite('dance_' + str(frame_index) + '.jpg', frame)
    frame_index += 1
    ret_ls.append(ret)
cap.release()
cv2.destroyAllWindows()
print('successful')


# In[4]:


pwd


# In[5]:


cd data\image_video


# In[6]:


# list all the images extracted from video
ls


# # The images are not sorted chronologically

# In[8]:


# Sort images extracted from video chronologically
img_gen_sorted = sorted(glob(f'{os.getcwd()}/*.jpg'), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
img_gen_sorted


# # Sorted chronologically

# In[9]:


# Create model
from utils.openpose_net import OpenPoseNet

net = OpenPoseNet()

# load weights of model
net_weights = torch.load(
    './weights/pose_model_scratch.pth', map_location={'cuda:0': 'cpu'})
keys = list(net_weights.keys())

weights_load = {}


for i in range(len(keys)):
    weights_load[list(net.state_dict().keys())[i]
                 ] = net_weights[list(keys)[i]]

state = net.state_dict()
state.update(weights_load)
net.load_state_dict(state)

print('load done')


# In[10]:


# Read image, predict the pose and write the result into predict_image folders.
for (index, img_seq) in enumerate(img_gen_sorted):
    oriImg = cv2.imread(img_seq)  # B,G,R
    # BGR->RGB
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    # Resize
    size = (368, 368)
    img1 = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)
    img1 = img1.astype(np.float32) / 255.
    
    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]

    preprocessed_img = img1.copy()  

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]
    # （height 、width、colors）→（colors、height、width）
    img1 = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

    # cho thông tin vào tensor
    img1 = torch.from_numpy(img1)

    x = img1.unsqueeze(0)
    
    # Create heatmap
    net.eval()
    predicted_outputs, _ = net(x)

    pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
    heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

    pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

    pafs = cv2.resize(
        pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(
        heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Get result images and write them into folder. 
    _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)
    cv2.imwrite('./data/predict_image/' + str(index) + '.jpg', result_img)
print('successful')


# In[11]:


cd E:\MachineLearning\Talent5\OpenPose\main\src\pose_estimation\data\predict_image


# In[13]:


# Sort predicted images
img_predict_sorted = sorted(glob(f'{os.getcwd()}/*.jpg'), key=lambda x: int(os.path.basename(x).split('.')[0]))
img_predict_sorted


# In[14]:


# Combine images to make a mp4 video
img_arr = []
for file in img_predict_sorted:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, layers = img.shape
    size = (width, height)
    img_arr.append(img)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('PredictVideo.mp4', fourcc, 25, size)

for i in range(len(img_arr)):
    out.write(img_arr[i])
out.release()
print('successful')


# In[17]:


cd E:\MachineLearning\Talent5\OpenPose\main\src\pose_estimation


# In[18]:


ls


# # We can see PredictVideo.mp4
