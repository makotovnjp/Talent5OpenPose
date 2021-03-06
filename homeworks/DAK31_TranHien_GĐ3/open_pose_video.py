# -*- coding: utf-8 -*-
"""open_pose_video.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G54FSaEMjy64Wbru1CrmTuhQh84JPoBz

# input: data/dance.mp4
# output: data/pose_dance.mp4
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)
# %cd gdrive/MyDrive/Talent5OpenPose/src3/pose_estimation

# lib tieu chuan
import typing
import os

# third party libs 
import cv2
import torch

# Internal libs
from utils.openpose_net import OpenPoseNet

# define constant
OK = 0
NG = -1

# Step1: tách lấy file ảnh từ video
def extract_video(file_path: str) -> int:
    """
    tách lấy file ảnh từ videl
    """
    print("start extract_video")

    # initialize return value
    ret_val = OK

    # check arguments: check xem file co ton tai hay ko
   # assert os.path.isfile(file_path)
    print (os.path.isfile)
    cap = cv2.VideoCapture(file_path)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('dance_' + str(frame_index) + '.jpg', frame)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return ret_val

def predict(img_file_paths: typing.List, model_path: str) -> int:
    """
    step2: predict với những ảnh đã tạo ra
    """
    print("start predict")

    # create model to predict
    net = OpenPoseNet()

    net_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    keys = list(net_weights.keys())

    weights_load = {}

    for key in range(len(keys)):
        print(key)
        weights_load[list(net.state_dict().keys())[key]] \
            = net_weights[list(keys)[key]]

    print(weights_load)
    #
    # state = net.state_dict()
    # state.update(weights_load)
    #
    #
    # net.load_state_dict(state)

    for image_path in img_file_paths:
        # predict
        pass

# Tạo predicted tensor
out_path = ''

# Step3: nối những ảnh đã predict xong thành 1 video
def compose_predicted_images(file_paths) -> str:
    predicted_video_path = ""

    return predicted_video_path

if __name__ == "__main__":
    print("start")
    ret = extract_video('/content/drive/MyDrive/Talent5OpenPose/src3/pose_estimation/data/dance.mp4')
    print(ret)
    if ret == OK:
        print("hoge")
        ret = predict(img_file_paths=[],
                      model_path='/content/gdrive/MyDrive/Talent5OpenPose/src3/pose_estimation/weights/pose_model_scratch.pth')

cv2.VideoWriter

# luu file anh da tách vao foder
'''
import os
os.chdir('/content/drive/MyDrive/Talent5OpenPose/src3/pose_estimation/data')
!pwd
'''





