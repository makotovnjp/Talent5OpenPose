# Input: data/dance.mp4
# Output: data/pose_dance.mp4

#############################
# Author: Tuan Nguyen
#############################

# standard libs
import typing
import os

# third party libs
import cv2
import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# Internal libs
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose

# define constant
OK = 0
NG = -1

size = (368, 368)

# chuẩn hóa
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]


# Step1: tách lấy file ảnh từ videl
def extract_video(file_path: str, output_path:str) -> []:
    """
    tách lấy file ảnh từ videl
    """
    print("start extract_video")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # initialize return value
    ret_val = OK

    # check arguments
    assert os.path.exists(file_path)

    cap = cv2.VideoCapture(file_path)
    
    img_files = []
    
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            ret_val = NG
            break
        file_img = os.path.join(output_path, 'frame_' + f'{frame_index:05d}' + '.jpg')
        cv2.imwrite(file_img, frame)
        img_files.append(file_img)
        frame_index += 1

    cap.release()
    #cv2.destroyAllWindows()

    return img_files


def predict(img_file_paths: typing.List, model_path: str, result_path:str) -> []:
    """
    step2: predict với những ảnh đã tạo ra
    """
    print("start predict")
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    # create model to predict
    net = OpenPoseNet()

    #net_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    net_weights = torch.load(model_path)
    keys = list(net_weights.keys())

    weights_load = {}

    for key in range(len(keys)):
        #print(key)
        weights_load[list(net.state_dict().keys())[key]] \
            = net_weights[list(keys)[key]]

    #print(weights_load)
    #
    state = net.state_dict()
    state.update(weights_load)
    
    
    net.load_state_dict(state)
    
    result_img_files = []
    
    for image_path in img_file_paths:
        #state = net.state_dict()
        #state.update(weights_load)
    
        #net.load_state_dict(state)
        # predict
        print("Processing: ", image_path)
        oriImg = cv2.imread(image_path)  # B,G,R

        # BGR->RGB
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
        #plt.imshow(oriImg)
        #plt.show()

        # Resize
        #size = (368, 368)
        img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.

        # chuẩn hóa
        #color_mean = [0.485, 0.456, 0.406]
        #color_std = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()  

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

        # （height 、width、colors）→（colors、height、width）
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

        # cho thông tin vào tensor
        img = torch.from_numpy(img)

        x = img.unsqueeze(0)
        #pass
        # Tạo heatmap
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
        
        _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)
        
        #save result to images
        file_img = os.path.join(result_path, os.path.basename(image_path))
        print("Saving: ", file_img)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_img, result_img)
        result_img_files.append(file_img)
        
        #for testing
        #break
        
    return result_img_files
        
# Step3: nối những ảnh đã predict xong thành 1 video
def compose_predicted_images(file_paths, result_path, video_name) -> str:
    predicted_video_path = os.path.join(result_path, video_name)
    
    frame = cv2.imread(file_paths[0])
    height, width, layers = frame.shape
    
    #video = cv2.VideoWriter(predicted_video_path, 0, 1, (width,height))
    #video = cv2.VideoWriter(predicted_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (width,height))
    video = cv2.VideoWriter(predicted_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width,height))
    
    for f in file_paths:
        video.write(cv2.imread(f))
     
    video.release()
    
    return predicted_video_path


if __name__ == "__main__":
    print("start")
    
    img_files = extract_video('./data/dance.mp4', './frame_extracted')
    #img_files = [f for f in os.listdir('./frame_extracted')]
    #print(ret)
    if len(img_files) > 0:
        print("Processing...")
        result_img_files = predict(img_files,
                      model_path='./weights/pose_model_scratch.pth', result_path='./frame_result')
        
        #result_img_files = [os.path.join('./frame_result', f) for f in os.listdir('./frame_result') if f.endswith('.jpg')]
        
        predicted_video_path = compose_predicted_images(result_img_files, result_path='./frame_result', video_name='pose_dance.mp4')
        print('Processed video: ', predicted_video_path)
