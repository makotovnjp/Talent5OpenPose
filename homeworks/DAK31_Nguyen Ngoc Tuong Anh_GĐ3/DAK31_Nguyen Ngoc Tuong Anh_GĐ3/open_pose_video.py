import cv2
import typing
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose

OK = 0
NG = -1 

def extract_video(file_path: str) -> None:
    # Opens the Video file
    print("start extract video")
    
    # initialize return value 
    ret = OK 
    
    # check arguments
    assert os.path.exists(file_path)
    
    cap= cv2.VideoCapture(file_path)
    frame_index = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('dance_' + str(frame_index) + '.jpg',frame)
        frame_index +=1

    cap.release()
    cv2.destroyAllWindows()
    
    return ret

# step2: Predict with extracted images
def predict(img_file_paths: typing.List, model_path: str):

    # predicted img paths
    predicted_imgs = []
    
    # create model to predict 
    net = OpenPoseNet()
    
    # load weights of model
    net_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    keys = list(net_weights.keys())

    
    weights_load = {}


    for key in range(len(keys)):
        weights_load[list(net.state_dict().keys())[key]] = net_weights[list(keys)[key]]
        
    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    print('load done')

    for image_path in img_file_paths:
        # predict 
        print(image_path)
        # Read image
        oriImg = cv2.imread(image_path)  # B,G,R

        # BGR->RGB
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

        # Resize
        size = (368, 368)
        img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.

        # chuẩn hóa
        color_mean = [0.485, 0.456, 0.406]
        color_std = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()  

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

        # （height 、width、colors）→（colors、height、width）
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

        # cho thông tin vào tensor
        img = torch.from_numpy(img)

        x = img.unsqueeze(0)

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

        cv2.imwrite('predicted_' + image_path, result_img)
    
        predicted_imgs.append('predicted_' + image_path)

    return predicted_imgs

# step3: nối những ảnh đã predict thành 1 video 
def compose_predicted_images(file_paths) -> str:
    predicted_video_path = "predicted_dance.mp4"
    img_array = []
    predicted_imgs = file_paths

    for img_path in predicted_imgs:
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(predicted_video_path,cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    return predicted_video_path 


if __name__ == "__main__":
    print(os.getcwd())
    print("start")
    ret = extract_video("./data/dance.mp4")    
    print(ret)
    # predicted_vid = compose_predicted_images(['predicted_dance_' + str(i) + '.jpg' for i in range(0,96)])
    if ret == OK:
        print("hoge")
        ret = predict(img_file_paths=['dance_' + str(i) + '.jpg' for i in range(0,96)],
                        model_path='./weights/pose_model_scratch.pth')
        predicted_vid = compose_predicted_images(ret)