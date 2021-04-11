# Input: data/dance.mp4
# Output: data/pose_dance.mp4

# standard libs
import numpy as np
import typing
import os
import cv2
import torch
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose

net = OpenPoseNet()
# define constant
OK = 0
NG = -1

# Step1: tách lấy file ảnh từ videl
def extract_video(file_path: str) -> int:
    """
    tách lấy file ảnh từ videl
    """
    print("start extract_video")

    # initialize return value
    ret_val = OK

    # check arguments
    assert os.path.exists(file_path)

    cap = cv2.VideoCapture(file_path)

    frame_index = 0
    os.chdir('./data/dance')
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        cv2.imwrite('dance_' + str(frame_index) + '.jpg', frame)

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return ret_val

def read_img(image):
    path_read = os.path.join('./data/dance/' + image)
    oriImg = cv2.imread(path_read)  # B,G,R

    # BGR->RGB
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    print('/')
    #plt.imshow(oriImg)
    #plt.show()

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

    return oriImg, x

def create_heap(oriImg, x):

    net.eval()
    predicted_outputs, _ = net(x)

    pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
    heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

    size = (368, 368)
    pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

    pafs = cv2.resize(
        pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(
        heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    return pafs, heatmaps

def predict(img_file_paths: typing.List, model_path: str) -> int:
    """
    step2: predict với những ảnh đã tạo ra
    """
    print("start predict")

    # create model to predict


    net_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    keys = list(net_weights.keys())

    weights_load = {}

    for key in range(len(keys)):
        weights_load[list(net.state_dict().keys())[key]] \
            = net_weights[list(keys)[key]]

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    index = 0
    for image_path in img_file_paths:
        oriImg, x = read_img(image_path)
        pafs, heatmaps = create_heap(oriImg, x)
        _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        path_save = './data/dance_pose/'
        cv2.imwrite(path_save + 'dance_pose_' + str(index) + '.jpg', result_img)
        index += 1

# Step3: nối những ảnh đã predict xong thành 1 video
def compose_predicted_images(file_paths) -> str:

    img_array = []
    for filename in glob.glob('./data/dance_pose/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    return predicted_video_path

def sort_len(x):
    return len(x)

if __name__ == "__main__":
    print("start")
    # ret = extract_video('./data/dance.mp4')
    ret = OK
    if ret == OK:
        print("hoge")
        img_file_paths = os.listdir('./data/dance')
        img_file_paths.sort(key=sort_len)
        ret = predict(img_file_paths,
                      model_path='./weights/pose_model_scratch.pth')
