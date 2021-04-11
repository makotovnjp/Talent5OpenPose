# Input: data/dance.mp4
# Output: data/pose_dance.mp4

# standard libs
import typing
import os
import numpy as np
import glob
# third party libs
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
# Internal libs
from utils.openpose_net import OpenPoseNet

# define constant
OK = 0
NG = -1

img_array = []

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        a = cv2.imwrite('./data/IMG/'+'dance_' + str(frame_index) + '.jpg', frame)
        frame_index += 1
        ret = predict(img_file_paths=frame,
                      model_path='./weights/pose_model_scratch.pth')
        img_array.append(ret)


    cap.release()
    cv2.destroyAllWindows()

    return img_array


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

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    print('load done')

    # Read image

    # oriImg = cv2.imread(img_file_paths)  # B,G,R
    oriImg = img_file_paths

    # BGR->RGB
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    # plt.title('Origin')
    # plt.imshow(oriImg)
    # plt.show()

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

    from utils.decode_pose import decode_pose
    _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)

    # 結果を描画
    # plt.imshow(oriImg)
    # plt.show()
    #
    # plt.imshow(result_img)
    # plt.title('Result')
    # plt.savefig('Result.png')
    # plt.show()
    return result_img

# Step3: nối những ảnh đã predict xong thành 1 video
def compose_predicted_images(file_paths) -> str:
    # img_array = []
    # path = r'F:\0.2 Pro\Talent5\Code\data\IMG\\'
    # file = os.listdir(r'F:\0.2 Pro\Talent5\Code\data\IMG')
    size = (1920, 1080)
    # for filename in glob.glob(r'F:\0.2 Pro\Talent5\Code\data\IMG\*.jpg'):
    #     # for i in range(len(file)):
    #     img = cv2.imread(filename)
    #     # height, width, layers = img.shape
    #     # size = (width, height)
    #     img_array.append(img)
    #     # cv2.imshow('a', img)

    out = cv2.VideoWriter('Result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    predicted_video_path = out

    return predicted_video_path


if __name__ == "__main__":
    print("start")
    ret = extract_video('./data/dance.mp4')
    print(ret)
    if ret == OK:
        print("hoge")
        ret = predict(img_file_paths=[],
                      model_path='./weights/pose_model_scratch.pth')
    ret = compose_predicted_images(ret)
