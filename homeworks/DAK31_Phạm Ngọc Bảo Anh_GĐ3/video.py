# input: data/dance.mp4
# Output: data/pose_dance.mp4

# standard libs
import os
import typing

# third-party libs
import cv2
import torch
import numpy as np
from tqdm import tqdm

# internal libs
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose

# define constants
DATA_DIR = './data'
EXTRACT_DIR = './data/extract'
PREDICTED_DIR = './data/predict'
PRETRAINED_MODEL_PATH = './weights/pose_model_scratch.pth'
FRAME_SIZE = (1920, 1080)
OK = 0 
NG = -1

# create data dirs if not exists
if not os.path.exists(PREDICTED_DIR):
    os.mkdir(PREDICTED_DIR)
if not os.path.exists(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

# step 1: extract images from video

def extract_from_video(file_path: str) -> int:
    """
    Extract frames from video to list of images to feed OpenPose model later
    """
    print("Start extracting video")

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
        cv2.imwrite(os.path.join(EXTRACT_DIR, 'dance_' + str(frame_index) + '.jpg'), frame)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return ret_val

# step 2: predict pose from each extracted image
def predict(extract_dir: str, model_path: str) -> int:
    """
    Predict each extracted image from step 1 with pretrained OpenPose model
    input: 
        extract_dir: path to extracted images dir from step 1
        model_path: path to pretrained model for loading and inference
    output:
        OK: if everyting work well
        NG: otherwise
    """
    print("Start predicting")

    ret_val = OK
    if not os.path.exists(model_path):
        ret_val = NG 
        return ret_val

    # loading pretrained model
    net = OpenPoseNet()
    net_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})

    keys = list(net_weights.keys())

    weights_load = {}

    for key in range(len(keys)):
        weights_load[list(net.state_dict().keys())[key]] \
            = net_weights[list(keys)[key]]

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)
    net.eval()

    file_paths = os.listdir(extract_dir)
    def getnumber_fn(img_path: str) -> int:
        """
        extract number from image path: dance_9.png -> 9
        """
        return int(img_path.split('.')[0].split('_')[-1])

    # sort img_paths    
    file_paths.sort(key=getnumber_fn)

    file_paths = [os.path.join(EXTRACT_DIR, path) for path in file_paths]
    for idx in tqdm(range(len(file_paths))):
        # Read image

        oriImg = cv2.imread(file_paths[idx])  # B,G,R

        # BGR->RGB
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

        # Resize
        size = (368, 368)
        img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.

        # Normalization
        color_mean = [0.485, 0.456, 0.406]
        color_std = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()  

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

        # （height 、width、colors）->（colors、height、width）
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

        img = torch.from_numpy(img)
        x = img.unsqueeze(0)

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
        cv2.imwrite(os.path.join(PREDICTED_DIR, 'dance_' + str(idx) + '.jpg'), result_img)

    return ret_val

        

# step 3: compose predicted images into a video
def compose_predicted_images(predicted_dir: str) -> str:
    """
    Compose result images from step 2 into a result video
    Input:
        predicted_dir: path to predicted images from step 2
    Output:
        path to output video
    """
    print("Start composing")
    predicted_video_path = "pose_dance.avi"
    frameSize = FRAME_SIZE

    out = cv2.VideoWriter(predicted_video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, frameSize)

    img_paths = os.listdir(predicted_dir)

    def getnumber_fn(img_path: str) -> int:
        """
        extract number from image path: dance_9.png -> 9
        """
        return int(img_path.split('.')[0].split('_')[-1])

    # sort img_paths    
    img_paths.sort(key=getnumber_fn)

    img_paths = [os.path.join(predicted_dir, path) for path in img_paths]

    for img_path in img_paths:
        img = cv2.imread(img_path)
        out.write(img)

    out.release()

    return predicted_video_path

if __name__ == "__main__":
    ret = extract_from_video("./data/dance.mp4")

    if ret == OK:
        ret = predict(extract_dir=EXTRACT_DIR, 
                      model_path=PRETRAINED_MODEL_PATH)
    
    compose_predicted_images(PREDICTED_DIR)