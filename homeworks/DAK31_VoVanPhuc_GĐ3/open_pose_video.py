# Input: data/dance.mp4
# Output: data/pose_dance.mp4

# third party libs
import cv2
import torch
import numpy as np
# Internal libs
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose


def predict_in_video(video_path: str, model_path: str) -> None:
    net = OpenPoseNet()

    net_weights = torch.load(model_path, map_location=torch.device('cuda:0'))
    keys = list(net_weights.keys())

    weights_load = {}

    for i in range(len(keys)):
        weights_load[list(net.state_dict().keys())[i]
        ] = net_weights[list(keys)[i]]

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)
    net.eval()
    print('load done')
    size = (368, 368)
    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('dak31_gd3.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    while cap.isOpened():
        _, frame = cap.read()
        img = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.
        preprocessed_img = img.copy()
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img)
        x = img.unsqueeze(0)
        predicted_outputs, _ = net(x)
        pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
        heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

        pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

        pafs = cv2.resize(pafs, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(heatmaps, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        _, result_img, _, _ = decode_pose(frame, heatmaps, pafs)

        out.write(result_img)
        # cv2.imshow('out', result_img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_in_video('./data/dance.mp4', './weights/pose_model_scratch.pth')
