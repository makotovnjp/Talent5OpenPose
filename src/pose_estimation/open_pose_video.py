# Input: data/dance.mp4
# Output: data/pose_dance.mp4

# standard libs
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


# Step3: nối những ảnh đã predict xong thành 1 video
def compose_predicted_images(file_paths) -> str:
    predicted_video_path = ""

    return predicted_video_path


if __name__ == "__main__":
    print("start")
    ret = extract_video('./data/dance.mp4')
    print(ret)
    if ret == OK:
        print("hoge")
        ret = predict(img_file_paths=[],
                      model_path='./weights/pose_model_scratch.pth')
