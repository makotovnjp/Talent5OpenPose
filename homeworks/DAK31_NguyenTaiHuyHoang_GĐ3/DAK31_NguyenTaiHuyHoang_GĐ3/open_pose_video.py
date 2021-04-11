# Input: data/dance.mp4
# Output: data/pose_dance.mp4

import os
import typing
import glob
import cv2
import torch
from PIL import Image
from utils.openpose_net import OpenPoseNet

OK = 0
NG = -1

#Step 1
def extract_video(file_path: str) -> None:
    print("start extract_video")

    ret_val = OK

    # check arguments
    assert os.path.exists(file_path)

    cap= cv2.VideoCapture(file_path)
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('dance_'+str(frame_index)+'.jpg',frame)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return ret_val

#Step 2
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
    state = net.state_dict()
    state.update(weights_load)
    #
    #
    net.load_state_dict(state)

    for image_path in img_file_paths:
        # predict
        pass

    # Step3: nối những ảnh đã predict xong thành 1 video
def compose_predicted_images(file_paths) ->str:
    predicted_video_path = "dance.mp4"
    img_array = []
    for key in range(0,95 ):
        img = cv2.imread('/Users/hoangwalker/PycharmProjects/Talent5OpenPose\ python\=3.8\ 1/OpenPose/Talent5OpenPose-main/src/pose_estimation/dance_[0].jpg'.format(key))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('data/pose_dance.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

    for key in range(len(img_array)):
        out.write(img_array[key])
    out.release()
    
    return predicted_video_path









if __name__ == "__main__":
    print("start")
    ret = extract_video('./data/dance.mp4')
    print(ret)
    if ret == OK:
        print("hoge")
        ret = predict(img_file_paths=[],
                      model_path='./weights/pose_model_scratch.pth')




