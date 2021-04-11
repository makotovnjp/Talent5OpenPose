#!/usr/bin/env python

from PIL import Image
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import torch

import os, fnmatch

from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose


def mkFolder(srcDir, dataDir, resultDir, resultImgDir):
    if not os.path.exists(srcDir):
        os.mkdir(srcDir)
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)
    if not os.path.exists(resultImgDir):
        os.mkdir(resultImgDir)


def extractFrame (desPath, videoPath):
    cap= cv2.VideoCapture(videoPath)
    i=0
    fps = cap.get(cv2.CAP_PROP_FPS)

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(desPath, str(i)+'.jpg'),frame)

        i+=1

    return int(width), int(height), fps


def compressVideo (fps, frameSize, framePath, outputFilename):

    out = cv2.VideoWriter(outputFilename, 
                          cv2.VideoWriter_fourcc(*'DIVX'), 
                          fps, 
                          frameSize)

    filenames = fnmatch.filter(os.listdir(framePath), '*.jpg')
    for i in range(len(filenames)):
        img = cv2.imread(os.path.join(framePath,str(i)+".jpg"))
        out.write(img)

    out.release()

def getResultImg (net, imgPath, resultFilename, resultDir):   
    oriImg = cv2.imread(imgPath)  # B,G,R

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
    plt.imsave(os.path.join(resultDir, resultFilename+'.jpg'), result_img)


def predict(modelPath, imgDir, resultDir):
    net = OpenPoseNet()

    # load weights of model
    net_weights = torch.load(
        modelPath, map_location={'cuda:0': 'cpu'})
    keys = list(net_weights.keys())

    weights_load = {}


    for i in range(len(keys)):
        weights_load[list(net.state_dict().keys())[i]
                     ] = net_weights[list(keys)[i]]

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    print('load model done')
    
    filenames = fnmatch.filter(os.listdir(imgDir), '*.jpg')
    for i in range(len(filenames)):
        imgPath = os.path.join(imgDir,str(i)+".jpg")
        
        getResultImg(net, imgPath, str(i), resultDir)

    print("finish predict frames")





if __name__ == "__main__":
    curDir = os.getcwd()
    videoPath = './data/dance.mp4'
    srcDir = './videoPose'
    dataDir = './videoPose/ExtractedImg'
    resultDir = './videoPose/Result'
    resultImgDir = './videoPose/Result/ResultImg'


    print("start")
    mkFolder(srcDir, dataDir, resultDir, resultImgDir)

    width, height, fps = extractFrame(dataDir, videoPath)
    modelPath ='./weights/pose_model_scratch.pth'
    
    predict(modelPath, dataDir, resultImgDir)
    outputFilename = os.path.join(resultDir, "dance.mp4")
    compressVideo(fps, (width, height), resultImgDir, outputFilename)

    print("finish")




