# -*- coding: utf-8 -*-
"""openpose_video.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KyETUdESGa0ZVPSlWfZSMmEpeOW-0iDR
"""

import os
import urllib.request
import cv2
from PIL import Image
import numpy as np
import torch
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose

def extract_video(file_path):
  

	assert os.path.exists(file_path)
	video = cv2.VideoCapture(file_path)

	frame_index = 0
	while video.isOpened():
		ret, frame = video.read()
		if not ret:
			break

		cv2.imwrite(f'./data/dance_{frame_index}.jpg', frame)

		frame_index += 1

	video.release()
	cv2.destroyAllWindows()

def predict_img(model_path, image_list):
	net = OpenPoseNet()

	# load weights of model
	net_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
	keys = list(net_weights.keys())

	weights_load = {}


	for i in range(len(keys)):
	    weights_load[list(net.state_dict().keys())[i]
	                 ] = net_weights[list(keys)[i]]

	state = net.state_dict()
	state.update(weights_load)
	net.load_state_dict(state)


	for test_image in image_list:

		oriImg = cv2.imread(test_image)  # B,G,R

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
		result = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
		cv2.imwrite(test_image, result)

def compose_predict_img(image_list):
	img = []

	for i in image_list:
		img.append(cv2.imread(i))

	height,width,layers=img[1].shape
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

	video=cv2.VideoWriter('result_video.mp4', fourcc ,20 ,(width,height))


	for j in range(len(image_list)):
		video.write(img[j])

	cv2.destroyAllWindows()
	video.release()

def main():

	extract_video('./data/dance.mp4')

	image_list = []
	for i in range(96):
		image_list.append(f'./data/dance_{i}.jpg')

	predict_img('./weights/pose_model_scrath.pth', image_list)
	compose_predict_img(image_list)

if __name__ == '__main__':
	main()

