from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord


from skimage.measure import regionprops
def extract_features(mask, prob_map):
    props = regionprops(mask, prob_map)  # 使用区域属性分析工具
    features = []
    for prop in props:
        features.append({
            'coordinate_x': prop.centroid[1],
            'coordinate_y': prop.centroid[0],
            'cell_type': 1,  # 设为1表示肿瘤细胞
            'probability': np.mean(prob_map[prop.coords[:, 0], prop.coords[:, 1]]),
            'area': prop.area,
            'convex_area': prop.convex_area,
            'eccentricity': prop.eccentricity,
            'extent': prop.extent,
            'filled_area': prop.filled_area,
            'major_axis_length': prop.major_axis_length,
            'minor_axis_length': prop.minor_axis_length,
            'orientation': prop.orientation,
            'perimeter': prop.perimeter,
            'solidity': prop.solidity,
            'pa_ratio': prop.major_axis_length / prop.minor_axis_length
        })
    return features
class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		# ##获取细胞的特征
		# image = img.resize((256,256))
		# image =self.roi_transforms(image).unsqueeze(0)
		# from models.DeepCMorph_model import DeepCMorph
		# torch.backends.cudnn.deterministic = True
		# device = torch.device("cuda")
		# model = DeepCMorph(num_classes=41)
		# # Loading model weights corresponding to the network trained on combined datasets
		# # Possible 'dataset' values: TCGA, TCGA_REGULARIZED, CRC, COMBINED
		# model.load_weights(dataset=None, path_to_checkpoints="J:\\CLAM-master\\models\\pretrained_models\\DeepCMorph_Datasets_Combined_41_classes_acc_8159.pth")
		# model.to(device)
		# model.eval()
		# image = image.to(device)
		#
		# # Get the predicted class for a sample input image
		# predictions = model(image)
		# _, predicted_class = torch.max(predictions.data, 1)
		#
		# # Get feature vector of size 2560 for a sample input image
		# features = model(image, return_features=True)
		#
		# # Get predicted segmentation and classification maps for a sample input image
		# nuclei_segmentation_map, nuclei_classification_maps = model(image, return_segmentation_maps=True)
		# from skimage.measure import regionprops, label
		# from skimage.morphology import convex_hull_image
		# nuclei_classification_maps= nuclei_classification_maps.squeeze().cpu().numpy()
		# labeled_nuclei, num_nuclei = label(nuclei_classification_maps, return_num=True, connectivity=2)
		#
		# # 存储每个细胞核的特征
		# nuclei_features = []
		#
		# for region in regionprops(labeled_nuclei):
		# 	if region.area < 20:  # 忽略过小的细胞核
		# 		continue
		#
		# 	# 核心特征提取
		# 	coordinates = region.coords  # 细胞核中所有的像素坐标
		# 	centroid = region.centroid  # 质心 (coordinate_x, coordinate_y)
		# 	area = region.area  # 面积
		# 	convex_image = convex_hull_image(region.image)
		# 	convex_area = convex_image.sum()  # 凸包面积
		# 	eccentricity = region.eccentricity  # 离心率
		# 	extent = region.extent  # 范围
		# 	filled_area = region.filled_area  # 填充区域
		# 	major_axis_length = region.major_axis_length  # 主轴长度
		# 	minor_axis_length = region.minor_axis_length  # 次轴长度
		# 	orientation = region.orientation  # 方向
		# 	perimeter = region.perimeter  # 周长
		# 	solidity = region.solidity  # 密实度
		# 	pa_ratio = region.major_axis_length / region.minor_axis_length  # 主次轴比
		#
		# 	# 通过质心获取分类标签
		# 	centroid_y, centroid_x = np.round(centroid).astype(int)
		# 	cell_type = classification_map[centroid_y, centroid_x]
		#
		# 	# 从概率图中获取细胞核的平均概率
		# 	probability = np.mean(probability_map[coordinates[:, 0], coordinates[:, 1]])
		#
		# 	# 将所有特征存储到字典中
		# 	nuclei_features.append({
		# 		'coordinate_x': centroid_x,
		# 		'coordinate_y': centroid_y,
		# 		'cell_type': cell_type,
		# 		'probability': probability,
		# 		'area': area,
		# 		'convex_area': convex_area,
		# 		'eccentricity': eccentricity,
		# 		'extent': extent,
		# 		'filled_area': filled_area,
		# 		'major_axis_length': major_axis_length,
		# 		'minor_axis_length': minor_axis_length,
		# 		'orientation': orientation,
		# 		'perimeter': perimeter,
		# 		'solidity': solidity,
		# 		'pa_ratio': pa_ratio
		# 	})
		#
		# # 转换为DataFrame并保存为CSV文件
		# nuclei_df = pd.DataFrame(nuclei_features)
		# nuclei_df.to_csv('cell_summary.csv', index=False)
		#
		# print(f"Extracted features for {len(nuclei_df)} nuclei and saved to 'cell_summary.csv'.")
		# # # 转换为numpy格式
		# # mask = nuclei_classification_maps.squeeze().cpu().numpy()
		# # prob_map = nuclei_classification_maps.squeeze().cpu().numpy()
		# #
		# # # 提取细胞核特征
		# # features = extract_features(mask, prob_map)
		# #
		# # # 加入所有特征
		# # all_features=[]
		# # all_features.extend(features)


		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




