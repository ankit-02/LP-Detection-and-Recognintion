from torch.utils.data import *
from imutils import paths
import cv2
import numpy as np

class Label_BB_LP_TestDataLoader(Dataset):
	def __init__(self,img_dirs,img_size):
		self.img_dirs = img_dirs
		self.img_size = img_size
		self.img_paths = []
		for img_dir in self.img_dirs:
			self.img_paths += [images for images in paths.list_images(img_dir) ]
		self.no_of_image = len(self.img_paths)
		print("no of images for LabelFPSdirectory : ",self.no_of_image)

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self,index):
		img_name = self.img_paths[index]
		img = cv2.imread(img_name)
		resizedImage = cv2.resize(img,self.img_size)
		resizedImage = np.transpose(resizedImage,(2,0,1)).astype('float32')/255.0
		iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
		LP_number = iname[-3]
		[leftUp_BB, rightDown_BB] = [[int(x_or_y) for x_or_y in corner.split('&')] for corner in iname[2].split('_')]
		ori_w, ori_h = [float(int(length)) for length in [img.shape[1], img.shape[0]]]
		BB_features = [(leftUp_BB[0] + rightDown_BB[0]) / (2 * ori_w), (leftUp_BB[1] + rightDown_BB[1]) / (2 * ori_h),
                      (rightDown_BB[0] - leftUp_BB[0]) / ori_w, (rightDown_BB[1] - leftUp_BB[1]) / ori_h]
		
		return resizedImage, BB_features,LP_number

class Label_LP_TestDataLoader(Dataset):
	def __init__(self,img_dirs,img_size):
		self.img_dirs = img_dirs
		self.img_size = img_size
		self.img_paths = []
		for img_dir in self.img_dirs:
			self.img_paths += [images for images in paths.list_images(img_dir) ]
		self.no_of_image = len(self.img_paths)
		print("no of images in directory : ",self.no_of_image)

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self,index):
		img_name = self.img_paths[index]
		img = cv2.imread(img_name)
		resizedImage = cv2.resize(img,self.img_size)
		resizedImage = np.transpose(resizedImage,(2,0,1)).astype('float32')/255.0
		LP_number = img_name.split('/')[-1].split('.')[0].split('-')[-3]
		return resizedImage,LP_number

class DemoTestDataLoader(Dataset):
	def __init__(self,img_dirs,img_size):
		self.img_dirs = img_dirs
		self.img_size = img_size
		self.img_paths = []
		for img_dir in self.img_dirs:
			self.img_paths += [images for images in paths.list_images(img_dir) ]
		self.no_of_image = len(self.img_paths)
		print("no of images in Demo directory : ",self.no_of_image)
	
	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self,index):
		img_name =self.img_paths[index]
		img = cv2.imread(img_name)
		resizedImage = cv2.resize(img,self.img_size)
		resizedImage = np.transpose(resizedImage,(2,0,1)).astype('float32')/255.0
		iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')

		[leftUp_BB, rightDown_BB] = [[int(x_or_y) for x_or_y in corner.split('&')] for corner in iname[2].split('_')]
		ori_w, ori_h = [float(int(length)) for length in [img.shape[1], img.shape[0]]]
		BB_features = [(leftUp_BB[0] + rightDown_BB[0]) / (2 * ori_w), (leftUp_BB[1] + rightDown_BB[1]) / (2 * ori_h),
                      (rightDown_BB[0] - leftUp_BB[0]) / ori_w, (rightDown_BB[1] - leftUp_BB[1]) / ori_h]
		return resizedImage,img_name,BB_features

# BB -> Bounding Box
class Label_BB_TestDataLoader(Dataset):
	def __init__(self,img_dirs,img_size):
		self.img_dirs = img_dirs
		self.img_size = img_size
		self.img_paths = []
		for img_dir in self.img_dirs:
			self.img_paths += [images for images in paths.list_images(img_dir)]
			print("image dir : ", img_dir ,  len(self.img_paths))
		self.no_of_image = len(self.img_paths)
		print("no of images for Labeldirectory : ",self.no_of_image)

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self,index):
		img_name =self.img_paths[index]
		img = cv2.imread(img_name)
		
		resizedImage = cv2.resize(img,self.img_size)
		resizedImage = np.transpose(resizedImage,(2,0,1)).astype('float32')/255.0
		iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
		
		[leftUp_BB, rightDown_BB] = [[int(x_or_y) for x_or_y in corner.split('&')] for corner in iname[2].split('_')]
		ori_w, ori_h = [float(int(length)) for length in [img.shape[1], img.shape[0]]]
		BB_features = [(leftUp_BB[0] + rightDown_BB[0]) / (2 * ori_w), (leftUp_BB[1] + rightDown_BB[1]) / (2 * ori_h),
                      (rightDown_BB[0] - leftUp_BB[0]) / ori_w, (rightDown_BB[1] - leftUp_BB[1]) / ori_h]
		

		return resizedImage, BB_features
