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

	def __getitem__(self,index):
		img_name = self.img_paths[index]
		img = cv2.imread(img_name)
		# print("img_shape",img.shape)
		# print("img_size",img.size)
		resizedImage = cv2.resize(img,self.img_size)
		# print("resized_shape",resizedImage.shape)
		# print("resized_size",resizedImage.size)
		resizedImage = np.transpose(resizedImage,(2,0,1)).astype('float32')/255.0
		# print("resized_shape",resizedImage.shape)
		# print("resized_size",resizedImage.size)
		iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
		LP_number = iname[-3]
		[leftUp_BB, rightDown_BB] = [[int(x_or_y) for x_or_y in corner.split('&')] for corner in iname[2].split('_')]
		ori_w, ori_h = [float(int(length)) for length in [img.shape[1], img.shape[0]]]
		BB_features = [(leftUp_BB[0] + rightDown_BB[0]) / (2 * ori_w), (leftUp_BB[1] + rightDown_BB[1]) / (2 * ori_h),
                      (rightDown_BB[0] - leftUp_BB[0]) / ori_w, (rightDown_BB[1] - leftUp_BB[1]) / ori_h]
		
		# BB_features = [BB__center_x,BB__center_y,BB_length_x,BB_length_y] all lie between 0-1

		return resizedImage, BB_features,LP_number, img_name

class Label_LP_TestDataLoader(Dataset):
	def __init__(self,img_dirs,img_size):
		self.img_dirs = img_dirs
		self.img_size = img_size
		self.img_paths = []
		# print("assas",len(self.img_dirs))
		for img_dir in self.img_dirs:
			self.img_paths += [images for images in paths.list_images(img_dir) ]
		self.no_of_image = len(self.img_paths)
		print("no of images in directory : ",self.no_of_image)

	def __getitem__(self,index):
		img_name = self.img_paths[index]
		img = cv2.imread(img_name)
		# print("img_shape",img.shape)
		resizedImage = cv2.resize(img,self.img_size)
		# print("resized_shape",resizedImage.shape)
		resizedImage = np.transpose(resizedImage,(2,0,1)).astype('float32')/255.0
		# print("resized_shape",resizedImage.shape)
		LP_number = img_name.split('/')[-1].split('.')[0].split('-')[-3]
		return resizedImage,LP_number,img_name

class DemoTestDataLoader(Dataset):
	def __init__(self,img_dirs,img_size):
		self.img_dirs = img_dirs
		self.img_size = img_size
		self.img_paths = []
		# print("assas",len(self.img_dirs))
		for img_dir in self.img_dirs:
			self.img_paths += [images for images in paths.list_images(img_dir) ]
		self.no_of_image = len(self.img_paths)
		print("no of images in Demo directory : ",self.no_of_image)

	def __getitem__(self,index):
		img_name = self.img_paths[index]
		img = cv2.imread(img_name)
		# print("img_shape",img.shape)
		resizedImage = cv2.resize(img,self.img_size)
		# print("resized_shape",resizedImage.shape)
		resizedImage = np.transpose(resizedImage,(2,0,1)).astype('float32')/255.0
		# print("resized_shape",resizedImage.shape)
		return resizedImage,img_name

# BB -> Bounding Box
class Label_BB_TestDataLoader(Dataset):
	def __init__(self,img_dirs,img_size):
		self.img_dirs = img_dirs
		self.img_size = img_size
		self.img_paths = []
		for img_dir in self.img_dirs:
			# print("image dir : ", img_dir)
			self.img_paths += [images for images in paths.list_images(img_dir) ]
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
		# print("imgnm : ",img_name)
		# print("iname : ",iname)
		[leftUp_BB, rightDown_BB] = [[int(x_or_y) for x_or_y in corner.split('&')] for corner in iname[2].split('_')]
		ori_w, ori_h = [float(int(length)) for length in [img.shape[1], img.shape[0]]]
		BB_features = [(leftUp_BB[0] + rightDown_BB[0]) / (2 * ori_w), (leftUp_BB[1] + rightDown_BB[1]) / (2 * ori_h),
                      (rightDown_BB[0] - leftUp_BB[0]) / ori_w, (rightDown_BB[1] - leftUp_BB[1]) / ori_h]
		
		# BB_features = [BB__center_x,BB__center_y,BB_length_x,BB_length_y] all lie between 0-1

		return resizedImage, BB_features
# class ChaLocDataLoader(Dataset):
# 	def __init__(self, img_dirs,imgSize):
# 		self.img_dirs = img_dirs
# 		self.img_paths = []
# 		for img_dir in img_dirs:
# 		    self.img_paths += [el for el in paths.list_images(img_dir)]
# 		self.no_of_image = len(self.img_paths)
# 		self.img_size = imgSize

# 	def __getitem__(self, index):
# 		img_name = self.img_paths[index]
# 		img = cv2.imread(img_name)
# 		resizedImage = cv2.resize(img, self.img_size)

# 		#or transpose
# 		resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))
# 		resizedImage = resizedImage.astype('float32')
# 		resizedImage /= 255.0


# 		iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
# 		[leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
# 		ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
# 		assert img.shape[0] == 1160
# 		BB_features = [(leftUp[0] + rightDown[0])/(2*ori_w), (leftUp[1] + rightDown[1])/(2*ori_h), (rightDown[0]-leftUp[0])/ori_w, (rightDown[1]-leftUp[1])/ori_h]
# 		return resizedImage, BB_features



# Z = Label_BB_TestDataLoader(['ccpd_dataset'],(480,480))
# c,d = Z[1]
# print(type(c))
# print(d)