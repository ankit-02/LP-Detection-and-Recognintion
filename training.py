import torch
import torch.nn as nn
from load_data import *
from time import time
from torch.autograd import Variable
from torch._thnn import type2backend
from torch.autograd.function import Function


is_gpu_available = torch.cuda.is_available()
print("is_gpu_available",is_gpu_available)
if is_gpu_available:
	print("torch cuda device count",torch.cuda.device_count())

no_of_output = 4
img_size = (480,480)
batchsize = 3
print("batchsize : ",batchsize)

class Maxpool_LP(Function):
	def __init__(self):
		super(Maxpool_LP, self).__init__()
		self.width = 16
		self.height = 8

	def forward(self, input_i):
		output = input_i.new()
		indices = input_i.new().long()
		self._backend = type2backend[input_i.type()]
		self._backend.SpatialAdaptiveMaxPooling_updateOutput(
			self._backend.library_state, input_i, output, indices,
			self.width, self.height)
		return output


def roi_lp(input, rois):
	output = []
	rois = rois.data.long()
	num_rois = rois.size(0)

	for i in range(num_rois):
		roi = rois[i]
		im = input[i,:, roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
		temp = im.shape
		im = im.reshape(1,temp[0],temp[1],temp[2])
		output.append(Maxpool_LP()(im))

	return torch.cat(output, 0)


class BB_model(nn.Module):
	def __init__(self,no_of_output_class=1):
		super(BB_model, self).__init__()
		hidden1 = nn.Sequential(
		    nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
		    nn.BatchNorm2d(num_features=48),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
		    nn.Dropout(0.2)
		)
		hidden2 = nn.Sequential(
		    nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
		    nn.BatchNorm2d(num_features=64),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
		    nn.Dropout(0.2)
		)
		hidden3 = nn.Sequential(
		    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
		    nn.BatchNorm2d(num_features=128),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
		    nn.Dropout(0.2)
		)
		hidden4 = nn.Sequential(
		    nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
		    nn.BatchNorm2d(num_features=160),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
		    nn.Dropout(0.2)
		)
		hidden5 = nn.Sequential(
		    nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
		    nn.BatchNorm2d(num_features=192),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
		    nn.Dropout(0.2)
		)
		hidden6 = nn.Sequential(
		    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
		    nn.BatchNorm2d(num_features=192),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
		    nn.Dropout(0.2)
		)
		hidden7 = nn.Sequential(
		    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
		    nn.BatchNorm2d(num_features=192),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
		    nn.Dropout(0.2)
		)
		hidden8 = nn.Sequential(
		    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
		    nn.BatchNorm2d(num_features=192),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
		    nn.Dropout(0.2)
		)
		hidden9 = nn.Sequential(
		    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
		    nn.BatchNorm2d(num_features=192),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
		    nn.Dropout(0.2)
		)
		hidden10 = nn.Sequential(
		    nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
		    nn.BatchNorm2d(num_features=192),
		    nn.ReLU(),
		    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
		    nn.Dropout(0.2)
		)
		self.features = nn.Sequential(
		    hidden1,
		    hidden2,
		    hidden3,
		    hidden4,
		    hidden5,
		    hidden6,
		    hidden7,
		    hidden8,
		    hidden9,
		    hidden10
		)

		#add relu in classifier, batch normalization
		self.classifier = nn.Sequential(
		    nn.Linear(23232, 100),
		    nn.ReLU(),
		    nn.Linear(100, 100),
		    nn.ReLU(),
		    nn.Linear(100, no_of_output_class),
		)

	def forward(self, x):
		x1 = self.features(x)
		x11 = x1.view(x1.size(0), -1)
		x = self.classifier(x11)
		return x

class ROI_final_Classifier(nn.Module):
	def __init__(self,BB_model_path=None):
		super(ROI_final_Classifier,self).__init__()
		self.load_BB_model(BB_model_path)
		self.hidden_1 = nn.Sequential(
			nn.Linear(53248,38),
		)
		self.hidden2 = nn.Sequential(
			nn.Linear(53248,25),
		)
		self.hidden3 = nn.Sequential(
			nn.Linear(53248,35),
		)
		self.hidden4 = nn.Sequential(
			nn.Linear(53248,35),
			)
		self.hidden5 = nn.Sequential(
			nn.Linear(53248,35),
			)
		self.hidden6 = nn.Sequential(
			nn.Linear(53248,35),
			)
		self.hidden7 = nn.Sequential(
			nn.Linear(53248,35),
			)

	def load_BB_model(self,path):
		if path is None:
			print("No trained model given for LP detection, initializing new one")
			self.BB_model = BB_model(4)
		else :
			self.BB_model = torch.load(path)
		self.BB_model.train(False)

	def forward(self,image):
		BB_roi_1= torch.FloatTensor([[122.0,0,122.0,0],[0,122.0,0,122.0],[-61.0,0,61.0,0],[0,-61.0,0,61.0]]).cuda()
		BB_roi_2= torch.FloatTensor([[63.0,0,63.0,0],[0,63.0,0,63.0],[-31.5,0,31.5,0],[0,-31.5,0,31.5]]    ).cuda()
		BB_roi_3= torch.FloatTensor([[33.0,0,33.0,0],[0,33.0,0,33.0],[-16.5,0,16.5,0],[0,-16.5,0,16.5]]    ).cuda()

		hidden1 = self.BB_model.module.features[0](image)
		hidden2 = self.BB_model.module.features[1](hidden1)
		hidden3 = self.BB_model.module.features[2](hidden2)
		hidden4 = self.BB_model.module.features[3](hidden3)
		hidden5 = self.BB_model.module.features[4](hidden4)
		hidden6 = self.BB_model.module.features[5](hidden5)
		hidden7 = self.BB_model.module.features[6](hidden6)
		hidden8 = self.BB_model.module.features[7](hidden7)
		hidden9 = self.BB_model.module.features[8](hidden8)
		hidden10 = self.BB_model.module.features[9](hidden9)
		BB_pred = self.BB_model.module.classifier(hidden10.view(hidden10.size(0),-1))

		if(BB_pred.data.size()[1]) !=4:
			print("size is not 4")

		ROI_hidden2 = roi_lp(hidden2,BB_pred.mm(BB_roi_1).clamp(min=0,max=122))
		ROI_hidden4 = roi_lp(hidden4,BB_pred.mm(BB_roi_2).clamp(min=0,max=63) )
		ROI_hidden6 = roi_lp(hidden6,BB_pred.mm(BB_roi_3).clamp(min=0,max=33) )

		Roi_temp = torch.cat((ROI_hidden2,ROI_hidden4,ROI_hidden6),1)
		Roi = Variable(Roi_temp.view(Roi_temp.size(0),-1))

		Char_pred = [0 for i in range(7)]
		
		Char_pred[0] = self.hidden_1(Roi)
		Char_pred[1] = self.hidden2(Roi)
		Char_pred[2] = self.hidden3(Roi)
		Char_pred[3] = self.hidden4(Roi)
		Char_pred[4] = self.hidden5(Roi)
		Char_pred[5] = self.hidden6(Roi)
		Char_pred[6] = self.hidden7(Roi)

		return BB_pred,Char_pred

print("loading data")
# Final_Test_data_filename = Label_BB_TestDataLoader(['ccpd_base'],img_size)
Final_Train_data_filename = Label_BB_LP_TestDataLoader(['base_40k'],img_size)

print("loading data ...")
# Final_Test_data = DataLoader(FInal_Test_data_filename,batch_size=10,shuffle = False,num_workers=8)
Final_Train_data = DataLoader(Final_Train_data_filename,batch_size=batchsize,shuffle = False,num_workers=8)

print("loaded")

def train_model(model,criterion,optimiser,epochs,start_epoch = 0):

	if start_epoch > 0 :
		lrScheduler    = torch.optim.lr_scheduler.StepLR(conv_optimiser,step_size = 4, gamma = 0.1,last_epoch = start_epoch-1)

	for epoch_i in range(start_epoch, epochs):
		print("\nepoch : ",epoch_i)
		Avg_loss = []
		model.train(True)
		lrScheduler.step(epoch = epoch_i)
		start =  time()
		Total_loss = 0.0

		for i,(image,BB_features) in enumerate(Train_data):
			BB_features = np.array([ x.numpy() for x in  BB_features]).T
			if is_gpu_available:
				x = Variable(image.cuda(0))
				y = Variable(torch.FloatTensor(BB_features).cuda(0))
			else:
				x = Variable(image)
				y = Variable(torch.FloatTensor(BB_features))

			if len(y) == batchsize:
				y_pred  = model(x)
				loss = 0.0
				loss+= 0.8 * criterion.cuda()(y_pred[:,:2],y[:,:2])
				loss+= 0.2 * criterion.cuda()(y_pred[:,2:],y[:,2:])
				Avg_loss.append(loss.item() / batchsize)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

			if i % 40 == 0:
				print('trained %s images in %s seconds, current loss :  %s' % ( (i + 1)*batchsize, time() - start, sum(Avg_loss[-40:]) / min(i+1,40) ))

			if (i* batchsize) % 24000 == 0:
				no_of_image = i* batchsize 
				torch.save(model,"model__raat_" + str(epoch_i) + "_" + str(no_of_image))
				print("model saved")

		
		print('trained full images in %s seconds, Total_loss :  %s' % ( time() - start, float(sum(Avg_loss))/len(Avg_loss) ))
		torch.save(model,"model__raat_" + str(epoch_i))
		print("model saved")

def test_model_BB(model,threshold = 0.7):
	print('threshold',threshold)
	model.train(False)
	start =  time()
	correct = 0.0
	not_correct = 0.0
	zero_tensor = torch.tensor(0.0)
	for i,(image,BB_features) in enumerate(Test_data):

		BB_features = np.array([ x.numpy() for x in  BB_features]).T
		if is_gpu_available:
			zero_tensor = Variable(zero_tensor.cuda(0))
			x = Variable(image.cuda(0))
			y = Variable(torch.FloatTensor(BB_features).cuda(0))
		else:
			x = Variable(image)
			y = Variable(torch.FloatTensor(BB_features))

		y_pred  = model(x)

		Area_real = y[:,2] * y[:,3]
		Area_mine = y_pred[:,2] * y_pred[:,3]

		common_x_left =   torch.max(y[:,0]-y[:,2]/2.0,y_pred[:,0]-y_pred[:,2]/2.0)
		common_x_right =  torch.min(y[:,0]+y[:,2]/2.0,y_pred[:,0]+y_pred[:,2]/2.0)
		common_y_uper =   torch.min(y[:,1]+y[:,3]/2.0,y_pred[:,1]+y_pred[:,3]/2.0)
		common_y_nichey = torch.max(y[:,1]-y[:,3]/2.0,y_pred[:,1]-y_pred[:,3]/2.0)
		comman_x = torch.max(zero_tensor, common_x_right - common_x_left)
		comman_y = torch.max(zero_tensor, common_y_uper - common_y_nichey)
		comman_area = comman_x * comman_y
		IoU = comman_area / (Area_real + Area_mine - comman_area)

		correct +=(IoU >= threshold).sum().item()
		not_correct +=(IoU < threshold).sum().item()
		if i % 20 ==0:
			print("i, correct, wrong, percentage : ",i, correct, not_correct,correct * 100.0 /(correct + not_correct))

	print("correct      : ",correct)
	print("Not correct  : ",not_correct)
	print("Accuracy     : ",correct * 100.0 /(correct + not_correct),"%")

def make_BB(model):
	Demo_data_filename = DemoTestDataLoader(['Testing_final'],img_size)
	Demo_Test_data = DataLoader(Demo_data_filename,batch_size=1,shuffle = False,num_workers=8)
	model.train(False)
	start =  time()
	zero_tensor = torch.tensor(0.0)
	for i,(image,img_name,BB_features) in enumerate(Demo_Test_data):
		BB_features = np.array([ x.numpy() for x in  BB_features]).T
		if is_gpu_available:
			zero_tensor = Variable(zero_tensor.cuda(0))
			x = Variable(image.cuda(0))
			y = Variable(torch.FloatTensor(BB_features).cuda(0))
		else:
			x = Variable(image)
			y = Variable(torch.FloatTensor(BB_features))

		y_pred  = model(x)

		img = cv2.imread(img_name[0]) 

		x_left_mine = (y_pred[:,0]-y_pred[:,2]/2.0 ) * img.shape[1]
		x_right_mine = (y_pred[:,0]+y_pred[:,2]/2.0 ) * img.shape[1]
		y_niche_mine = (y_pred[:,1]-y_pred[:,3]/2.0 ) * img.shape[0]
		y_uper_mine =  (y_pred[:,1]+y_pred[:,3] /2.0) * img.shape[0]

		x_left_real = (y[:,0]-y[:,2]/2.0 ) * img.shape[1]
		x_right_real = (y[:,0]+y[:,2]/2.0 ) * img.shape[1]
		y_niche_real = (y[:,1]-y[:,3]/2.0 ) * img.shape[0]
		y_uper_real =  (y[:,1]+y[:,3] /2.0) * img.shape[0]

		cv2.rectangle(img,(int(x_left_mine),int(y_uper_mine)),(int(x_right_mine),int(y_niche_mine)),color = (0,255,0),thickness =3 )
		cv2.rectangle(img,(int(x_left_real),int(y_uper_real)),(int(x_right_real),int(y_niche_real)),color = (0,0,255),thickness =3 )
		cv2.imwrite(img_name[0],img)


def test_model_final(model,file_names):
	Final_Test_data_filename = Label_LP_TestDataLoader(file_names,img_size)
	Final_Test_data = DataLoader(Final_Test_data_filename,batch_size=batchsize,shuffle = False,num_workers=8)

	model.train(False)
	start =  time()
	correct = 0
	total_no = 0

	for i,(image,LP_number) in enumerate(Final_Test_data):

		real_char = [[int(c) for c in cc.split('_')[:7]] for cc in LP_number]
		real_char = np.array(real_char)

		x = Variable(image.cuda(0))

		try:
			_,char_pred = model(x)
		except:
			print("error in try-test_model_final")
			continue
		y_pred_cpu = [ch.data.cpu().numpy() for ch in char_pred]
		y_pred_argmax = np.array([ ch.argmax(1) for  ch in y_pred_cpu]).transpose()
		correct += ((y_pred_argmax == real_char).sum(1) == 7).sum()
		total_no += batchsize

		if i%20 == 0:
			print("i, correct, total,percentage : ",i, correct, total_no,correct * 100.0 /total_no)

def train_final(model,criterion_final,criterion_bb,optimiser_final,epochs = 10,start_epoch =0):

	if start_epoch > 0 :
		# lrScheduler_final  = torch.optim.lr_scheduler.StepLR(optimiser_final,step_size = 4, gamma = 0.1)
		lrScheduler_final  = torch.optim.lr_scheduler.StepLR(optimiser_final,step_size = 4, gamma = 0.1,last_epoch = start_epoch-1)

	for epoch_i in range(start_epoch,epochs):
		correct = 0
		print("\nepoch : ",epoch_i)
		Avg_loss = []
		model.train(False)
		# lrScheduler_final.step(epoch = epoch_i)
		lrScheduler_final.step()
		start = time()
		Total_loss = 0.0

		for i,(image, BB_features,LP_number) in enumerate(Final_Train_data):

			if i > 960:
				break
			
			BB_features = np.array([ x.numpy() for x in  BB_features]).T

			if is_gpu_available:
				x = Variable(image.cuda(0))
			else:
				x = Variable(image)

			if len(x) == batchsize:
				try:
					y_pred,char_pred = model(x)
				except:
					print("error in try-exceptttty")
					continue

				loss = 0.0

				real_char = [[int(c) for c in cc.split('_')[:7]] for cc in LP_number]

				for c in range(7):
					char_loss = Variable(torch.LongTensor([ch[c] for ch in real_char]).cuda(0))
					loss +=criterion_final(char_pred[c],char_loss)

				Avg_loss.append(loss.item() / batchsize)

				optimiser_final.zero_grad()
				loss.backward()
				optimiser_final.step()

				y_pred_cpu = [ch.data.cpu().numpy() for ch in char_pred]
				y_pred_argmax = [ ch.argmax(1) for  ch in y_pred_cpu ] #7 * batchsize
				y_pred_argmax = np.array(y_pred_argmax).transpose() #batchsize*7
				real_char = np.array(real_char)
				correct += ((y_pred_argmax == real_char).sum(1) == 7).sum() 
				
				if i % 80 == 0:
					print('trained %s images in %s seconds,where %s correct (%s) %%  current loss :  %s' % ( (i + 1)*batchsize , time() - start, correct,correct*100/((i + 1)*batchsize),sum(Avg_loss[-80:]) / min(i+1,80) ))
		print('trained full images in %s seconds, Total_loss :  %s' % ( time() - start, float(sum(Avg_loss))/len(Avg_loss) ))
		torch.save(model,"model__prev_night_full10K")
		print("model saved")


myFinal_model = torch.load("model__prev_night_full10K")

loss_criterion_BB = nn.L1Loss()
loss_criterion_final = nn.CrossEntropyLoss()
optimiser_final = torch.optim.SGD(myFinal_model.parameters(),lr=0.000002,momentum=0.9)
lrScheduler_final    = torch.optim.lr_scheduler.StepLR(optimiser_final,step_size = 3, gamma = 0.6)

if is_gpu_available:
	myFinal_model = torch.nn.DataParallel(myFinal_model,device_ids=range(torch.cuda.device_count())).cuda()

train_final(myFinal_model,loss_criterion_final,loss_criterion_BB,optimiser_final,epochs = 100,start_epoch=1)
