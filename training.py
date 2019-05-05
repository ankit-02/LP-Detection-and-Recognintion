import torch
import torch.nn as nn
from load_data import *
from time import time
from torch.autograd import Variable

is_gpu_available = torch.cuda.is_available()
# is_gpu_available = False
print(is_gpu_available)
if is_gpu_available:
	print("torch cuda device count",torch.cuda.device_count())

no_of_output = 4
# img_size = (512,512)
img_size = (480,480)
# batchsize = 8
# batchsize = 10
batchsize = 25
print("batchsize : ",batchsize)
epochs = 10
# model_name = 

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

		# self.classifier = nn.Sequential(
		#     nn.Linear(23232, no_of_output_class),
		# )

	def forward(self, x):
		x1 = self.features(x)
		x11 = x1.view(x1.size(0), -1)
		x = self.classifier(x11)
		return x

myModel = BB_model(no_of_output)

if is_gpu_available:
	myModel = torch.nn.DataParallel(myModel,device_ids=range(torch.cuda.device_count())).cuda()

def get_no_of_parameters(model):
	no_of_parameters = 0
	for x in list(model.parameters()):
		print(' x-size : ', x.size())
		multiplicity = 1
		for y in list(x.size()):
			# print(' y : ', y)
			multiplicity*=y
		no_of_parameters+=multiplicity
	return no_of_parameters

# print(myModel)
# print(get_no_of_parameters(myModel))

print("optimising")
# loss_criterion = nn.MSELoss()
loss_criterion = nn.L1Loss()

#can change lr for different training
conv_optimiser = torch.optim.SGD(myModel.parameters(),lr=0.001,momentum=0.9)
lrScheduler    = torch.optim.lr_scheduler.StepLR(conv_optimiser,step_size = 5, gamma = 0.1)

print("loading data")
# Train_data_filename = Label_BB_TestDataLoader(['ccpd_rotate'],img_size)
# Train_data_filename = Label_BB_TestDataLoader(['base_10k'],img_size)
Train_data_filename = Label_BB_TestDataLoader(['base_10k','base_40k','ccpd_rotate','ccpd_tilt','base_50k','ccpd_db','ccpd_fn','ccpd_weather','base_30k',"base_25k_Second"],img_size)
# Train_data_filename = Label_BB_TestDataLoader(['base_10k','base_40k','ccpd_rotate','ccpd_tilt','base_50k','ccpd_db','ccpd_fn','ccpd_weather','base_30k'],img_size)

# Test_data_filename = Label_BB_TestDataLoader(['ccpd_challenge'],img_size)

print("loading data ...")
# Train_data = DataLoader(Train_data,batch_size=batchsize)
Train_data = DataLoader(Train_data_filename,batch_size=batchsize,shuffle = True,num_workers=8)
# Test_data = DataLoader(Test_data_filename,batch_size=batchsize,shuffle = False,num_workers=8)
print("loaded")

def train_model(model,criterion,optimiser,epochs):
	#change starting epoch for saved models
	for epoch_i in range(epochs):
		print("\nepoch : ",epoch_i)
		Avg_loss = []
		#because of dropout
		model.train(True)
		lrScheduler.step()
		start =  time()
		Total_loss = 0.0

		#can return index for enumerate
		# can use separate variable i rather than enumerate
		for i,(image,BB_features) in enumerate(Train_data):
			# print('%s %s' % (i, time()-start))
			BB_features = np.array([ x.numpy() for x in  BB_features]).T
			if is_gpu_available:
				x = Variable(image.cuda(0))
				y = Variable(torch.FloatTensor(BB_features).cuda(0),requires_grad=False)
			else:
				x = Variable(image)
				y = Variable(torch.FloatTensor(BB_features), requires_grad = False)

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

			if (i* batchsize) % 20000 == 0:
				no_of_image = i* batchsize 
				torch.save(model,"model_" + str(epoch_i) + "_" + str(no_of_image))
				print("model saved")

		
		print('trained full images in %s seconds, Total_loss :  %s' % ( time() - start, float(sum(Avg_loss))/len(Avg_loss) ))
		torch.save(model,"model_" + str(epoch_i))
		print("model saved")

def test_model_BB(model,threshold = 0.7):
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
			y = Variable(torch.FloatTensor(BB_features).cuda(0),requires_grad=False)
		else:
			x = Variable(image)
			y = Variable(torch.FloatTensor(BB_features), requires_grad = False)

		y_pred  = model(x)

		Area_real = y[:,2] * y[:,3]
		Area_mine = y_pred[:,2] * y_pred[:,3]
		x_left_real = y[:,0]-y[:,2]/2.0
		x_left_mine = y_pred[:,0]-y_pred[:,2]/2.0
		x_right_real = y[:,0]+y[:,2]/2.0
		x_right_mine = y_pred[:,0]+y_pred[:,2]/2.0

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

	print("correct      : ",correct)
	print("Not correct  : ",not_correct)
	print("Accuracy     : ",correct * 100.0 /(correct + not_correct),"%")

# myModel = torch.load("model2")
myModel_Trained = train_model(myModel,loss_criterion,conv_optimiser,epochs=epochs)

# test_model_BB(myModel,0.5)





