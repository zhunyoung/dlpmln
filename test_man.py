from dlpmln import DeepLPMLN
from digit_network import Net, device
import torch
import numpy as np
# import torch.nn.functional as F
# import torch.optim as optim
from torchvision import datasets, transforms
import sys


dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(A,0,N1), digit(B,0,N2), N=N1+N2.
nn(m(X,1), digit, [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

# data = [{'i1': }]
use_cuda = False

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=2, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=1000, shuffle=True, **kwargs)

m = Net().to(device)

nnDic = {"m":m}


optimizer = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}

obs = [""]

data_m = datasets.MNIST('../data', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

dataList = []
obstxt = ""
obsList = []
for dataIdx, data in enumerate(train_loader):
	# print("This is data 1:",data[0][0])
	# print("This is data 2:", data[0][1])
	# print(data[0].shape)
	# sys.exit()
	dataList.append({"m":{"i1":data[0][0].view(1, 1, 28, 28), "i2":data[0][1].view(1, 1, 28, 28)}})
	obsList.append(":- not addition(i1, i2, {}).".format( data[1][0]+data[1][1]))
	if dataIdx % 1000 == 0:
		obstxt += "addition(i1, i2, {}, {}).\n".format(data[1][0],data[1][1])
		obstxt += "#evidence\n"
		# if dataIdx == 10:
		# 	break
	
with open("evidence.txt", "w") as f:
	f.write(obstxt)


dlpmlnObj = DeepLPMLN(dprogram, nnDic, optimizer)

for i in range(2):
	print(i)
	dlpmlnObj.learn(dataList, obsList, 1)

	dlpmlnObj.test_nn("m", test_loader)

# print(dlpmlnObj.mvpp)