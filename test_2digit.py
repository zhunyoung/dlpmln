from dlpmln import DeepLPMLN
from digit_network import Net, device
import torch
from torchvision import datasets, transforms
import time


dprogram = '''
img(i1; i2; i3; i4). 
addition(i1,i2,i3,i4,N) :- digit(i1,0,N1), digit(i2,0,N2), digit(i3,0,N3), digit(i4,0,N4), N=N1*10+N2 + N3*10+N4.
nn(m(X,1), digit, [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=2, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=1000, shuffle=True, **kwargs)

m = Net().to(device)
functions = {'m':m}
optimizers = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}


dataList = []
obsList = []
for batch in train_loader:
	dataList.append({"i1":batch[0][0].view(1, 1, 28, 28), "i2":batch[0][1].view(1, 1, 28, 28)})
	obsList.append(":- not addition(i1, i2, {}).".format( batch[1][0]+batch[1][1]))

# dataList = [{"i1":batch[0][0].view(1, 1, 28, 28), "i2":batch[0][1].view(1, 1, 28, 28)} for batch in train_loader]
# obsList = [":- not addition(i1, i2, {}).".format( batch[1][0]+batch[1][1]) for batch in train_loader]

dlpmlnObj = DeepLPMLN(dprogram, functions, optimizers)

for i in range(1):
	time1 = time.time()
	dlpmlnObj.learn(dataList=dataList, obsList=obsList, epoch=1)
	time2 = time.time()
	dlpmlnObj.testNN("m", test_loader)
	print("--- train time: %s seconds ---" % (time2 - time1))
	print("--- test time: %s seconds ---" % (time.time() - time2))