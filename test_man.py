from dlpmln import deeplpmln
from digit_network import Net, device
import torch
# import torch.nn.functional as F
# import torch.optim as optim
from torchvision import datasets, transforms
import sys


dprogram = '''
img(i1). img(i2).

addition(A,B,N) :- digit(A,1,N1), digit(B,1,N2), N=N1+N2.

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

# is this __name__ necessary?
m.__name__ = "m"

optimizer = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}

dlpmlnObj = deeplpmln(dprogram, [m])

obs = [""]

data_m = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
# data_m = torch.Tensor(data_m)

# print(data_m.size())

data = {'m':train_loader}

dlpmlnObj.learn(data, obs, optimizer, 1)


# print(dlpmlnObj.mvpp)
