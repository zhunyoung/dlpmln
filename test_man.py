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

# data is a dictionary, where the keys are the name of neural network and the values are the corresponding input data. 
# obs is a list, in which each obs_i is relative to one data. 
# optimizer is also a dictionary, where the keys are the name of neural network and the values are the corresponding optimizer. 
def learn(self, data, obs, optimizer):
	# get the iteration by the length of data
	length = len(data[self.functions[0].__name__])

	# add one attribut, type, to self.func. 
	# since currently we don't have this att, I set the type of each functions be 10 in digit example, in general func.type = k
	for funcIdx in range(len(self.functions)):
		self.functions[funcIdx].type = 10
	
	# get the mvpp program by self.mvpp, so far self.mvpp is a string
	dmvpp = MVPP(self.mvpp)
	
	# get the parameters by the output of neural networks.
	for dataIdx in range(length):
		probs = []
		output = []
		for func in self.functions:
			print(data[func.__name__][dataIdx])
			sys.exit()
			output_func = func(data[func.__name__][dataIdx])
			output.append(output_func)
			if func.type > 2:

				probs.append(output_func)
			else:
				for para in output_func:
					probs.append([para, 1-para])

		# set the values of parameters of mvpp
		dmvpp.parameters = probs
		gradients = dmvpp.gradients_one_obs(obs[dataIdx])
		if device.type == 'cuda':
			grad_by_prob = -1 * torch.cuda.FloatTensor(gradients)
		else:
			grad_by_prob = -1 * torch.FloatTensor(gradients)
		
		for outIdx, out in enumerate(output):
			out.backward(grad_by_prob[outIdx], retain_graph=True)
			optimizer[self.functions[outIdx].__name__].step()
			optimizer[self.functions[outIdx].__name__].zero_grad()


