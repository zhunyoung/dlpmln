import sys
import time
import random

import numpy as np
from numpy.random import permutation
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from dlpmln import DeepLPMLN

#############################
# DeepLPMLN program
#############################

dprogram = '''
grid(g).
nn(m(g, 24), nn_edge, [t, f]) :- grid(g). 

nn_edge(X) :- nn_edge(g,X,t).

sp(0,1) :- nn_edge(0).
sp(1,2) :- nn_edge(1).
sp(2,3) :- nn_edge(2).
sp(4,5) :- nn_edge(3).
sp(5,6) :- nn_edge(4).
sp(6,7) :- nn_edge(5).
sp(8,9) :- nn_edge(6).
sp(9,10) :- nn_edge(7).
sp(10,11) :- nn_edge(8).
sp(12,13) :- nn_edge(9).
sp(13,14) :- nn_edge(10).
sp(14,15) :- nn_edge(11).
sp(0,4) :- nn_edge(12).
sp(4,8) :- nn_edge(13).
sp(8,12) :- nn_edge(14).
sp(1,5) :- nn_edge(15).
sp(5,9) :- nn_edge(16).
sp(9,13) :- nn_edge(17).
sp(2,6) :- nn_edge(18).
sp(6,10) :- nn_edge(19).
sp(10,14) :- nn_edge(20).
sp(3,7) :- nn_edge(21).
sp(7,11) :- nn_edge(22).
sp(11,15) :- nn_edge(23).

sp(X,Y) :- sp(Y,X).
'''

remove_con = '''
% [nr] 1. No removed edges should be predicted
mistake :- nn_edge(X), removed(X).
'''

path_con = '''
% [p] 2. Prediction must form simple path(s)
% that is: the degree of nodes should be either 0 or 2
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
'''

reach_con = '''
% [r] 3. Every 2 nodes in the prediction must be reachable
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).
'''

opt_con = '''
% [o] 4. Predicted path should contain least edges
:~ nn_edge(X). [1, X]
'''

#############################
# Neural network model
#############################

class FC(nn.Module):

    def __init__(self, *sizes):
        super(FC, self).__init__()
        layers = []
        print("Neural Network (MLP) Structure: {}".format(sizes))
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Sigmoid())
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)

########
# Construct model and set optimizer
########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = FC(40, 50, 50, 50, 50, 50, 24).to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=0.001)


######################################
# Generate Dataset
######################################

class GridData():
    def __init__(self, data_path):
        np.random.seed(0)
        data = []
        labels = []
        with open(data_path) as file:
            for line in file:
                tokens = line.strip().split(',')
                if(tokens[0] != ''):
                    removed = [int(x) for x in tokens[0].split('-')]
                else:
                    removed = []

                inp = [int(x) for x in tokens[1].split('-')]
                paths = tokens[2:]
                # print(paths)
                # sys.exit()
                data.append(np.concatenate((to_one_hot(removed, 24, True), to_one_hot(inp, 16))))
                pathind = 0
                if len(paths) > 1:
                    pathind = random.randrange(len(paths))
                path = [int(x) for x in paths[0].split('-')]
                labels.append(to_one_hot(path, 24))


        # We're going to split 60/20/20 train/test/validation
        perm = permutation(len(data))
        train_inds = perm[:int(len(data)*0.6)]
        valid_inds = perm[int(len(data)*0.6):int(len(data)*0.8)]
        test_inds = perm[int(len(data)*0.8):]
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.train_data = self.data[train_inds, :]
        self.valid_data = self.data[valid_inds, :]
        self.test_data = self.data[test_inds, :]
        self.train_labels = self.labels[train_inds, :]
        self.valid_labels = self.labels[valid_inds, :]
        self.test_labels = self.labels[test_inds, :]

        # Count what part of the batch we're attempt
        self.batch_ind = len(train_inds)
        self.batch_perm = None
        np.random.seed()

        self.dic = {}
        self.dic["train"] = self.train_data
        self.dic["test"] = self.test_data
        self.dic["valid"] = self.valid_data

        self.dic["train_label"] = self.train_labels
        self.dic["test_label"] = self.test_labels

    # REMOVE?
    def get_batch(self, size):
        # If we're out:
        if self.batch_ind >= self.train_data.shape[0]:
            # Rerandomize ordering
            self.batch_perm = permutation(self.train_data.shape[0])
            # Reset counter
            self.batch_ind = 0

        # If there's not enough
        if self.train_data.shape[0] - self.batch_ind < size:
            # Get what there is, append whatever else you need
            ret_ind = self.batch_perm[self.batch_ind:]
            d, l = self.train_data[ret_ind, :], self.train_labels[ret_ind, :]
            size -= len(ret_ind)
            self.batch_ind = self.train_data.shape[0]
            nd, nl = self.get_batch(size)
            return np.concatenate(d, nd), np.concatenate(l, nl)

        # Normal case
        ret_ind = self.batch_perm[self.batch_ind: self.batch_ind + size]
        return self.train_data[ret_ind, :], self.train_labels[ret_ind, :]

def to_one_hot(dense, n, inv=False):
    one_hot = np.zeros(n)
    one_hot[dense] = 1
    if inv:
        one_hot = (one_hot + 1) % 2
    return one_hot

def generateDataset(inPath, outPath):
    grid_data = GridData(inPath)
    names = ["train", "test", "valid"]
    for name in names:
        fname = outPath+name+".txt"
        # erase the content of each file
        # open(fname, 'w').close()
        with open(fname, 'w') as f:
            for data in grid_data.dic[name].tolist():
                removed = data[:24]
                startEnd = data[24:]
                # print(removed)
                # print(startEnd)
                removed = [i for i, x in enumerate(removed) if x == 0]
                startEnd = [i for i, x in enumerate(startEnd) if x == 1]
                evidence = ":- mistake.\n"
                for edge in removed:
                    evidence += "removed({}).\n".format(edge)
                for node in startEnd:
                    evidence += "sp(external, {}).\n".format(node)
                # print(evidence)
                f.write(evidence)
                f.write("#evidence\n")
    return grid_data.dic

dataset = generateDataset("data/shortestPath.data", "evidence/shorteatPath_")


sys.exit()








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

dataList = []
# obstxt = ""
obsList = []
for dataIdx, data in enumerate(train_loader):
	dataList.append({"m":{"i1":data[0][0].view(1, 1, 28, 28), "i2":data[0][1].view(1, 1, 28, 28)}})
	obsList.append(":- not addition(i1, i2, {}).".format( data[1][0]+data[1][1]))
# 	obstxt += "addition(i1, i2, {}).\n#evidence\n".format( data[1][0]+data[1][1])
	
# with open("evidence.txt", "w") as f:
# 	f.write(obstxt)


dlpmlnObj = DeepLPMLN(dprogram, nnDic, optimizer)

for i in range(1):
	time1 = time.time()
	dlpmlnObj.learn(dataList=dataList, obsList=obsList, epoch=1)
	time2 = time.time()
	dlpmlnObj.test_nn("m", test_loader)
	print("--- train time: %s seconds ---" % (time2 - time1))
	print("--- test time: %s seconds ---" % (time.time() - time2))

# print(dlpmlnObj.mvpp)
