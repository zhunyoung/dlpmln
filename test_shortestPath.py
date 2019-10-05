import random
import sys
import time

import numpy as np
from numpy.random import permutation
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable

from dlpmln import DeepLPMLN

#############################
# DeepLPMLN program
#############################

nnRule = '''
grid(g).
nn(m(g, 24), nn_edge, [t, f]) :- grid(g). 
'''

aspRule = '''
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


########
# Construct nnMapping and set optimizers
########

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = FC(40, 50, 50, 50, 50, 50, 24).to(device)
nnMapping = {"m":m}
optimizers = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}

######################################
# Construct DataList and obsList
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
                data.append(np.concatenate((self.to_one_hot(removed, 24, inv=True), self.to_one_hot(inp, 16))))
                path = [int(x) for x in paths[0].split('-')]
                labels.append(self.to_one_hot(path, 24))


        # We're going to split 60/20/20 train/test/validation
        perm = permutation(len(data))
        train_inds = perm[:int(len(data)*0.6)]
        valid_inds = perm[int(len(data)*0.6):int(len(data)*0.8)]
        test_inds = perm[int(len(data)*0.8):]
        data = np.array(data)
        labels = np.array(labels)

        np.random.seed()

        self.dic = {}
        self.dic["train"] = data[train_inds, :]
        self.dic["test"] = data[test_inds, :]
        self.dic["valid"] = data[valid_inds, :]
        self.dic["train_label"] = labels[train_inds, :]
        self.dic["test_label"] = labels[test_inds, :]
        self.dic["valid_label"] = labels[valid_inds, :]

    @staticmethod
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
        with open(fname, 'w') as f:
            for data in grid_data.dic[name].tolist():
                removed = data[:24]
                startEnd = data[24:]
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

with open('evidence/shorteatPath_train.txt', 'r') as f:
    obsList = f.read().strip().strip("#evidence").split("#evidence")

with open('evidence/shorteatPath_test.txt', 'r') as f:
    obsListTest = f.read().strip().strip("#evidence").split("#evidence")

dataList = []
for data in dataset['train']:
    dataList.append({'g': Variable(torch.from_numpy(data).float(), requires_grad=False)})

dataListTest = []
for data in dataset['test']:
    dataListTest.append({'g': Variable(torch.from_numpy(data).float(), requires_grad=False)})


# 1234
# dlpmlnObj = DeepLPMLN(nnRule+aspRule+remove_con+path_con+reach_con+opt_con, nnMapping, optimizers)
# 234
dlpmlnObj = DeepLPMLN(nnRule+aspRule+path_con+reach_con+opt_con, nnMapping, optimizers)
# 23
# dlpmlnObj = DeepLPMLN(nnRule+aspRule+path_con+reach_con, nnMapping, optimizers)
# 2
# dlpmlnObj = DeepLPMLN(nnRule+aspRule+path_con, nnMapping, optimizers)

# dlpmlnObj = DeepLPMLN(dprogram, nnMapping, optimizers)
# print(dlpmlnObj.mvpp['program'])
# print(dlpmlnObj.functions)
# print(dlpmlnObj.nnOutputs)

mvppList = [remove_con, path_con, reach_con, remove_con+path_con, remove_con+reach_con, path_con+reach_con, remove_con+path_con+reach_con, remove_con+path_con+reach_con+opt_con]
# mvppList = [remove_con+path_con+reach_con+opt_con]
mvppList = [aspRule+i for i in mvppList]
# print(mvppList[0])

print('-------------------')
for idx, constraint in enumerate(mvppList):
    print('Constraint {} is\n{}\n-------------------'.format(idx+1, constraint))

startTime = time.time()
for i in range(500):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    dlpmlnObj.learn(dataList=dataList, obsList=obsList, epoch=1, opt=True)
    time2 = time.time()
    dlpmlnObj.testConstraint(dataList=dataListTest, obsList=obsListTest, mvppList=mvppList)
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )


