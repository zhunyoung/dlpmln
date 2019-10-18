import sys
sys.path.append("../../")
from dlpmln import DeepLPMLN
import torch
from torch.autograd import Variable
import numpy as np
import time
from network import FC
from dataGen import GridProbData
import random

dprogram = '''
grid(g).
nn(m(g,24), nn_edge, [t,f]) :- grid(g).
sp(0,1) :- nn_edge(g, 0, t).
sp(1,2) :- nn_edge(g, 1, t).
sp(2,3) :- nn_edge(g, 2, t).
sp(4,5) :- nn_edge(g, 3, t).
sp(5,6) :- nn_edge(g, 4, t).
sp(6,7) :- nn_edge(g, 5, t).
sp(8,9) :- nn_edge(g, 6, t).
sp(9,10) :- nn_edge(g, 7, t).
sp(10,11) :- nn_edge(g, 8, t).
sp(12,13) :- nn_edge(g, 9, t).
sp(13,14) :- nn_edge(g, 10, t).
sp(14,15) :- nn_edge(g, 11, t).
sp(0,4) :- nn_edge(g, 12, t).
sp(4,8) :- nn_edge(g, 13, t).
sp(8,12) :- nn_edge(g, 14, t).
sp(1,5) :- nn_edge(g, 15, t).
sp(5,9) :- nn_edge(g, 16, t).
sp(9,13) :- nn_edge(g, 17, t).
sp(2,6) :- nn_edge(g, 18, t).
sp(6,10) :- nn_edge(g, 19, t).
sp(10,14) :- nn_edge(g, 20, t).
sp(3,7) :- nn_edge(g, 21, t).
sp(7,11) :- nn_edge(g, 22, t).
sp(11,15) :- nn_edge(g, 23, t).
sp(X,Y) :- sp(Y,X).
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).
'''

dprogram_test = '''
grid(g).
sp(0,1) :- nn_edge(g, 0, t).
sp(1,2) :- nn_edge(g, 1, t).
sp(2,3) :- nn_edge(g, 2, t).
sp(4,5) :- nn_edge(g, 3, t).
sp(5,6) :- nn_edge(g, 4, t).
sp(6,7) :- nn_edge(g, 5, t).
sp(8,9) :- nn_edge(g, 6, t).
sp(9,10) :- nn_edge(g, 7, t).
sp(10,11) :- nn_edge(g, 8, t).
sp(12,13) :- nn_edge(g, 9, t).
sp(13,14) :- nn_edge(g, 10, t).
sp(14,15) :- nn_edge(g, 11, t).
sp(0,4) :- nn_edge(g, 12, t).
sp(4,8) :- nn_edge(g, 13, t).
sp(8,12) :- nn_edge(g, 14, t).
sp(1,5) :- nn_edge(g, 15, t).
sp(5,9) :- nn_edge(g, 16, t).
sp(9,13) :- nn_edge(g, 17, t).
sp(2,6) :- nn_edge(g, 18, t).
sp(6,10) :- nn_edge(g, 19, t).
sp(10,14) :- nn_edge(g, 20, t).
sp(3,7) :- nn_edge(g, 21, t).
sp(7,11) :- nn_edge(g, 22, t).
sp(11,15) :- nn_edge(g, 23, t).
sp(X,Y) :- sp(Y,X).
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).
'''

m = FC(40, *[50, 50, 50, 50, 50], 24)

nnMapping = {'m': m}

optimizer = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}

dlpmlnObj = DeepLPMLN(dprogram, nnMapping, optimizer)


# process the data 
dataset = GridProbData("data/data.txt")

# print(dataset.valid_data.shape)
# print(dataset.train_labels[0])
# sys.exit()

dataList = []
obsList = []

for i, d in enumerate(dataset.train_data):
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    dataList.append({"g": d_tensor})

with open("data/evidence_train.txt", 'r') as f:
    obsList = f.read().strip().strip("#evidence").split("#evidence")

# testing 
testData = []
testObsLost = []

for d in dataset.test_data:
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    testData.append({"g": d_tensor})

with open("data/evidence_test.txt", 'r') as f:
    testObsLost = f.read().strip().strip("#evidence").split("#evidence")

for i in range(200):
	dlpmlnObj.learn(dataList = dataList, obsList = obsList, epoch=1, opt=True, storeSM=True)
	dlpmlnObj.testConstraint(testData, testObsLost,[dprogram_test])