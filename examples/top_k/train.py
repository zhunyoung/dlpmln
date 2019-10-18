import sys
sys.path.append("../../")
sys.path.append("../")
from dlpmln import DeepLPMLN
import torch
from torch.autograd import Variable
import numpy as np
import time
from network import FC
from dataGen import KsData
import random


dprogram='''
% define k 
#const k = 7.

topk(k).
nn(m(k,10), in, [t,f]) :- topk(k).

% we make a mistake if the total weight of the chosen items exceeds maxweight 
mistake :- #sum{1, I : in(k,I,t)} > k.
'''

dprogram_test='''
% define k 
#const k = 7.

topk(k).
% we make a mistake if the total weight of the chosen items exceeds maxweight 
mistake :- #sum{1, I : in(k,I,t)} > k.
'''

m = FC(10, *[50, 50, 50, 50, 50], 10)

nnMapping = {'m': m}

optimizer = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}

dlpmlnObj = DeepLPMLN(dprogram, nnMapping, optimizer)

dataset = KsData("data/data.txt",10)
# print(dataset.train_labels.shape)
# print(dataset.train_labels[0])
# sys.exit()
dataList = []
obsList = []

for i, d in enumerate(dataset.train_data):
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    dataList.append({"k": d_tensor})

    
with open("data/evidence_train.txt", 'r') as f:
    obsList = f.read().strip().strip("#evidence").split("#evidence")


# testing 

testData = []
testObsLost = []

for d in dataset.test_data:
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    testData.append({"k": d_tensor})
    

with open("data/evidence_test.txt", 'r') as f:
    testObsLost = f.read().strip().strip("#evidence").split("#evidence")


for i in range(200):
	dlpmlnObj.learn(dataList = dataList, obsList = obsList, epoch=1, opt=True, storeSM=True)
	dlpmlnObj.testConstraint(testData, testObsLost,[dprogram_test])