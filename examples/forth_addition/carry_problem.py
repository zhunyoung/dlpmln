import sys
sys.path.append("../../")
sys.path.append("../")
from dlpmln import DeepLPMLN
import torch

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from data_function import create_data_sample,format_dataList,format_observations,add_test



dprogram='''
num1(1,num1_1).
num1(0,num1_0).
num2(1,num2_1).
num2(0,num2_0).
1{carry(0,Carry): Carry=0..1}1.
nn(m1(A, B, Carry, 1), result, [0,1,2,3,4,5,6,7,8,9]) :- num1(P,A), num2(P,B), Carry=0..1.
nn(m2(A, B, Carry, 1), carry, [0,1]) :- num1(P,A), num2(P,B), Carry=0..1.

result(P,X) :- num1(P, A), num2(P, B), carry(P, Carry), result(A,B,Carry,0,X).
carry(P+1,X) :- num1(P, A), num2(P, B), carry(P, Carry), carry(A,B,Carry,0,X).
'''




class FC(nn.Module):

    def __init__(self, *sizes):
        super(FC, self).__init__()
        layers = []
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Softmax(1))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


m1 = FC(30,25,10) # network for adding the numbers
m2 = FC(30,5,2)   # network for finding the carry out 



functions = {'m1':m1, 'm2':m2}

optimizers = {'m1':torch.optim.Adam(m1.parameters(), lr=0.01),'m2':torch.optim.Adam(m2.parameters(), lr=0.01)}



dataList = []
obsList = []


train_size=1000
test_size=100

add_test_dataset=add_test(test_size)

add_test_dataloader=DataLoader(add_test_dataset,batch_size=4,shuffle=True)


for i in range(train_size):
    
    obs,str_list=create_data_sample()
    
    dataList.append(format_dataList(obs,str_list))
    obsList.append(format_observations(obs,str_list))



dlpmlnObj = DeepLPMLN(dprogram, functions, optimizers, dynamicMVPP=False)
# dlpmlnObj.device='cpu'
# print(dlpmlnObj.mvpp['program'])
# print('k', dlpmlnObj.k)
# print('nnOutputs', dlpmlnObj.nnOutputs)
# print('functions', dlpmlnObj.functions)
# print(dlpmlnObj.const)
# sys.exit()

print('training...')

# print(dataList[0])
# sys.exit()

for i in range(5):
	time1 = time.time()
	dlpmlnObj.learn(dataList=dataList, obsList=obsList, epoch=1)
	time2 = time.time()
	dlpmlnObj.testNN("m1", add_test_dataloader) #test m1 network
	#dlpmlnObj.testNN("m2", test_loader) #test m2 network
	print("--- train time: %s seconds ---" % (time2 - time1))
	print("--- test time: %s seconds ---" % (time.time() - time2))















