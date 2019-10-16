import sys
sys.path.append("../../")
import time

import torch
from torchvision import datasets, transforms

from dlpmln import DeepLPMLN
from network import Net

######################################
# The dlpmln program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(A,0,N1), digit(B,0,N2), N=N1+N2.
nn(m(X,1), digit, [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

########
# Define nnMapping and optimizers, initialze DeepLPMLN object
########

m = Net()
nnMapping = {'m':m}
optimizers = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}

dlpmlnObj = DeepLPMLN(dprogram, nnMapping, optimizers)

########
# Define dataList, obsList
########

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=2, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True, **kwargs)

dataList = []
obsList = []
for batch in train_loader:
    dataList.append({"i1":batch[0][0].view(1, 1, 28, 28), "i2":batch[0][1].view(1, 1, 28, 28)})
    obsList.append(":- not addition(i1, i2, {}).".format( batch[1][0]+batch[1][1]))

########
# Start training and testing
########

startTime = time.time()
for i in range(1):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    dlpmlnObj.learn(dataList=dataList, obsList=obsList, epoch=1)
    time2 = time.time()
    dlpmlnObj.testNN("m", test_loader)
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )



# TEST INFERENCE
# dlpmlnObj.testConstraint(dataList=dataList, obsList=obsList, mvppList=[':~ a. [0]'])
# test = dlpmlnObj.infer(dataDic=dataList[0], obs='', mvpp='')
# print(test)