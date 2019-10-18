import sys
sys.path.append("../../")
import time

import torch

from dlpmln import DeepLPMLN
from dataGen import fileToLists, trainLoader, testLoader
from network import Net, FC


######################################
# The dlpmln program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram1 = '''
img(i).
color(color1; color2).

nn(m1(X,1), neural_coin, [head, tail]) :- img(X).
nn(m2(X,1), neural_color, [red, green, blue]) :- color(X).

% we win if both balls have the same color
win :- neural_color(color1, 0, C), neural_color(color2, 0, C).

% or if the coin came up heads and we have at least one red ball
win :- neural_coin(i, 0, head), 1{neural_color(color1, 0, red); neural_color(color2, 0, red)}.

loss :- not win.
'''

dprogram2 = '''
img(i).
color(color1; color2).

nn(m1(X,1), neural_coin, [head, tail]) :- img(X).
nn(m2(X,1), neural_color, [red, green, blue]) :- color(X).

% we simultaneously learn the probability of getting head for the coin and 
% learn the probability of each color for each urn
@0.5 random_coin(i, head); @0.5 random_coin(i, tail).
@0.5 random_color(color1, red); @0.5 random_color(color1, blue).
@0.33 random_color(color2, red); @0.33 random_color(color2, green); @0.33 random_color(color2, blue).


coin(HT) :- neural_coin(i, HT), random_coin(i, HT).
urn1(C) :- neural_color(color1, 0, C), random_color(color1, C).
urn2(C) :- neural_color(color2, 0, C), random_color(color2, C).

% we win if both balls have the same color
win :- urn1(C), urn2(C).

% or if the coin came up heads and we have at least one red ball
win :- coin(head), 1{urn1(red); urn2(red)}.

loss :- not win.

% we make a mistake if 2 predictions violate
mistake :- neural_coin(i, 0, HT1), random_coin(i, HT2), HT1!=HT2.
mistake :- neural_color(C, 0, C1), random_color(C, C2), C1!=C2.
'''

########
# Define nnMapping and optimizers, initialze DeepLPMLN object
########
model1 = Net()
model2 = FC(3,3)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1.0)

nnMapping = {'m1': model1, 'm2': model2}
optimizers = {'m1': optimizer1, 'm2': optimizer2}

dlpmlnObj = DeepLPMLN(dprogram2, nnMapping, optimizers)


########
# Define dataList, obsList, and the dataset to test m2
########

dataList, obsList, m1dataset, m2dataset = fileToLists('./data/coinUrn_train.txt', trainLoader)

print(trainLoader[0][0].size())
print(trainLoader[0][1])
sys.exit()

########
# Start training and testing
########

startTime = time.time()
for i in range(100):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    dlpmlnObj.learn(dataList=dataList, obsList=obsList, epoch=1, opt=False, storeSM=False)
    print(dlpmlnObj.normalProbs)
    time2 = time.time()
    dlpmlnObj.testNN("m1", m1dataset)
    dlpmlnObj.testNN("m2", m2dataset)
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )

# TEST
# dlpmlnObj.testConstraint(dataList=dataList, obsList=obsList, mvppList=[':~ a. [0]'])
# test = dlpmlnObj.infer(dataDic=dataList[0], obs='', mvpp='')
# print(test)