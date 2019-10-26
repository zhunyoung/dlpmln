import sys
sys.path.append("../../")
import time

import torch

from dataGen import dataList, obsList, add_test_dataloader, carry_test_dataloader
from dlpmln import DeepLPMLN
from network import FC

######################################
# The dlpmln program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

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

########
# Define nnMapping and optimizers, initialze DeepLPMLN object
########

m1 = FC(30,25,10) # network for adding the numbers
m2 = FC(30,5,2)   # network for finding the carry out 

nnMapping = {'m1':m1, 'm2':m2}
optimizers = {'m1':torch.optim.Adam(m1.parameters(), lr=0.01),'m2':torch.optim.Adam(m2.parameters(), lr=0.01)}
dlpmlnObj = DeepLPMLN(dprogram, nnMapping, optimizers)

########
# Start training and testing
########

startTime = time.time()
for i in range(5):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    dlpmlnObj.learn(dataList=dataList, obsList=obsList, epoch=1)
    time2 = time.time()
    dlpmlnObj.testNN("m1", add_test_dataloader) #test m1 network
    dlpmlnObj.testNN("m2", carry_test_dataloader) #test m2 network
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )