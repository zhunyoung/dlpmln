import sys
sys.path.append("../../")
import time

import torch

from dlpmln import DeepLPMLN
from network import FC
from dataGen import obsList, obsListTest, dataList, dataListTest

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
# Construct nnMapping, set optimizers, and initialize DeepLPMLN object
########

m = FC(40, 50, 50, 50, 50, 50, 24)
nnMapping = {"m":m}
optimizers = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}

# 1234
# dlpmlnObj = DeepLPMLN(nnRule+aspRule+remove_con+path_con+reach_con+opt_con, nnMapping, optimizers)
# 234
dlpmlnObj = DeepLPMLN(nnRule+aspRule+path_con+reach_con+opt_con, nnMapping, optimizers)
# 23
# dlpmlnObj = DeepLPMLN(nnRule+aspRule+path_con+reach_con, nnMapping, optimizers)
# 2
# dlpmlnObj = DeepLPMLN(nnRule+aspRule+path_con, nnMapping, optimizers)


########
# Start training and testing on a list of different MVPP programs
########
mvppList = [remove_con, path_con, reach_con, remove_con+path_con, remove_con+reach_con, path_con+reach_con, remove_con+path_con+reach_con, remove_con+path_con+reach_con+opt_con]
mvppList = [aspRule+i for i in mvppList]

print('-------------------')
for idx, constraint in enumerate(mvppList):
    print('Constraint {} is\n{}\n-------------------'.format(idx+1, constraint))

startTime = time.time()
for i in range(500):
    print('Epoch {}...'.format(i+1))
    time1 = time.time()
    dlpmlnObj.learn(dataList=dataList, obsList=obsList, epoch=1, opt=True, storeSM=True)
    time2 = time.time()
    dlpmlnObj.testConstraint(dataList=dataListTest, obsList=obsListTest, mvppList=mvppList)
    print("--- train time: %s seconds ---" % (time2 - time1))
    print("--- test time: %s seconds ---" % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )


