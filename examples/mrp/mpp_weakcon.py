import sys
sys.path.append("../../")
from dlpmln import DeepLPMLN
import torch
from torch.autograd import Variable
import numpy as np
import time
from network import FC, device
from grid_data import GridProbData

dprogram = '''

grid(g).
nn(m(g,24), nn_edge, [t,f]) :- grid(g).
nn_edge(g, X):- nn_edge(g,X,t).
sp(0,1) :- nn_edge(g, 0).
sp(1,2) :- nn_edge(g, 1).
sp(2,3) :- nn_edge(g, 2).
sp(4,5) :- nn_edge(g, 3).
sp(5,6) :- nn_edge(g, 4).
sp(6,7) :- nn_edge(g, 5).
sp(8,9) :- nn_edge(g, 6).
sp(9,10) :- nn_edge(g, 7).
sp(10,11) :- nn_edge(g, 8).
sp(12,13) :- nn_edge(g, 9).
sp(13,14) :- nn_edge(g, 10).
sp(14,15) :- nn_edge(g, 11).
sp(0,4) :- nn_edge(g, 12).
sp(4,8) :- nn_edge(g, 13).
sp(8,12) :- nn_edge(g, 14).
sp(1,5) :- nn_edge(g, 15).
sp(5,9) :- nn_edge(g, 16).
sp(9,13) :- nn_edge(g, 17).
sp(2,6) :- nn_edge(g, 18).
sp(6,10) :- nn_edge(g, 19).
sp(10,14) :- nn_edge(g, 20).
sp(3,7) :- nn_edge(g, 21).
sp(7,11) :- nn_edge(g, 22).
sp(11,15) :- nn_edge(g, 23).

sp(X,Y) :- sp(Y,X).
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).

'''

m = FC(40, *[50, 50, 50, 50, 50], 24).to(device)

functions = {'m': m}

optimizer = {'m':torch.optim.Adam(m.parameters(), lr=0.001)}

dlpmlnObj = DeepLPMLN(dprogram, functions, optimizer)


# process the data 
dataset = GridProbData("data/data.txt")
dataList = []
obsList = []

for d in dataset.train_data:
    d_tensor = Variable(torch.from_numpy(d).float(), requires_grad=False)
    dataList.append({"g": d_tensor})

with open("data/evidence_train.txt", 'r') as f:
    obsList = f.read().strip().strip("#evidence").split("#evidence")


dlpmlnObj.learn(dataList = dataList, obsList = obsList, epoch=1, opt=True)

# ######################################
# # Define the function to generate dataset
# ######################################


# ######################################
# # Define Training Iterations
# ######################################

# def get_models_for_all_data(args, mvppProgram, dataset, mode = 3):
#     if mode == 1:
#         print("start to compute models for data of mode {}".format(mode))
#         with open(args.testEvi, 'r') as f:
#             list_of_obs = f.read().strip().strip("#evidence").split("#evidence")
#         test_data = dataset.test_data
#     elif mode == 2:
#         print("start to compute models for validation data")
#         with open(args.validEvi, 'r') as f:
#             list_of_obs = f.read().strip().strip("#evidence").split("#evidence")
#         test_data = dataset.valid_data
#     else:
#         print("start to compute models for training data")
#         with open(args.trainEvi, 'r') as f:
#             list_of_obs = f.read().strip().strip("#evidence").split("#evidence")
#         test_data = dataset.train_data
    
#     models = []
#     for dataIdx, data in enumerate(test_data):
#         if args.optimal:
           
#             models.append(mvppProgram.find_all_optimal_SM_under_obs(list_of_obs[dataIdx]))

#         else:
#             models.append(mvppProgram.find_all_SM_under_obs(list_of_obs[dataIdx]))
#     count_model_for_each_data = [[len(i)] for i in models]
#     print("compute models sucessfully!")
#     print(count_model_for_each_data)
#     return models


# train_loss = []

# # train 1 is sgd training.
# def train_1(args, model, device, optimizer, mvppProgram, dataset, clingomodels):
#     model.train()
#     training_red_correct = 0

#     ######
#     # 0. turn file trainEvi into list of obs
#     ######
#     with open(args.trainEvi, 'r') as f:
#         list_of_obs = f.read().strip().strip("#evidence").split("#evidence")

#     train_data = dataset.train_data
#     train_label = dataset.train_labels
#     loss_list = []
#     for dataIdx, data in enumerate(train_data):

#         # 1. for each observation, we feed the data into NN and obtain probs
#         data = Variable(torch.from_numpy(data).float(), requires_grad=False)
#         data = data.to(device)
#         output = model(data)
#         probs = output.tolist()
#         label = torch.from_numpy(train_label[dataIdx]).float().to(device)
#         if args.use_label:
#             # label = torch.from_numpy(train_label[dataIdx]).float().to(device)
#             loss = F.binary_cross_entropy(output, label)
#             # loss_list.append(loss.tolist())
#             # print("loss of nn is: ", loss)
#             loss_list.append(loss.tolist())
                
#             loss.backward(retain_graph=True)

#         if args.alpha !=0:
#             # 2. replace the parameters in mvppProgram by the output of NN
#             parameters = [[p, 1-p] for p in probs]
#             predicts = (np.array(probs) > 0.5).astype(np.float)
  
#             mvppProgram.parameters = parameters
#             # 3. compute the gradients to 24 probs w.r.t. the single observation
#             gradients = mvppProgram.gradients_one_obs_by_models(clingomodels[dataIdx])
#             # gradients = mvppProgram.gradients_one_obs(list_of_obs[dataIdx], opt=True)
#             gradients_deep = [v1 for (v1, v2) in gradients[:24]]
#             if args.debug and dataIdx % 100 == 0: print("Gradients from logic layer:\n{}".format(gradients_deep))
#             if device.type == 'cuda':
#                 gradients_deep = -args.alpha * torch.cuda.FloatTensor(gradients_deep)
#             else:
#                 gradients_deep = -args.alpha * torch.FloatTensor(gradients_deep)
#             # 4. update the parameters in NN for 1 iteration
#             output.backward(gradients_deep, retain_graph=True)
        
#         optimizer.step()
#         optimizer.zero_grad()

#         # 5. print logs
#         if dataIdx % 10 == 9:
#             if args.debug:
#                 print("Trained for {} instances of data".format(dataIdx+1))
#             elif dataIdx % 100 == 99:
#                 print("Trained for {} instances of data".format(dataIdx+1))
#     print("average loss: ",np.average(loss_list))
#     train_loss.append(np.average(loss_list))
#     # print("there are {} training data are predicted correctly".format(training_red_correct))

# def test_1(args, model, device, mvppProgram, dataset, mode=1):
#     model.eval()

#     count_all_correct = 0
#     count_constraint_correct = 0
#     count_single_edge_correct = 0
#     count_constraint_correct_opt = 0
#     pred_data = []
#     if mode == 1:
#         test_data = dataset.test_data
#         test_label = dataset.test_labels
#     elif mode == 2:
#         test_data = dataset.valid_data
#         test_label = dataset.valid_labels
#     else:
#         test_data = dataset.train_data
#         test_label = dataset.train_labels
#     for dataIdx, data in enumerate(test_data):
        
#         # 1. for each observation, we feed the data into NN and obtain probs
#         # then we check if the 24 predictions are all correct 
#         data = Variable(torch.from_numpy(data).float(), requires_grad=False)
#         data = data.to(device)
#         output = model(data)
#         probs = output.cpu().detach().numpy()
#         # if test == 1:
#         #     pred_data.append(probs)

#         predicts = (probs > 0.5).astype(np.int)
#         correct = np.array_equal(predicts, test_label[dataIdx].astype(np.int))
#         count_all_correct += int(correct)
#         count_single_edge_correct += sum((predicts == test_label[dataIdx].astype(np.int)).astype(np.int))
        

#         # 2. replace the parameters in mvppProgram by the output of NN
#         parameters = [[p, 1-p] for p in predicts]
#         mvppProgram.parameters = parameters


#         obs = ":- mistake.\n"
#         # obs = ""
#         initial_nodes = np.where(test_data[dataIdx][24:] == 1)[0]
#         for index, nodes in enumerate(initial_nodes):
#             obs += "sp(external, {}).\n".format(nodes)
#         # print(obs)
#         for nodeIdx, value in enumerate(predicts):
#             if value == 1:
#                 obs += ":- not nn_edge({}).\n".format(nodeIdx)
#             elif value == 0:
#                 obs += ":- nn_edge({}).\n".format(nodeIdx)
#             else:
#                 print("Error! value {} is not 0 or 1".format(value))
        
#         if mvppProgram.find_one_SM_under_obs(obs):
#             count_constraint_correct += 1         
#     # if test == 1:
#     #     torch.save(model.state_dict(), "saveModel_a={}/bestmodel_0822".format(int(args.alpha)))
#         # np.savetxt("result/mppweakcon_pred_dlpmln_sgd_a={}_0822.txt".format(str(args.alpha)), pred_data)
#     print("Accuracy (label): {}".format(float(count_all_correct)/len(test_data)))
#     print("Accuracy (incoherent): {}".format(float(count_single_edge_correct)/(args.output_size*len(test_data))))
#     print("Accuracy (constraint): {}".format(float(count_constraint_correct)/len(test_data)))
#     # print("Accuracy (optimal constraint): {}\n".format(float(count_constraint_correct_opt)/len(test_data)))
#     acc_mans = float(count_constraint_correct)/len(test_data)
    
#     return acc_mans

# def main_5():
#     # mvppProgram = MVPP(program_prob_grid)
#     dataset = GridProbData("data/data_5_prob.txt")
#     device = torch.device( "cpu")
#     model = FC(40,*[50,50,50,50,50],24).to(device)
#     model.load_state_dict(torch.load("saveModel_a=0/bestmodel_0816"))
#     get_bestmodel_preds(model, device, dataset, 1)

# def get_bestmodel_preds(model,device, dataset, test=1):
#     model.eval()

#     ######
#     # 0. turn file testEvi into list of obs
#     ######
#     pred_data = []
#     if test == 1:
#         test_data = dataset.test_data
#         test_label = dataset.test_labels
#     elif test == 2:
#         test_data = dataset.valid_data
#         test_label = dataset.valid_labels
#     else:
#         test_data = dataset.train_data
#         test_label = dataset.train_labels
    
#     data = Variable(torch.from_numpy(test_data).float(), requires_grad=False)
#     data = data.to(device)
#     output = model(data)
#     probs = output.cpu().detach().numpy()
#     # np.savetxt("result/mppweakcon_pred_nnonly_0816_test.txt", probs)
#     # print("done!")
#     return probs

# def constraint_acc(probs, obs, mvppProgram):
#     predicts = (probs > 0.5).astype(np.int)
#     # obs = list_of_obs[dataIdx]
#     for nodeIdx, value in enumerate(predicts):
#         if value == 1:
#             obs += ":- not nn_edge({}).\n".format(nodeIdx)
#         elif value == 0:
#             obs += ":- nn_edge({}).\n".format(nodeIdx)
#         else:
#             print("Error! value {} is not 0 or 1".format(value))
#     if mvppProgram.find_one_SM_under_obs(obs):
#         return True
#     else:
#         return False

# def test_ultimate_acc(mvppProgram, pred_file, mode, dataset):
#     # predicts = ['0.9697', '0.8390', '0.5303', '0.5055', '0.1739', '0.2276', '0.7696', '0.7308', '0.4883', '0.2136', '0.2121', '0.6630', '0.5389', '0.8977', '0.5393', '0.3026', '0.0260', '0.7347', '0.7524', '0.4735', '0.3397', '0.8893', '0.1301', '0.5748']
#     if mode == 1:
#         file = "data/evidence_constraints_test.txt"
#         list_of_para = dataset.test_data
#     elif mode == 2:
#         file = "data/evidence_constraints_valid.txt"
#         list_of_para = dataset.valid_data
#     else:
#         file = "data/evidence_constraints_train.txt"
#         list_of_para = dataset.train_data
#     with open(file, 'r') as f:
#         list_of_obs = f.read().strip().strip("#evidence").split("#evidence")
    
#     list_of_pred = np.loadtxt(pred_file)
#     count = 0
#     # constraint_butnot_optimal = []

#     for dataIdx, each_pred in enumerate(list_of_pred):

#         predicts = np.where(each_pred>0.5)[0].tolist()
        
#         sm_max = mvppProgram.find_all_optimal_SM_under_obs(list_of_obs[dataIdx])
#         list_of_models = []
#         for model in sm_max:
#             single_model = []
#             for atom in model:
#                 if atom.startswith("nn_edge"):
#                     single_model.append(int(atom[8:-1]))
#             list_of_models.append(single_model)
#         if predicts in list_of_models:
#             count+=1
#         # else:
            
#         #     constraint_butnot_optimal.append(dataIdx)
                
#     print("ultimate_acc:", count/len(list_of_pred))
#     # print(constraint_butnot_optimal)
#     # print(len(constraint_butnot_optimal))
#     return count/len(list_of_pred)

# def test_nearly_acc( mvppProgram, pred_file, mode, dataset):
#     # predicts = ['0.9697', '0.8390', '0.5303', '0.5055', '0.1739', '0.2276', '0.7696', '0.7308', '0.4883', '0.2136', '0.2121', '0.6630', '0.5389', '0.8977', '0.5393', '0.3026', '0.0260', '0.7347', '0.7524', '0.4735', '0.3397', '0.8893', '0.1301', '0.5748']
#     if mode == 1:
#         file = "data/evidence_constraints_test.txt"
#         list_of_para = dataset.test_data
#         labels = dataset.test_labels
#     elif mode == 2:
#         file = "data/evidence_constraints_valid.txt"
#         list_of_para = dataset.valid_data
#         labels = dataset.valid_labels
#     else:
#         file = "data/evidence_constraints_train.txt"
#         list_of_para = dataset.train_data
#         labels = dataset.train_labels
#     with open(file, 'r') as f:
#         list_of_obs = f.read().strip().strip("#evidence").split("#evidence")
    
#     list_of_pred = np.loadtxt(pred_file)
#     count = 0
#     # constraint_butnot_optimal = []
#     count = 0
#     for dataIdx, each_pred in enumerate(list_of_pred):

#         label = np.where(labels[dataIdx]==1)[0].tolist()

#         predicts = np.where(each_pred>0.5)[0].tolist()
        
#         label_prob = 1
#         pred_prob = 1
        
#         for i in label:
#             label_prob *= list_of_para[dataIdx][i]
#             # print(label_prob)
#         for i in predicts:
#             pred_prob *= list_of_para[dataIdx][i] 
#             # print(pred_prob)

#         if constraint_acc(each_pred, list_of_obs[dataIdx], mvppProgram ) :
#             if pred_prob >= label_prob*0.63:
#                 count += 1
#             else:
#                 print( pred_prob / label_prob)

#     print("nearly_acc:", count/len(list_of_pred))



# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch Path Example')

#     parser.add_argument('--epochs', type=int, default=500, metavar='N',
#                         help='number of epochs to train (default: 500)')
#     parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
#                         help='learning rate (default: 0.001)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')

#     parser.add_argument('--input_size', type=int, default=40, metavar='InputSize',
#                         help='the size of input layer')
#     parser.add_argument('--hidden_size', type=int, default=50, metavar='HiddenSize',
#                         help='the size of hidden layer')
#     parser.add_argument('--output_size', type=int, default=24, metavar='OutputSize',
#                         help='the size of output layer')
#     parser.add_argument('--num_hidden_layers', type=int, default=5, metavar='#HiddenLayers',
#                         help='the number of hidden layers')

#     parser.add_argument('--trainEvi', type=str, default="data/evidence_train.txt", metavar='Ftrain',
#                         help='the file containing training evidences')
#     parser.add_argument('--testEvi', type=str, default="data/evidence_test.txt", metavar='Ftest',
#                         help='the file containing testing evidences') 
#     parser.add_argument('--validEvi', type=str, default="data/evidence_valid.txt", metavar='Fvalid',
#                         help='the file containing validation evidences')

#     parser.add_argument('--use_label', action='store_true', default=True,
#                         help='use label for training as in Semantic Loss paper')
#     parser.add_argument('--alpha', type=float, default=0, metavar='A',
#                         help='percentage of the gradients (learned in logic part) that is used for training (default: 0.1)')
#     parser.add_argument('--debug', action='store_true', default=False,
#                         help='print more information for debugging')

#     parser.add_argument('--fromEpoch', type=int, default=1, metavar='FE',
#                         help='if FE is 1, we train from the beginning; otherwise, we load the model trained at epoch FE-1 and start to train for epoch FE')

#     parser.add_argument('--optimal', action='store_true', default=False,
#                         help='train with optimal constraints')
#     parser.add_argument('--visualize_convergency', action='store_true', default=False,
#                         help='train with optimal statement')
#     parser.add_argument('--modelfile', type='str', default="saveModel_dlpmln",
#                         help='generate the dataset for training and testing')
    
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#     torch.manual_seed(args.seed)
#     device = torch.device("cuda" if use_cuda else "cpu")

#     ########
#     # Set dataset for training and testing
#     ########
#     dataset = GridProbData("data/data.txt")

#     ########
#     # Construct model and set optimizer
#     ########
#     hidden_sizes = [args.hidden_size for i in range(args.num_hidden_layers)]
#     model = FC(args.input_size,*hidden_sizes,args.output_size).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     ########
#     # Initialize MVPP program
#     ########
    
#     mvppProgram_train = MVPP(program_prob_grid)
#     mvppProgram_test = MVPP(program_prob_grid)
   
#     clingo_models_train = []


#     ########
#     # Preload the parameters for model
#     ########
#     fromEpoch = int(args.fromEpoch)
#     if fromEpoch > 1:
#         print("load model from previous model")
#         model.load_state_dict(args.modelfile+"epoch{}".format(fromEpoch-1))

#     ########
#     # Start Training
#     ########
#     valid_acc = 0
#     visualize_convergency = []
#     if not path.exist(args.modelfile):
#         os.mkdir(args.modelfile)
    
#     for epoch in range(fromEpoch, args.epochs + fromEpoch):
#         # if epoch % 1000 == 1:
#         print("Training for epoch {}...".format(epoch))
#         train_1(args, model, device, optimizer, mvppProgram_train, dataset, clingo_models_train)
        
#         print("Testing for training ")
#         test_1(args, model, device, mvppProgram_test, dataset,3)
#         print("Testing for valid")
#         acc_mans = test_1(args, model, device, mvppProgram_test, dataset,2)
#         visualize_convergency.append(acc_mans)
#         if acc_mans > valid_acc:
#             valid_acc = acc_mans
#             print("best epoch is {}".format(epoch))
#         print("Testing for testing")
#         test_1(args, model, device, mvppProgram_test, dataset_test,1)

#         torch.save(model.state_dict(), args.modelfile+"/epoch{}".format( epoch))


# if __name__ == '__main__':
    
#     main()
#     