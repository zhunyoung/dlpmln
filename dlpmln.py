import re
import sys
from klpmln import MVPP
import clingo
import sys
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter



class DeepLPMLN(object):
    def __init__(self, dprogram, functions, optimizers):

        """
        @param dprogram: a string for a DeepLPMLN program
        @param functions: a list of neural networks
        """
        self.dprogram = dprogram
        self.k = {} # k would be 1 or N (>=3); note that k=2 in theorey is implemented as k=1
        self.e = {}
        self.nnOutputs = {}
        self.nnGradients = {}
        self.functions = functions
        self.optimizers = optimizers
        # self.mvpp is a dictionary consisting of 3 mappings: 
        # 1. 'program': a string denoting an MVPP program where the probabilistic rules generated from NN are followed by other rules;
        # 2. 'nnProb': a list of lists of tuples, each tuple is of the form (model, term, i, j)
        # 3. 'atom': a list of list of atoms, where each list of atoms is corresponding to a prob. rule
        # 4. 'nnPrRuleNum': an integer denoting the number of probabilistic rules generated from NN
        self.mvpp = {'nnProb': [], 'atom': [], 'nnPrRuleNum': 0, 'program': ''}
        self.mvpp['program'] = self.parse()

    def nnAtom2MVPPrules(self, nnAtom):
        """
        @param nnAtom: a string of a neural atom
        @param countIdx: a Boolean value denoting whether we count the index for the value of m(vin, i)[j]
        """
        mvppRules = []

        # STEP 1: obtain all information
        regex = '^nn\((.+\)),(.+),(\(.+\))[)]$'
        out = re.search(regex, nnAtom)
        model, vin = out.group(1).split('(')
        vin, e = vin.replace(')','').rsplit(',', 1)
        e = int(e)
        pred = out.group(2)
        domain = out.group(3).replace('(', '').replace(')','').split(',')
        k = len(domain)
        if k == 2:
            k = 1
        self.k[model] = k
        self.e[model] = e
        if model not in self.nnOutputs:
            self.nnOutputs[model] = {}
            self.nnGradients[model] = {}
        if vin not in self.nnOutputs[model]:
            self.nnOutputs[model][vin] = None
            self.nnGradients[model][vin] = None

        # STEP 2: generate MVPP rules
        # we have different translations when k = 2 or when k > 2
        if k == 1:
            for i in range(e):
                rule = '@0.0 {}({}, {}, {}); @0.0 {}({}, {}, {}).'.format(pred, vin, i, domain[0], pred, vin, i, domain[1])
                prob = [tuple((model, vin, i, 0))]
                atoms = ['{}({}, {}, {})'.format(pred, vin, i, domain[0]), '{}({}, {}, {})'.format(pred, vin, i, domain[1])]
                mvppRules.append(rule)
                self.mvpp['nnProb'].append(prob)
                self.mvpp['atom'].append(atoms)
                self.mvpp['nnPrRuleNum'] += 1

        elif k > 2:
            for i in range(e):
                rule = ''
                prob = []
                atoms = []
                for j in range(k):
                    atom = '{}({}, {}, {})'.format(pred, vin, i, domain[j])
                    rule += '@0.0 {}({}, {}, {}); '.format(pred, vin, i, domain[j])
                    prob.append(tuple((model, vin, i, j)))
                    atoms.append(atom)
                mvppRules.append(rule[:-2]+'.')
                self.mvpp['nnProb'].append(prob)
                self.mvpp['atom'].append(atoms)
                self.mvpp['nnPrRuleNum'] += 1
        else:
            print('Error: the number of element in the domain %s is less than 2' % domain)
        return mvppRules


    def parse(self):
        # 1. Generate grounded nn atoms
        clingo_control = clingo.Control(["--warn=none"])
        # remove weak constraints
        program = re.sub(r'\n:~ .+\.[ \t]*\[.+\]', '\n', self.dprogram)
        clingo_control.add("base", [], program.replace('[', '(').replace(']', ')'))
        clingo_control.ground([("base", [])])
        symbols = [atom.symbol for atom in clingo_control.symbolic_atoms]
        mvppRules = [self.nnAtom2MVPPrules(str(atom)) for atom in symbols if atom.name == 'nn']
        mvppRules = [rule for rules in mvppRules for rule in rules]

        # 2. Combine neural rules with the other rules
        lines = [line.strip() for line in self.dprogram.split('\n') if line and not line.startswith('nn(')]
        return '\n'.join(mvppRules + lines)
        
    def infer(self, dataDic, obs, mvpp=''):
        """
        @param dataDic: a dictionary that maps terms to tensors/np-arrays
        @param obs: a list of strings, where each string is a set of constraints denoting an observation
        @param mvpp: an MVPP program used in inference
        """

        # Step 1: get the output of each neural network
        for model in self.nnOutputs:
            for vin in self.nnOutputs[model]:
                self.nnOutputs[model][vin] = self.functions[model](dataDic[vin]).view(-1).tolist()
        print(self.nnOutputs)

        # Step 2: turn the NN outputs into a set of MVPP probabilistic rules
        mvppRules = ''
        for ruleIdx in range(self.mvpp['nnPrRuleNum']):
            probs = [self.nnOutputs[m][t][i*self.k[model]+j] for (m,t,i,j) in self.mvpp['nnProb'][ruleIdx]]
            if len(probs) == 1:
                mvppRules += '@{} {}; @{} {}.\n'.format(probs[0], self.mvpp['atom'][ruleIdx][0], 1 - probs[0], self.mvpp['atom'][ruleIdx][1])
            else:
                tmp = ''
                for atomIdx, prob in enumerate(probs):
                    tmp += '@{} {}; '.format(prob, self.mvpp['atom'][ruleIdx][atomIdx])
                mvppRules += tmp[:-2] + '.\n'

        # Step 3: find an optimal SM under obs
        dmvpp = MVPP(mvppRules + mvpp)
        return dmvpp.find_all_opt_SM_under_obs(obs=obs)


    def learn(self, dataList, obsList, epoch, opt=False):
        """
        @param dataList: a list of dictionaries, where each dictionary maps terms to tensors/np-arrays
        @param obsList: a list of strings, where each string is a set of constraints denoting an observation
        @param epoch: an integer denoting the number of epochs
        """
        assert len(dataList) == len(obsList), 'Error: the length of dataList does not equal to the length of obsList'

        # get the mvpp program by self.mvpp, so far self.mvpp is a string
        dmvpp = MVPP(self.mvpp['program'])

        # we train all nerual networks
        for func in self.functions:
            self.functions[func].train()

        # we train for epoch times of epochs
        for epochIdx in range(epoch):
            print('Training for epoch %d ...' % (epochIdx + 1))
            # for each training instance in the training data
            for dataIdx, data in enumerate(dataList):
                nnOutput = {}
                # Step 1: get the output of each neural network and initialize the gradients
                for model in self.nnOutputs:
                    nnOutput[model] = {}
                    for vin in self.nnOutputs[model]:
                        nnOutput[model][vin] = self.functions[model](data[vin])
                        self.nnOutputs[model][vin] = nnOutput[model][vin].view(-1).tolist()
                        # initialize the gradients for each output
                        self.nnGradients[model][vin] = [0.0 for i in self.nnOutputs[model][vin]]
                # print(self.nnOutputs)
                # print(self.nnGradients)

                # print(self.mvpp['nnProb'])
                # Step 2: replace the parameters in the MVPP program with nn outputs
                for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                    for (m,t,i,j) in self.mvpp['nnProb'][ruleIdx]:
                        if self.k[m] > 2:
                            dmvpp.parameters[ruleIdx] = [self.nnOutputs[m][t][i*self.k[model]+j]] 
                        else:
                            dmvpp.parameters[ruleIdx] = [self.nnOutputs[m][t][i*self.k[model]+j], 1-self.nnOutputs[m][t][i*self.k[model]+j]] 

                # Step 3: compute the gradients
                dmvpp.normalize_probs()
                gradients = dmvpp.gradients_one_obs(obsList[dataIdx], opt)

                # Step 4: update parameters in neural networks
                gradientsNN = gradients[:self.mvpp['nnPrRuleNum']].tolist()
                for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                    for probIdx, (m,t,i,j) in enumerate(self.mvpp['nnProb'][ruleIdx]):
                        # self.nnGradients[m][t][i*self.k[model]+j] = gradients
                        self.nnGradients[m][t][i*self.k[model]+j] = - gradientsNN[ruleIdx][probIdx]
                # backpropogate
                
                for m in nnOutput:
                    for t in nnOutput[m]:
                        # print(nnOutput[m][t])
                        # print(self.nnGradients[m][t])
                        # sys.exit()
                        nnOutput[m][t].backward(torch.FloatTensor(np.reshape(np.array(self.nnGradients[m][t]),nnOutput[m][t].shape)), retain_graph=True)
                for opt in self.optimizers:
                    self.optimizers[opt].step()
                    self.optimizers[opt].zero_grad()

                # Step 5: update probabilities in normal prob. rules
                # if dataIdx % 100 == 99:
                #     for name, value in self.functions['m'].named_parameters():
                #         print(name, value)
                #         break
                pass
            # loger = SummaryWriter("./param.logs")
            # for name, value in self.functions['m'].named_parameters():
            #     loger.add_histogram(name, value, epoch)

                # print(name, value)


    def testNN(self, nn, testLoader):
        """
        @nn is the name of the neural network to check the accuracy. 
        @testLoader is the input and output pairs.
        """
        self.functions[nn].eval()
        # test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testLoader:
                # data, target = data.to(device), target.to(device)
                output = self.functions[nn](data)
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                if self.k[nn] >2 :
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                else: 
                    pass
        print("Test Accuracy {:.0f}%".format(100. * correct / len(testLoader.dataset)) )
    
    def testConstraint(self, dataList, obsList, mvppList, opt=False):
        """
        @param dataList: a list of dictionaries, where each dictionary maps terms to tensors/np-arrays
        @param obsList: a list of strings, where each string is a set of constraints denoting an observation
        @param mvppList: a list of MVPP programs (each is a string)
        """
        assert len(dataList) == len(obsList), 'Error: the length of dataList does not equal to the length of obsList'

        # we evaluate all nerual networks
        for func in self.functions:
            self.functions[func].eval()

        # we test for each DeepLPMLN program
        if opt==False:
            for programIdx, program in enumerate(mvppList):
                mvpp = MVPP(program)
                count = 0
                for dataIdx, data in enumerate(dataList):
                    nnOutput = {}
                    # Step 1: get the output of each neural network
                    for model in self.nnOutputs:
                        nnOutput[model] = {}
                        for vin in self.nnOutputs[model]:
                            nnOutput[model][vin] = self.functions[model](data[vin])
                            self.nnOutputs[model][vin] = nnOutput[model][vin].view(-1).tolist()
                    # print(self.nnOutputs)

                    # Step 2: turn the NN outputs into a set of ASP facts
                    aspFacts = ''
                    for ruleIdx in range(self.mvpp['nnPrRuleNum']):
                        probs = [self.nnOutputs[m][t][i*self.k[model]+j] for (m,t,i,j) in self.mvpp['nnProb'][ruleIdx]]
                        if len(probs) == 1:
                            atomIdx = int(probs[0] > 0.5)
                        else:
                            atomIdx = probs.index(max(probs))
                        aspFacts += self.mvpp['atom'][ruleIdx][atomIdx] + '.\n'
                    # print(aspFacts)

                    # Step 3: check if the mvpp program is satisfiable with the facts generated from NN outputs
                    mvpp.pi_prime += aspFacts
                    if mvpp.find_one_SM_under_obs(obs=obsList[dataIdx]):
                        count += 1
                print('The accuracy for the {}th program is {}'.format(programIdx+1, float(count)/len(dataList)))
        else:
            nn_atom = "nn_edge"
            nn = "m"
            term = "g"
            dmvpp = MVPP(self.mvpp['program'])
            models = []
            count = 0
            for dataIdx, data in enumerate(dataList):
                nnOutput = {}
                # Step 1: get the output of each neural network
                for model in self.nnOutputs:
                    nnOutput[model] = {}
                    for vin in self.nnOutputs[model]:
                        nnOutput[model][vin] = self.functions[model](data[vin])
                        self.nnOutputs[model][vin] = nnOutput[model][vin].view(-1).tolist() 
                    sm_max = dmvpp.find_all_opt_SM_under_obs_WC(obsList[dataIdx])
                    list_of_models = []
                    for model in sm_max:
                        single_model = []
                        for atom in model:
                            if atom.startswith(nn_atom) and atom[-2] == "t":
                                single_model.append(int(atom[len(nn_atom)+1:-1].split(",")[1]))
                        list_of_models.append(single_model)
                    # print(self.nnOutputs[nn][term])
                    nn_prediction = np.where(np.array(self.nnOutputs[nn][term])>0.5)[0].tolist()
                    # print(nn_prediction)
                    # print(list_of_models)
                    if nn_prediction in list_of_models:
                        count+=1
                    
            print("The optimal accuracy is {}".format(count/len(dataList)))
        

    def dataloader_error_checker(self, pred2data, preds):
        # we check for each experiment
        for pred,arity in preds:
            if pred not in pred2data:
                print("Error: the data for predicate '{}'' cannot be found in Dataloader!".format(pred))
                return False
            else:
                # the arity of pred may be more than 1, thus each data is in the form of a list, whose arity is the same as the arity of pred
                for data_list in pred2data[pred]:
                    if len(data_list) != arity:
                        print("Error: the arity of predicate '{}' is {} but it does not match with data:\n{}".format(pred, arity, data_list))
                        return False
        return True

    def run(self, program, dataloader, mode):
        import lpmln_parser
        import clingo
        import subprocess
        import math

        # 1. Find img (with 1 input) in program
        lpmln, program_wo_input_rules, preds  = self.parse_dlpmln(program)
        # print("preds: {}".format(preds))

        # 2. Call dataloader, each element in dataloader is a mapping from predicate name to data
        # We invoke one LPMLN inference for each element in dataloader
        for batch_i, pred2data in enumerate(dataloader):
            new_lpmln = lpmln
            print("Experiment {}:".format(batch_i))
            if not self.dataloader_error_checker(pred2data, preds):
                continue

            # for each experiment, we need a new mapping from constant to data
            constant2data = {}
            inputRules = []
            for pred,arity in preds:
                for inputIdx, data_list in enumerate(pred2data[pred]):
                    constants = ""
                    for dataIdx, data in enumerate(data_list):
                        # constantName = pred_inputIdx_dataIdx
                        constantName = "{}_{}_{}".format(pred, inputIdx, dataIdx)
                        constant2data[constantName] = data
                        constants += ", {}".format(constantName)
                    inputRules.append("{}({}).\n".format(pred, constants[2:]))

            # 3. We ground the deepLPMLN program and obtain all nn(*) atoms
            # we construct the lpmln program with the evidences for the input data
            for rule in inputRules:
                program_wo_input_rules += rule
            # we use lpmln2asp to turn lpmln to asp, and then use gringo to ground it
            lpmlnParser = lpmln_parser.lpmln_parser()
            # print(lpmln_parser.lpmln_to_asp_parser(lpmln))

            clingo_control = clingo.Control()
            clingo_control.add("base", [], lpmlnParser.lpmln_to_asp_parser(program_wo_input_rules))
            clingo_control.ground([("base", [])])
            symbols = [atom.symbol for atom in clingo_control.symbolic_atoms]
            for atom in symbols:
                if atom.name == "nn":
                    arg1_of_nn = atom.arguments[0]
                    nn_model_name = str(atom.arguments[1])
                    predicate_name = str(arg1_of_nn.name)
                    constants_input2nn = str(arg1_of_nn.arguments[:-1]).replace('[', '').replace(']','')
                    
                    # 4. We call the yolo function to get outputs
                    # 4.1 we separate constants of the form "a, b, c" into [a, b, c]
                    input_nn = constants_input2nn.split(", ")
                    # 4.2 we replace each constant if it is in the constant2data mapping
                    for i, constant in enumerate(input_nn):
                        if constant in constant2data:
                            input_nn[i] = constant2data[constant]
                    # 4.3 we feed the constants to neural network model
                    list_of_tuples = self.functions[nn_model_name](input_nn)
                    # 4.4 we add soft rules to lpmln program according to NN outputs
                    for one_tuple in list_of_tuples:
                        tmp = ', '.join(map(str,one_tuple[:-1]))
                        p = one_tuple[-1]
                        if p == 1:
                            w = ""
                        elif p == 0:
                            w = ":- "
                        else:
                            w = str(math.log(p/(1-p)))+" "
                        rule = "{}{}({}, {}).\n".format(w, predicate_name, constants_input2nn, tmp)
                        new_lpmln += rule
            lpmln_filename = "lpmln{}.tmp".format(batch_i)
            print("\n\nLPMLN program stored in {}:\n".format(lpmln_filename)+new_lpmln)

            # Clear the lpmln file
            open(lpmln_filename, "w").close()
            # Write the lpmln file
            with open(lpmln_filename, "w") as f:
                f.write(new_lpmln)

            
            if mode == "MAP":
                print("LPMLN command line:")
                print("lpmln_infer", lpmln_filename, "\n")
                print("LPMLN outputs:")
                subprocess.run(["lpmln_infer", lpmln_filename])
            elif mode.startswith("EX"):
                print("LPMLN command line:")
                print("lpmln_infer", lpmln_filename, "{}".format(mode[2:].strip()), "-ex\n")
                print("LPMLN outputs:")
                subprocess.run(["lpmln_infer", lpmln_filename, "{}".format(mode[2:].strip()), "-ex"])
            elif mode.startswith("APPROX"):
                print("LPMLN command line:")
                print("lpmln_infer", lpmln_filename, "{}".format(mode[2:].strip()), "-approx\n")
                print("LPMLN outputs:")
                subprocess.run(["lpmln_infer", lpmln_filename, "{}".format(mode[2:].strip()), "-approx"])
            print("===================================================\n")
    
    def dataloader4singleFiles(self, pred_path_n):
        import glob
        from PIL import Image
        import numpy as np

        data = []

        pred_path_n = pred_path_n.strip().split('\n')
        numOfPred = len(pred_path_n)
        pred = [None] * numOfPred
        path = [None] * numOfPred
        n = [None] * numOfPred
        files = [None] * numOfPred
        for i in range(numOfPred):
            pred[i], path[i], n[i] = pred_path_n[i].split(' ')
            n[i] = int(n[i])
            # sorted(glob.glob('*.png'))
            files[i] = sorted(glob.glob(path[i]))

        # enumerate all possible tasks
        numOfTasks = min(int(len(files[i])/n[i]) for i in range(numOfPred))
        for i in range(numOfTasks):
            mapping = {}
            # we consider the i-th task
            for j in range(numOfPred):
                # we consider the j-th predicate
                mapping[pred[j]] = []
                for k in range(n[j]):
                    path = files[j][i*n[j]+k]
                    try:
                        im=np.array(Image.open(path))
                        mapping[pred[j]].append([im])
                    except IOError:
                        print("File {} is not an image".format(path))
            data.append(mapping)

        return data
