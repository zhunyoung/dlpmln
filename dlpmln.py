import re
import sys

import clingo

class deeplpmln(object):
    def __init__(self, dprogram, functions):
        '''
        @param dprogram: a string for a DeepLPMLN program
        @param functions: a list of neural networks
        '''
        self.dprogram = dprogram
        self.functions = dict()
        for function in functions:
            self.functions[function.__name__] = function
        self.mvpp = self.parse()

    @staticmethod
    def nnAtom2MVPPrules(nnAtom):
        mvppRules = []

        # STEP 1: obtain all information
        regex = '^nn\((.+\)),(.+),(\(.+\))[)]$'
        out = re.search(regex, nnAtom)
        model, vin = out.group(1).split('(')
        vin, e = vin.replace(')','').split(',')
        e = int(e)
        pred = out.group(2)
        domain = out.group(3).replace('(', '').replace(')','').split(',')
        k = len(domain)

        # STEP 2: generate MVPP rules
        # we have different translations when k = 2 or when k > 2
        if k == 2:
            pass
        elif k > 2:
            patoms = []
            for i in range(e):
                rule = ''
                for j in range(k):
                    rule += '@{}({},{})[{}] {}({}, {}, {}); '.format(model, vin, i, j, pred, vin, i, domain[j])
                mvppRules.append(rule[:-2]+'.')
        else:
            print('Error: the number of element in the domain %s is less than 2' % domain)
        return mvppRules


    def parse(self):
        # 1. Generate grounded nn atoms
        clingo_control = clingo.Control(["--warn=none"])
        clingo_control.add("base", [], self.dprogram.replace('[', '(').replace(']', ')'))
        clingo_control.ground([("base", [])])
        symbols = [atom.symbol for atom in clingo_control.symbolic_atoms]
        mvppRules = [self.nnAtom2MVPPrules(str(atom)) for atom in symbols if atom.name == 'nn']
        mvppRules = [rule for rules in mvppRules for rule in rules]
        # mvppRules = [self.nnAtom2MVPPrules(atom) for atom in nnAtoms]

        lines = [line.strip() for line in self.dprogram.split('\n') if line and not line.startswith('nn')]
        # return mvppRules + lines
        return '\n'.join(mvppRules + lines)
        # lines = [self.neuralRuleToMVPPRule(rule) for rule in lines]

        # return mvppRules, lines
        # preds = []
        # program = self.dprogram
        # program_wo_input_rules = ""
        # lpmln = ""
        # lines = program.readlines()
        # for line in lines:
        #     if "@input" in line:
        #         pred ,arity = line.strip()[:-1].split(" ")[1].split("/")
        #         preds.append((pred, int(arity)))
        #     else:
        #         program_wo_input_rules += line.strip() + "\n"
        #         if not line.startswith("nn"):
        #             lpmln += line.strip() + "\n"
        # return lpmln, program_wo_input_rules, preds
    
    def learn(self):
        pass

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