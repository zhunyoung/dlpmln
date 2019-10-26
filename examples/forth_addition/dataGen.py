import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
np.random.seed=0

def create_data_sample():
    add1=[np.random.randint(0,10),np.random.randint(0,10)]
    add2=[np.random.randint(0,10),np.random.randint(0,10)]
    
    carry=np.random.randint(0,2)
        
    num1=int(str(add1[0])+str(add1[1]))
    num2=int(str(add2[0])+str(add2[1]))
    num_sum=num1+num2+carry
    num_sum_str=str(num_sum)
    
    if num_sum_str.__len__()==3:
        numstr1=num_sum_str[0]
        numstr2=num_sum_str[1]
        numstr3=num_sum_str[2]
    elif num_sum_str.__len__()==2:
        numstr1=0
        numstr2=num_sum_str[0]
        numstr3=num_sum_str[1]
    elif num_sum_str.__len__()==1:
        numstr1=0
        numstr2=0
        numstr3=num_sum_str[0]
    
    rule= 'add([{0},{1}],[{2},{3}],{4},[{5},{6},{7}]).'.format(add1[0],add1[1],add2[0],add2[1],carry,numstr1,numstr2,numstr3)    
    return rule,[add1[0],add1[1],add2[0],add2[1],carry,numstr1,numstr2,numstr3]

def format_dataList(obs,str_list):
    n_digits=10
    size=3
    y=torch.zeros(size,n_digits)
    x=torch.LongTensor(size,1).random_()%n_digits
    x[0,0]=5
    y.scatter_(1,x,1)
    
    DATA1_idx=torch.LongTensor([[str_list[1]],[str_list[3]],[0]])
    DATA2_idx=torch.LongTensor([[str_list[1]],[str_list[3]],[1]])
    DATA3_idx=torch.LongTensor([[str_list[0]],[str_list[2]],[0]])
    DATA4_idx=torch.LongTensor([[str_list[0]],[str_list[2]],[1]])
    
    DATA1=torch.zeros(size,n_digits)
    DATA2=torch.zeros(size,n_digits)
    DATA3=torch.zeros(size,n_digits)
    DATA4=torch.zeros(size,n_digits)

    DATA1=DATA1.scatter_(1,DATA1_idx,1).view(1,30)
    DATA2=DATA2.scatter_(1,DATA2_idx,1).view(1,30)
    DATA3=DATA3.scatter_(1,DATA3_idx,1).view(1,30)
    DATA4=DATA4.scatter_(1,DATA4_idx,1).view(1,30)
    
    key1='{0},{1},{2}'.format('num1_0','num2_0',0)
    key2='{0},{1},{2}'.format('num1_0','num2_0',1)
    key3='{0},{1},{2}'.format('num1_1','num2_1',0)
    key4='{0},{1},{2}'.format('num1_1','num2_1',1)
    
    dataList_dict={key1:DATA1,key2:DATA2,key3:DATA3,key4:DATA4}
    return dataList_dict


def format_observations(obs,str_list):
    obs_string= '''
    :-not carry(0,{}).
    :-not carry(2,{}).
    :-not result(1,{}).
    :-not result(0,{}).
    '''.format(str_list[4], str_list[5],str_list[6],str_list[7])
    return obs_string

class add_test(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,size):
        self.size=size
        self.data=[]
        self.create_data()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
    
    def create_data(self):
        for i in range(self.size):
            obs,str_list=create_data_sample()
            data_list_sample=format_dataList(obs,str_list)
            label_value=str_list[1]+str_list[3] +str_list[4]
            label_value=int(str(label_value)[-1])
            keys=[]
            for key in data_list_sample.keys():
                keys.append(key)
            if len(keys)<4:
                if str_list[4]==0:
                    key = keys[0]
                else:
                    key = keys[1]
            else:
                if str_list[4]==0:
                    key = keys[0]
                else:
                    key = keys[1]
            x = data_list_sample[key]
            y_value=int(str_list[-1])
            output_size=10
            y=torch.zeros(1,output_size)
            d=torch.LongTensor(1,1).random_()%output_size
            d[0,0]=label_value
            y.scatter_(1,d,1)
            y=y.squeeze()
            y=y.argmax(dim=0,keepdim=True)
            self.data.append([x.squeeze(),y])

        return None
        
class carry_test(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,size):
        self.size=size
        self.data=[]
        self.create_data()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
    
    def create_data(self):
        for i in range(self.size):
            obs,str_list=create_data_sample()
            data_list_sample=format_dataList(obs,str_list)
            sum_value=str_list[1]+str_list[3] +str_list[4]
            if len(str(sum_value))==1:
                label_value=0
            else:
                label_value=1
            keys=[]
            for key in data_list_sample.keys():
                keys.append(key)
            if len(keys)<4:
                if str_list[4]==0:
                    key = keys[0]
                else:
                    key = keys[1]
            else:
                if str_list[4]==0:
                    key = keys[0]
                else:
                    key = keys[1]
            x = data_list_sample[key]
            y_value=int(str_list[-1])
            output_size=2
            y=torch.zeros(1,output_size)
            d=torch.LongTensor(1,1).random_()%output_size
            d[0,0]=label_value
            y.scatter_(1,d,1)
            y=y.squeeze()
            y=y.argmax(dim=0,keepdim=True)
            self.data.append([x.squeeze(),y])
        return None

dataList = []
obsList = []
train_size=1000
test_size=100

add_test_dataset=add_test(test_size)
add_test_dataloader=DataLoader(add_test_dataset,batch_size=4,shuffle=True)
carry_test_dataset=carry_test(test_size)
carry_test_dataloader=DataLoader(carry_test_dataset,batch_size=4,shuffle=True)

for i in range(train_size):
    obs,str_list=create_data_sample()
    dataList.append(format_dataList(obs,str_list))
    obsList.append(format_observations(obs,str_list))