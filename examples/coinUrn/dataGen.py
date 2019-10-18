import torch
import torchvision
from torchvision import transforms

# Define the function to process the data
def fileToLists(path, dataLoader):
    """ Return dataList and obsList 

    @param path: a string denoting the path to the dataset file in the text form
    @param dataLoader: a Pytorch dataset object, where the data format is
    [
      [data, label]
      ...
      [data, label]
    ]
    """
    dataList = []
    obsList = []
    m1dataset = []
    m2dataset = []
    with open(path, 'r') as f:
        lines = f.readlines()
    # anaylize each line, one corresponds to an instance of training/testing data
    for dataIdx, line in enumerate(lines):
        _, imgIndex, color1, _, color2, result, _ = line.replace("(","$").replace(")","$").replace("[","$").replace("]","$").split("$")
        imgIndex = int(imgIndex.strip()[:-1])
        color1 = [float(i) for i in color1.split(",")]
        color2 = [float(i) for i in color2.split(",")]
        label1 = color1.index(max(color1))
        label2 = color2.index(max(color2))
        color1 = torch.FloatTensor(color1)
        color2 = torch.FloatTensor(color2)
        result = result.strip().replace(",", "")
        dataList.append({'i': dataLoader[imgIndex][0].view(1, 1, 28, 28), 'color1': color1.unsqueeze(0), 'color2': color2.unsqueeze(0)})
        obsList.append(':- mistake.\n:- not {}.'.format(result))
        m1dataset.append([dataLoader[imgIndex][0].view(1, 1, 28, 28), torch.tensor(dataLoader[imgIndex][1])])
        m2dataset.append([color1.unsqueeze(0), torch.tensor(label1)])
        m2dataset.append([color2.unsqueeze(0), torch.tensor(label2)])
    return dataList, obsList, m1dataset, m2dataset


"""
Load the MNIST dataset
The data format of trainLoader is
[
  [data, label]
  ...
  [data, label]
]
where data is a tensor for a digit image and label is an integer denoting its label

"""
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
trainLoader = torchvision.datasets.MNIST(root='../../data/', train=True, download=True,transform=transform)
testLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../../data/', train=False, transform=transform), batch_size=1000)

# import sys

# for batch in testLoader:
#     print(1)
#     print(batch)
#     sys.exit()

# sys.exit()
