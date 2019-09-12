# I assume m is only one neural atom, so in digit example, to get the lists of m(i1, 1) and m(i2, 1), 
# you can call get_m_outputs(m(i1, 1), dic_nn, dic_data) and get_m_outputs(m(i2, 1), dic_nn, dic_data). 
# let me know if you want to improve this function in other way. 

def get_m_outputs(m, dic_nn, dic_data):
    # dic_nn is a dictionary, where the key are the names of neural networks, the value are the  matched neural network models.
    model = dic_nn[m]
    # m = pred(x, y), extract x from m. 
    data_key = m.split("(")[1].split(",")[0]

    # get data from dic_data
    data = Variable(torch.from_numpy(dic_data[data_key]).float(), requires_grad=False)
    
    # feed data to model.
    output = model(data)
    probs = output.tolist()
    return probs