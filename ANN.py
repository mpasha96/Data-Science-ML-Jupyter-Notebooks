# coding: utf-8

import pandas as pd
import numpy as np

df = pd.read_csv("NN-DATA.csv")
df = (df - df.mean().values)/df.std().values

X = np.array(df.iloc[:,0:3])
y = np.array(df.iloc[:,-1])



def forward_prop(weights_hidden,weights_output,biases_hidden,bias_output,X):
    
    def sigmoid(arr):
        new_arr = []
        for i in arr:
            new_arr.append(1/(1+np.exp(-i)))
        return new_arr

    hidden = weights_hidden.T.dot(X)+biases_hidden
    normed_hidden = sigmoid(hidden)

    output = weights_output.T.dot(normed_hidden)+bias_output
    normed_output = sigmoid(output)
    
    return normed_hidden,normed_output
    
        
def backward_prop(hidden,output,errors_hidden,errors_output,weights_hidden,weights_output,biases_hidden,bias_output,X,y):
    
    def get_errors():
        for i in range(len(output)):
            errors_output[i] = output[i]*(1-output[i])*(y-output[i])
        for i in range(len(hidden)):
            errors_hidden[i] = np.float64(hidden[i]*(1-hidden[i])*(errors_output[0])*(weights_output[i]))
        return errors_hidden,errors_output

    def update_weights_and_biases():
        # weight update
        for i in range(len(weights_output)):
            weights_output[i] += alpha*errors_output[0]*hidden[i]
        for i in range(len(weights_hidden)):
            weights_hidden[i] += alpha*errors_hidden[i%2]*X[i-len(X)]

        # Biases update
        for i in range(len(bias_output)):
            bias_output[i] += alpha*(errors_output[0])
        for i in range(len(biases_hidden)):
            biases_hidden[i] += alpha*(errors_hidden[i])
            
    errors_hidden,errors_output = get_errors()
    update_weights_and_biases()


weights_hidden = np.reshape( [0.0, 0.0,
                              0.0, 0.0,
                              0.0, 0.0], (3,2))
weights_output = np.reshape([0.0,
                             0.0], (2,1))

biases_hidden = [0.0,0.0]
bias_output = [0.0]

errors_hidden = [0,0]
errors_output = [1]

alpha = 1

error_of_each_row = []
mean_errors_of_each_epoch = [1]
epoch = 0

while True:
    print("\nEpoch #",epoch+1)
    alpha = 1/(epoch+1)
    for i in range(len(X)):
        while True:
            hidden,output = forward_prop(weights_hidden,weights_output,biases_hidden,bias_output,X[i])
            backward_prop(hidden,output,errors_hidden,errors_output,weights_hidden,weights_output,biases_hidden,bias_output,X[i],y[i])
            if(errors_output[0]<0):
                error_of_each_row.append(0.0)
                break
            elif(errors_output[0]<0.01):
                error_of_each_row.append(errors_output[0])
                break  
    mean_errors_of_each_epoch.append(np.mean(error_of_each_row))
    print("Mean Error of all rows:",np.mean(error_of_each_row))
    print("Error difference with previous epoch:",mean_errors_of_each_epoch[-2]-mean_errors_of_each_epoch[-1])
    epoch+=1
    if((mean_errors_of_each_epoch[-2]-mean_errors_of_each_epoch[-1])<0.00001):
        break
