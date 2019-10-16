#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(100)
        raw_input = pd.read_csv(train)
        
        
#        Number recognition dataset
        nrows = raw_input.shape[0]
        train_rows = int(0.8*nrows)
        test_rows = nrows-train_rows
        self.train_data = raw_input.loc[:train_rows,:]
        self.test_data = raw_input.loc[train_rows+1:,:]
        
        
        train_dataset = self.preprocess(self.train_data)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.n = nrows

        self.X = train_dataset.iloc[:,1:].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:,0].values.reshape(nrows, 1)

        
        
        
##        Loan Prediction Dataset
#        train_dataset = self.preprocess(raw_input)
#    
#        ncols = len(train_dataset.columns)
#        nrows = len(train_dataset.index)
#        self.n = nrows
#        output_dataset = pd.read_csv("Loan_Y_train.csv")
#        labelencoder_X = LabelEncoder()
#        output_dataset["Target"] = labelencoder_X.fit_transform(output_dataset["Target"])
#        self.X = train_dataset.iloc[:,:].values.reshape(nrows, ncols)
#        self.y = output_dataset.iloc[:, 0].values.reshape(nrows, 1)
#        train_dataset.drop(["Loan_ID"], axis=1)
        
        
        
        #Common part
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])

        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
#            print("else") 
            output_layer_size = len(self.y[0])


        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    
    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation  == "tanh":
            self.__tanh(self,x)
            
        elif activation == "ReLu": 
            self.__relu(self,x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation  == "tanh":
            self.__tanh_derivate(self,x)
            
        elif activation == "ReLu": 
            self.__relu_derivate(self,x)

    def __tanh(self,x):
        return np.tanh(x)
    
    
    def __tanh_derivate(self,x):
        value = (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
        return 1 - np.power(value,2)
    
    def __relu(self,x):
        return np.maximum(0,x)


    def __relu_derivate(self, x):
         x[x<=0] = 0
         x[x>0] = 1
         return x


    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    
    
    def preprocess(self, X):
        
#        Preprocessing for Number recognition 
        X = X.replace(np.NaN, X.mean())
        scaler = MinMaxScaler()
        X.loc[:,:] = scaler.fit_transform(X)
        return X
        
    
##       Preprocessing for Loan dataset 
#        labelencoder_X = LabelEncoder()
#        X["Gender"] = labelencoder_X.fit_transform(X["Gender"])
#        X["Married"] = labelencoder_X.fit_transform(X["Married"])
#        X["Education"] = labelencoder_X.fit_transform(X["Education"])
#        X["Self_Employed"] = labelencoder_X.fit_transform(X["Self_Employed"])
#        max_value =  X["ApplicantIncome"].max()
#        min_value =  X["ApplicantIncome"].min()
#        X["ApplicantIncome"] =  X["ApplicantIncome"].apply(lambda x: (x-min_value)/(max_value-min_value))
#        
#        max_value =  X["CoapplicantIncome"].max()
#        min_value =  X["CoapplicantIncome"].min()
#        X["CoapplicantIncome"] =  X["CoapplicantIncome"].apply(lambda x: (x-min_value)/(max_value-min_value))
#        
#        max_value =  X["LoanAmount"].max()
#        min_value =  X["LoanAmount"].min()
#        X["LoanAmount"] =  X["LoanAmount"].apply(lambda x: (x-min_value)/(max_value-min_value))
#        
#         
#        max_value =  X["Loan_Amount_Term"].max()
#        min_value =  X["Loan_Amount_Term"].min()
#        X["Loan_Amount_Term"] =  X["Loan_Amount_Term"].apply(lambda x: (x-min_value)/(max_value-min_value))
#        
#        
#        max_value =  X["Dependents"].max()
#        if(max_value == "3+"):
#            max_value = 3
#        min_value =  X["Dependents"].min()
#
#        
#        X["Dependents"] =  X["Dependents"].apply(lambda x: (int(3)-int(min_value))/(int(max_value)-int(min_value)) if x == "3+"  else (int(x)-int(min_value))/(int(max_value)-int(min_value)))
#  
#      
#        
#        X["Property_Area"] = labelencoder_X.fit_transform(X["Property_Area"])
#        max_value =  X["Property_Area"].max()
#        min_value =  X["Property_Area"].min()
#        X["Property_Area"] =  X["Property_Area"].apply(lambda x: (x-min_value)/(max_value-min_value))
#        
#        
#        X["Loan_ID"] = labelencoder_X.fit_transform(X["Loan_ID"])
#        max_value =  X["Loan_ID"].max()
#        min_value =  X["Loan_ID"].min()
#        X["Loan_ID"] =  X["Loan_ID"].apply(lambda x: (x-min_value)/(max_value-min_value))
#
#        
#        return X



    def train(self, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation = "tanh")
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation="tanh")
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)/self.n) + "     " +str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)
        return out

    def forward_pass(self, activation ="sigmoid"):
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif activation == "tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif activation == "relu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
#            print("tanh")
            delta_output = (self.y - out) * (self.__tanh_derivate(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivate(out))
        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivate(self.X23))
        elif activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivate(self.X23))
        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivate(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivate(self.X12))

        self.delta12 = delta_hidden_layer1


    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test, header = True):
        raw_test_input = pd.read_csv(test)
        print("Test")
        # TODO: Remember to implement the preprocess method
        
#        NumberPrediction Dataset
        test_dataset = self.preprocess(self.test_data)
        ncols = len(test_dataset.columns)
        nrows = len(test_dataset.index)
        self.n = nrows
        self.X = test_dataset.iloc[:,1:].values.reshape(nrows, ncols-1)
        self.y = test_dataset.iloc[:,0].values.reshape(nrows, 1)
        self.X01 = self.X
        self.X12 = np.zeros((len(self.X), 4))
        self.X23 = np.zeros((len(self.X), 2))
        out = self.forward_pass()
        error = 0.5 * np.power((out - self.y), 2)
        print("error sum " ,np.sum(error))
        print("error sum mean" ,np.sum(error)/nrows)
        
        
##        Loan Prediction Dataset
#        raw_input = pd.read_csv(test)
#        test_dataset = self.preprocess(raw_input)
#        ncols = len(test_dataset.columns)
#        nrows = len(test_dataset.index)
#        self.n = nrows
#        output_dataset = pd.read_csv("Loan_Y_test.csv")
#        labelencoder_X = LabelEncoder()
#        output_dataset["Target"] = labelencoder_X.fit_transform(output_dataset["Target"])
#        
#        self.X = test_dataset.iloc[:,:].values.reshape(nrows, ncols)
#        self.y = output_dataset.iloc[:, 0].values.reshape(nrows, 1)
#        
#        self.X01 = self.X
#        self.X12 = np.zeros((len(self.X), 4))
#        self.X23 = np.zeros((len(self.X), 2))
#        out = self.forward_pass()
#        error = 0.5 * np.power((out - self.y), 2)
#        print("error sum " ,np.sum(error))
#        print("error sum mean" ,np.sum(error)/nrows)



        return out


if __name__ == "__main__":
    
#    Number prediction dataset
    neural_network = NeuralNet("Numbers.csv")
    neural_network.train()
    testError = neural_network.predict("test_number.csv")

    
##    LoanPrediction Dataset
#    neural_network = NeuralNet("Loan_X_train.csv")
#    neural_network.train()
#    testError = neural_network.predict("Loan_X_test.csv")

