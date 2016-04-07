import csv
import numpy as np
import random
import Network as net
import ActivationFunctions as af

X = np.empty(shape=(4000,401), dtype=np.int8)
Y = np.zeros(shape=(4000,10), dtype=np.int8)
X_test = np.empty(shape=(200,401), dtype=np.int8)
Y_test = np.empty(shape=(200,401), dtype=np.int8)

def loadData():
    random.seed(1)
    initElement = np.empty(shape=(28,28))
    fixedElement = np.empty(shape=(401,))

    global X, Y, X_test, Y_test

    with open('train.csv', "rb") as data:
        reader = csv.reader(data)
        for row in reader:
            lineNumber = reader.line_num
            if row == '' or X.shape[0] + X_test.shape[0] <= lineNumber:
                break

            colNum = -1
            for column in row:
                if colNum == -1:
                    if lineNumber < X.shape[0]:
                        Y[lineNumber,column] = 1
                    else:
                        Y_test[lineNumber - X.shape[0],column] = 1
                else:
                    x = colNum // 28
                    y = colNum - x * 28
                    activation = 0
                    if int(column) > 160:
                        activation = 1
                    initElement[x, y] = activation
                colNum += 1

            stepX = 4
            stepY = 4
            for x in range(20):
                for y in range(20):
                    fixedElement[x * 20 + y] = initElement[x + stepX, y + stepY]

            fixedElement[400] = 1
            if lineNumber < X.shape[0]:
                X[lineNumber] = fixedElement
            else:
                X_test[lineNumber - X.shape[0]] = fixedElement


if __name__ == "__main__":

    '''
    Example method to load MNIST dataset from csv
    '''
    loadData()
    '''
    Define activation function. From: ActivationFunctions.py
    '''
    myAf = af.Sigmoid()
    '''
    Format data (output) to fit activation function
    '''
    Y = myAf.format_data(Y)
    Y_test = myAf.format_data(Y_test)
    '''
    Create neural network with defined activation function and hidden layer size
    '''
    neuron = net.NeuralNetwork(myAf, 200)
    '''
    Train neural network with these X, Y, learning rate, iterations
    '''
    neuron.train(X, Y, 0.001, 10000)
    '''
    You can also call train multiple times
    '''
    neuron.train(X, Y, 0.001, 100)
    '''
    Validate trained neural network with this data: input, expected output, iteration count
    '''
    neuron.validate(X_test, Y_test, 200)
    '''
    Export neural network to JSON
    '''
    neuron.export_network("Tahn_w_1.json", "Tahn_w_2.json")

