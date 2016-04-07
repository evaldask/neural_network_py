import codecs
import numpy as np
import json
import ActivationFunctions as af

class NeuralNetwork:
    '''
    Notes:
        Main class of neural network. Used to train, import/export and validate neural net.

    Attributes:
        _activation_function (class): activation function class for neural network. Use classes from
        ActivationFunctions.py
        _hidden_size (int): size of the hidden layer in this neural network
        _W0 (numpy matrix): weights of the first layer in network
        _W1 (numpy matrix): weights of the second layer in network

    '''
    _activation_function = af.Tahn()
    _hidden_size = 1
    _W0 = 0
    _W1 = 0

    def __init__(self, activation_function, hidden_size):
        '''
        Note:
            Constructor of the class NeuralNetwork

        Args:
            activation_function: defines what activation function should be used in neural network
            hidden_size: size of the hidden layer in this neural network

        '''
        
        self._activation_function = activation_function
        self._hidden_size = hidden_size

    def train(self, data_x, data_y, alpha, iterations):
        '''
        Note:
            The main method of the class. Used to train W0 and W1 weights for neural network

        Args:
            data_x: input training data
            data_y: output training data
            alpha: learning rate
            iterations: how many times with given data should neural network be trained

        '''
        if isinstance(self._W0, np.ndarray) and isinstance(self._W1, np.ndarray):
            syn0 = self._W0
            syn1 = self._W1
        else:
            np.random.seed(1)
            syn0 = 2 * np.random.random((data_x.shape[1], self._hidden_size)) - 1
            syn1 = 2 * np.random.random((self._hidden_size, data_y.shape[1])) - 1

        print("Started training.")
        for iter in range(iterations + 1):
            '''
            Do the forward propagation
            '''
            layer_0 = data_x
            layer_1 = self._activation_function.forward(np.dot(layer_0, syn0))
            layer_2 = self._activation_function.forward(np.dot(layer_1, syn1))
            '''
            Calculate second layer error and delta
            '''
            l2_error =  layer_2 - data_y
            l2_delta = l2_error * self._activation_function.backward(layer_2)

            '''
            After every 100 training log general error for the network
            '''
            if(iter % 100) == 0:
                print("Error (iterations: " + str(iter) + "): " + str(np.mean(np.abs(l2_error *  data_y.shape[1]))))

            '''
            Calculate first's layer error and delta
            '''
            l1_error = l2_delta.dot(syn1.T)
            l1_delta = l1_error * self._activation_function.backward(layer_1)
            '''
            Modify synapses depending on how good network predicted output
            '''
            syn1 -= alpha * (np.dot(layer_1.T, l2_delta))
            syn0 -= alpha * (np.dot(layer_0.T, l1_delta))

        '''
        Save trained weights
        '''
        self._W0 = syn0
        self._W1 = syn1
        print("Training completed.")


    def import_network(self, name1, name2):
        '''
        Note:
            Imports weights from JSON

        Args:
            name1: name of file from where W0 should be imported
            name2: name of file from where W0 should be imported

        '''

        self._W0 = self._import_data(name1)
        self._W1 = self._import_data(name2)
        print("Weights imported.")

    
    def export_network(self, name1, name2):
        '''
        Note:
            Exports weights to JSON

        Args:
            name1: name of file where W0 should be exported
            name2: name of file where W1 should be exported

        '''
        
        self._export_data(self._W0, name1)
        self._export_data(self._W1, name2)
        print("Weights exported.")

    def validate(self, data_x, data_y, iterations):
        '''
        Note:
            Returns predicted output from given data. Uses trained/imported weights

        Args:
            data_x: validation input
            data_y: expected validation output
            iterations: how many elements should be validated

        Returns:
            wrong: how many elements were identified wrong from given data
        '''
        wrong = 0
        for i in range(iterations):
            calculated = self._predict(data_x[i])
            realValue = np.argmax(data_y[i])

            if calculated != realValue:
                print "Y: " + str(realValue) + " Predicted: " + str(calculated) + " at " + str(i) + " index"
                wrong += 1

        errorRate = round((wrong / float(iterations)) * 100, 6)
        print("Error %: " + str(errorRate))
        print("Total wrong: " + str(wrong))

        return wrong

    def _predict(self, data):
        '''
        Note:
            Returns predicted output from given data. Uses trained/imported weights

        Args:
            data: The first parameter.

        Returns:
            selected: selected class id
        '''

        l1 = self._activation_function.forward(np.dot(data, self._W0))
        l2 = (np.dot(l1, self._W1))
        selected = np.argmax(af.Activation().normalize(l2))
        return selected

    
    def _export_data(self, array, name):
        '''
        Note:
            Exports data to JSON file

        Args:
            array: array to export
            name: exported file name
        '''
        
        json.dump(array.tolist(), codecs.open(name, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    
    def _import_data(self, name):
        '''
        Note:
            Imports data from JSON file

        Args:
            name: name of file to import

        Returns:
            imported: imported network from file
        '''

        obj_text = codecs.open(name, 'r', encoding='utf-8').read()
        imported = np.array(json.loads(obj_text))

        return imported