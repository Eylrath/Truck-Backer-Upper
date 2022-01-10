import numpy as np

def sigm(x):
    return 1/(1+np.exp(-x))

def sigm_1(y):
     return np.log(y/(1-y))

class Layer:
    """class for layer"""
    def __init__(self, numberOfNeurons, numberOfInputs):
        """initializes layer with certain number of inputs, and certain number of neurons"""
        self.numberOfInputs = numberOfInputs
        self.numberOfNeurons = numberOfNeurons
        self.weights = np.random.randn(self.numberOfNeurons,self.numberOfInputs)
        self.bias = np.random.uniform(size=self.numberOfNeurons)
        
        self.tmp_input = None
        self.tmp_gradient = None
        self.tmp_output = None
        
    def forward(self, inputVector):
        """returns vector of results of all feeding inputVector into all neurons from this layer"""
        self.tmp_input = inputVector
#         self.tmp_output = f_act(np.dot(self.weights, inputVector) )#+ self.bias)
        self.tmp_output = sigm(np.dot(self.weights, inputVector) + self.bias)
        return np.array(self.tmp_output)
     
        
    def backward(self, gradient):
        """backward prop algorithm for every neuron in this layer"""
        self.tmp_gradient = gradient * self.tmp_output * (1-self.tmp_output)
        return np.dot(self.tmp_gradient, self.weights) 
        
    def learn(self, learningRate):
        """updates weights and biases in this layer"""
        dBias = dWeights = self.tmp_gradient
        dWeights = np.outer(dWeights, self.tmp_input)
        
        self.weights = self.weights + learningRate * dWeights
        self.bias = self.bias + learningRate * dBias


class Network:
    """class for neuralnet"""
    def __init__(self, inputs):
        """initializes neuralnet with number of inputs = inputs"""
        self.layers = []
        self.numberOfInputs = inputs
        self.numberOfLayers = 0
        
    def addLayer(self, neurons):
        """adds new Layer"""
        if len(self.layers) > 0:
            self.layers.append(Layer(neurons,self.layers[-1].numberOfNeurons))
        else:
            self.layers.append(Layer(neurons,self.numberOfInputs))
        self.numberOfLayers += 1
        
    def forward(self, inputVector):
        """propagates inputVector through the neural net"""
        for layer in self.layers:
            inputVector = layer.forward(inputVector)
        return np.array(inputVector)[0]
            
    
    def backward(self, gradient):
        """propagates inputVector backwards the neural net"""
        i = self.numberOfLayers - 1
        while i >=0:
            gradient = self.layers[i].backward(gradient)
            i -= 1
        return gradient
            
    def learn(self,learningRate):
        """update weights and biases after backward propagation"""
        for layer in self.layers:
            layer.learn(learningRate)


def learn_net(net, X, y, epochs, steps, lr):
    """network net learns from examples where 
        X is input matrix of data
        y is target vector
        epochs - number of epochs, which will be indicating a step of learning
        steps - number of steps per epoch
        lr - learning rate which will be passed into net"""
    error_list = []
    while epochs > 0:
        epoch_error = 0
        for step in range(steps):
            idx = np.random.randint(0, len(X) - 1)
            inp, result = X[idx], sigm(y[idx])
            out = net.forward(np.array(inp))
            error = result - out
            epoch_error += (error * error) / 2
            net.backward(error)
            net.learn(lr)
        epochs -= 1
        error_list.append(epoch_error / steps)
    return error_list

        