import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#layer initializer
class Layer_Dense:

    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    #forward pass
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
    
    #backwards pass
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)

#Relu activation func
class Activation_Relu:

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

#softmax activation function
class Activation_Softmax:

    def forward(self,inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities
    
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output,single_dvalues) in \
            enumerate(zip(self.output,dvalues)):

            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output,single_output.T)
        
        self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

#common loss
class Loss:

    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

#cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, \
                                         axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

#softmax classifier, combines softmax activation and crossentropy loss for a quicker backwards step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs / samples



if __name__ == '__main__':
    X,y = spiral_data(samples=100,classes=3)
    dense1 = Layer_Dense(2,3)
    activation1 = Activation_Relu()
    dense2 = Layer_Dense(3,3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    print(loss_activation.output[:5])
    print('Loss : ',loss)

    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions==y)

    print('Acc : ',accuracy)




    