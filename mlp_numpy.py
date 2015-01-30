import numpy as np
from scipy.special import expit
import copy

def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1.0 - np.tanh(x)**2.0
def linear(x):
    return x
def linear_prime(x):
    return 1.0
def relu(x):
    y = copy.deepcopy(x)
    y[y<=0] = 0.0
    return y

def relu_prime(x):
    y = copy.deepcopy(x)
    y[y>0] = 1.0
    y[y<=0] = 0.0
    return y

def logistic(x):
    return 1.0/(1+np.exp(-x))
 
def logistic_prime(x):
    ex=expit(-x)
    return ex/(1+ex)**2
 
def identity(x):
    return x
 
def identity_prime(x):
    return 1

class NeuralNetwork(object):
    def __init__(self, X, y, parameters):
        #Input data
        self.X=X
        #Output data
        self.y=y
        #Expect parameters to be a tuple of the form:
        #    ((n_input,0,0), (n_hidden_layer_1, f_1, f_1'), ...,
        #     (n_hidden_layer_k, f_k, f_k'), (n_output, f_o, f_o'))
        self.n_layers = len(parameters)
        #Counts number of neurons without bias neurons in each layer.
        self.sizes = [layer[0] for layer in parameters]
        #Activation functions for each layer.
        self.fs =[layer[1] for layer in parameters]
        #Derivatives of activation functions for each layer.
        self.fprimes = [layer[2] for layer in parameters]
        self.build_network()
 
    def build_network(self):
        #List of weight matrices taking the output of one layer to the input of the next.
        self.weights=[]
        #Bias vector for each layer.
        self.biases=[]
        #Input vector for each layer.
        self.inputs=[]
        #Output vector for each layer.
        self.outputs=[]
        #Vector of errors at each layer.
        self.errors=[]
        #We initialise the weights randomly, and fill the other vectors with 1s.
        for layer in range(self.n_layers-1):
            n = self.sizes[layer]
            m = self.sizes[layer+1]
            self.weights.append(np.random.normal(0,0.1, (m,n)))
            self.biases.append(np.random.normal(0,0.1,(m,1)))
            self.inputs.append(np.zeros((n,1)))
            self.outputs.append(np.zeros((n,1)))
            self.errors.append(np.zeros((n,1)))
        #There are only n-1 weight matrices, so we do the last case separately.
        n = self.sizes[-1]
        self.inputs.append(np.zeros((n,1)))
        self.outputs.append(np.zeros((n,1)))
        self.errors.append(np.zeros((n,1)))
 
    def feedforward(self, x):
        #Propagates the input from the input layer to the output layer.
        #k=len(x)
        #x.shape=(k,1)
        self.inputs[0]=x
        self.outputs[0]=x
        for i in range(1,self.n_layers):
            self.inputs[i]=self.weights[i-1].dot(self.outputs[i-1])+self.biases[i-1]
            self.outputs[i]=self.fs[i](self.inputs[i])
        return self.outputs[-1]
    def feedforwardnoisy(self, x):
        #Propagates the input from the input layer to the output layer.
        #k=len(x)
        #x.shape=(k,1)
        self.inputs[0]=x
        self.outputs[0]=x
        for i in range(1,self.n_layers):
            self.outputs[i-1]=copy.deepcopy(self.outputs[i-1]+np.random.normal(0, 0.1, self.outputs[i-1].shape))
            self.inputs[i]=self.weights[i-1].dot(self.outputs[i-1])+self.biases[i-1]
            self.outputs[i]=self.fs[i](self.inputs[i])
        return self.outputs[-1]
 
    def update_weights(self,x,y):
        #Update the weight matrices for each layer based on a single input x and target y.
        output = self.feedforwardnoisy(x)
        self.errors[-1]=self.fprimes[-1](self.outputs[-1])*(output-y)
 
        n=self.n_layers-2
        for i in xrange(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])
            self.weights[i] = self.weights[i]-self.learning_rate*np.outer(self.errors[i+1],self.outputs[i])
            self.biases[i] = self.biases[i] - self.learning_rate*self.errors[i+1]
        self.weights[0] = self.weights[0]-self.learning_rate*np.outer(self.errors[1],self.outputs[0])
        self.biases[0] = self.biases[0] - self.learning_rate*self.errors[1] 
    def update_weights_batch(self,x,y):
        #Update the weight matrices for each layer based on a single input x and target y.
        output = self.feedforwardnoisy(x)
        self.errors[-1]=self.fprimes[-1](self.outputs[-1])*(output-y)
 
        n=self.n_layers-2
        for i in xrange(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])
            self.weights[i] = self.weights[i]-self.learning_rate*np.outer(self.errors[i+1],self.outputs[i])
            self.biases[i] = self.biases[i] - self.learning_rate*self.errors[i+1]
        self.weights[0] = self.weights[0]-self.learning_rate*np.outer(self.errors[1],self.outputs[0])
        self.biases[0] = self.biases[0] - self.learning_rate*self.errors[1] 
    
    def train(self, X, Y, n_iter, learning_rate=1):
        #Updates the weights after comparing each input in X with y
        #repeats this process n_iter times.
        self.X = X
        self . y= Y
        self.learning_rate=learning_rate
        n=self.X.shape[0]
        for repeat in range(n_iter):
            #We shuffle the order in which we go through the inputs on each iter.
            index=list(range(n))
            np.random.shuffle(index)
            for row in index:
                x=self.X[row:row+1,:]
                y=self.y[row:row+1,:]
                self.update_weights(x.T,y.T)

 
    def train_sgd(self, X, Y, n_iter, learning_rate=1):
        #Updates the weights after comparing each input in X with y
        #repeats this process n_iter times.
        self.X = X
        self . y= Y
        self.learning_rate=learning_rate
        n=self.X.shape[0]
        for repeat in range(n_iter):
            #We shuffle the order in which we go through the inputs on each iter.
            index=list(range(n))
            np.random.shuffle(index)
            for row in index:
                x=self.X[row:row+1,:]
                y=self.y[row:row+1,:]
                self.update_weights(x.T,y.T)

 
    
    def predict_x(self, x):
        return self.feedforward(x)
 
    def predict(self, X):
        n = len(X)
        m = self.sizes[-1]
        ret = np.ones((n,m))
        for i in range(len(X)):
            ret[i,:] = self.feedforward(X[i:i+1,:].T).reshape((m))
        return ret
#expit is a fast way to compute logistic using precomputed exp.
def test_regression(plots=False):
    #First create the data.
    n=200
    X=np.linspace(0,3*np.pi,num=n)
    X.shape=(n,1)
    y=np.sin(X)
    #We make a neural net with 2 hidden layers, 20 neurons in each, using logistic activation
    #functions.
    param=((1,0,0),(20, expit, logistic_prime),(20, expit, logistic_prime),(1,identity, identity_prime))
    #Set learning rate.
    rates=[0.05]
    predictions=[]
    for rate in rates:
        N=NeuralNetwork(X,y,param)
        N.train(40, learning_rate=rate)
        predictions.append([rate,N.predict(X)])
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(1,1)
    if plots:
        ax.plot(X,y, label='Sine', linewidth=2, color='black')
        for data in predictions:
            ax.plot(X,data[1],label="Learning Rate: "+str(data[0]))
        ax.legend()
    plt.show()

if __name__ == "__main__":
    test_regression(True)