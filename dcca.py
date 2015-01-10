"""
This file is part of deepcca.

deepcca is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

deepcca is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with deepcca.  If not, see <http://www.gnu.org/licenses/>.
"""
"""

Deep Canonical Correlation Analysis

References:

[1] G. Andrew, R. Arora, J. Bilmes, and K. Livescu. \
Deep canonical correlation analysis. In Proc. of\
the 30th Intl. Conference on Machine Learning, p\
ages 1247–1255, Atlanta ,Georgia, USA, 2013.
[2] http://deeplearning.net/


"""
import os
import sys
import time
import gzip
import cPickle

import numpy

import theano
import theano.tensor as T

from mlp import load_data, HiddenLayer, MLP

class MLPCCA(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a sigmoid activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.sigmoid
        )

        self.logRegressionLayer = CCALayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=T.nnet.sigmoid
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.mse = (
            self.logRegressionLayer.mse
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3



class DCCAold(object):
    def __init__(self, rng, x1, x2, n_in1, n_hidden1, n_out1, n_in2, n_hidden2, n_out2):    
        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=x1,
            n_in=n_in1,
            n_out=n_hidden1,
            activation=T.nnet.sigmoid
        )

        self.lastLayer1 = CCALayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden1,
            n_out=n_out1,
            activation=T.nnet.sigmoid
        )
        
        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=x2,
            n_in=n_in2,
            n_out=n_hidden2,
            activation=T.nnet.sigmoid
        )

        self.lastLayer2 = CCALayer(
            rng=rng,
            input=self.hiddenLayer2.output,
            n_in=n_hidden2,
            n_out=n_out2,
            activation=T.nnet.sigmoid
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L11 = (
            abs(self.hiddenLayer1.W).sum()
            + abs(self.lastLayer1.W).sum()
        )

        self.L12 = (
            abs(self.hiddenLayer2.W).sum()
            + abs(self.lastLayer2.W).sum()
        )
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr1 = (
            (self.hiddenLayer1.W ** 2).sum()
            + (self.lastLayer1.W ** 2).sum()
        )
        self.L2_sqr2 = (
            (self.hiddenLayer2.W ** 2).sum()
            + (self.lastLayer2.W ** 2).sum()
        )

        self.correlation = (
            self.lastLayer1.correlation
        )
        
        self.errors = self.lastLayer1.errors
       
        self.params1 = self.hiddenLayer1.params + self.lastLayer1.params
        self.params2 = self.hiddenLayer2.params + self.lastLayer2.params

class DCCA(MLP):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.sigmoid
        )

        self.lastLayer = CCALayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=T.nnet.sigmoid
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.lastLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.lastLayer.W ** 2).sum()
        )

        self.correlation = (
            self.lastLayer.correlation
        )
        #self.errors = self.lastLayer.errors
        self.output = self.lastLayer.output
        self.params = self.hiddenLayer.params + self.lastLayer.params        
    
class CCALayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is sigmoid

        Hidden unit activation is given by: sigmoid(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.n_in = n_in
        self.n_out = n_out
        self.input = input
        self.activation = activation

        self.r1 = 0.001
        self.r2 = 0.001

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        
        self.params = [self.W, self.b]

    def correlation(self, H2):
        H1 = self.output.T
        
        H1bar = H1 #- T.mean(H1,axis=0)#(1.0/self.n_out)*T.dot(H1, T.ones_like())
        H2bar = H2 #- T.mean(H2,axis=0)#(1.0/self.n_out)*T.dot(H2, T.ones_like())
        SigmaHat12 = (1.0/(self.n_out-1))*T.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0/(self.n_out-1))*T.dot(H1bar, H1bar.T)
        SigmaHat11 = SigmaHat11 + self.r1*T.identity_like(SigmaHat11)
        SigmaHat22 = (1.0/(self.n_out-1))*T.dot(H2bar, H2bar.T)
        SigmaHat22 = SigmaHat22 + self.r2*T.identity_like(SigmaHat22)
        Tval = T.dot(SigmaHat11**(-0.5), T.dot(SigmaHat12, SigmaHat22**(-0.5)))
        corr = T.nlinalg.trace(T.dot(Tval.T, Tval))**(0.5)
        self.SigmaHat11 = SigmaHat11
        self.SigmaHat12 = SigmaHat12
        self.SigmaHat22 = SigmaHat22
        self.H1bar = H1bar
        self.H2bar = H2bar
        self.Tval = Tval
        return -1*corr
   
def test_dcca_old(learning_rate=0.01, L1_reg=0.0001, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    if 0:
        net1 = MLP(
            rng=rng,
            input=x,
            n_in=28 * 28,
            n_hidden=300,
            n_out=50
        )
        net2 = MLP(
            rng=rng,
            input=y,
            n_in=10,
            n_hidden=20,
            n_out=5
        )
    
    net = DCCA(
        rng=rng,
        x1=x,
        x2=y,
        n_in1=28 * 28,
        n_hidden1=300,
        n_out1=50,
        n_in2=10,
        n_hidden2=20,
        n_out2=5
    )
  

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost1 = (
        net.correlation(y)
        + L1_reg * net.L11
        + L2_reg * net.L2_sqr1
    )
    cost2 = (
        net.correlation(y)
        + L1_reg * net.L12
        + L2_reg * net.L2_sqr2
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    """
    test_model = theano.function(
        inputs=[index],
        outputs=net1.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    """

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams1 = [T.grad(cost1, param) for param in net.params1]
    gparams2 = [T.grad(cost2, param) for param in net.params2]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates1 = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(net.params1, gparams1)
    ]
    updates2 = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(net.params2, gparams2)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model1 = theano.function(
        inputs=[index],
        outputs=cost1,
        updates=updates1,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_model2 = theano.function(
        inputs=[index],
        outputs=cost2,
        updates=updates2,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        #import copy
        data_x, data_y = data_xy
        #daya_y = copy.deepcopy(data_x)
        data_y_new = numpy.zeros((data_y.shape[0], data_y.max()+1))
        for i in range(data_y.shape[0]):
            data_y_new[i, data_y[i]] = 1
        data_y = data_y_new
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def test_dcca(learning_rate=0.01, L1_reg=0.0001, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x1 = T.matrix('x1')  # the data is presented as rasterized images
    x2 = T.matrix('x2')  # the labels are presented as 1D vector of
                        # [int] labels
    h1 = T.matrix('h1')  # the data is presented as rasterized images
    h2 = T.matrix('h2')  # the labels are presented as 1D vector of

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    net1 = DCCA(
        rng=rng,
        input=x1,
        n_in=28 * 28,
        n_hidden=300,
        n_out=8
    )
    net2 = DCCA(
        rng=rng,
        input=x2,
        n_in=10,
        n_hidden=20,
        n_out=8
    )
 
    cost1 = (
        net1.correlation(h2)
        + L1_reg * net1.L1
        + L2_reg * net1.L2_sqr
    )
    cost2 = (
        net2.correlation(h1)
        + L1_reg * net2.L1
        + L2_reg * net2.L2_sqr
    )
   
    """
    test_model = theano.function(
        inputs=[index],
        outputs=net1.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    """
    fprop_model1 = theano.function(
        inputs=[],
        outputs=net1.output,
        givens={
            x1: test_set_x
        }
    )
    fprop_model2 = theano.function(
        inputs=[],
        outputs=net2.output,
        givens={
            x2: test_set_y
        }
    )
    if 1: # grad compute for net1
        U, V, D = theano.tensor.nlinalg.svd(net1.lastLayer.Tval)
        UVT = T.dot(U, V.T)
        Delta12 = T.dot(net1.lastLayer.SigmaHat11**(-0.5), T.dot(UVT, net1.lastLayer.SigmaHat22**(-0.5)))
        UDUT = T.dot(U, T.dot(D, U.T))
        Delta11 = (-0.5) * T.dot(net1.lastLayer.SigmaHat11**(-0.5), T.dot(UVT, net1.lastLayer.SigmaHat22**(-0.5)))
        grad_E_to_o = (1.0/8) * (2*Delta11*net1.lastLayer.H1bar+Delta12*net1.lastLayer.H2bar)
        gparam1_W = (grad_E_to_o) * (net1.lastLayer.output*(1-net1.lastLayer.output)) * (net1.hiddenLayer.output)
        gparam1_b = (grad_E_to_o) * (net1.lastLayer.output*(1-net1.lastLayer.output)) * 1
        #gparams1 = [T.grad(cost1, param) for param in net1.params]
        gparams1 = [T.grad(cost1, param) for param in net1.hiddenLayer.params]
        gparams1.append(gparam1_W)
        gparams1.append(gparam1_b)
    if 1: # grad compute for net2
        U, V, D = theano.tensor.nlinalg.svd(net2.lastLayer.Tval)
        UVT = T.dot(U, V.T)
        Delta12 = T.dot(net2.lastLayer.SigmaHat11**(-0.5), T.dot(UVT, net2.lastLayer.SigmaHat22**(-0.5)))
        UDUT = T.dot(U, T.dot(D, U.T))
        Delta11 = (-0.5) * T.dot(net2.lastLayer.SigmaHat11**(-0.5), T.dot(UVT, net2.lastLayer.SigmaHat22**(-0.5)))
        grad_E_to_o = (1.0/8) * (2*Delta11*net2.lastLayer.H1bar+Delta12*net2.lastLayer.H2bar)
        gparam2_W = (grad_E_to_o) * (net2.lastLayer.output*(1-net2.lastLayer.output)) * (net2.hiddenLayer.output)
        gparam2_b = (grad_E_to_o) * (net2.lastLayer.output*(1-net2.lastLayer.output)) * 1
        #gparams1 = [T.grad(cost1, param) for param in net1.params]
        gparams2 = [T.grad(cost2, param) for param in net2.hiddenLayer.params]
        gparams2.append(gparam2_W)
        gparams2.append(gparam2_b)

        #gparams2 = [T.grad(cost2, param) for param in net2.params]

    updates1 = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(net1.params, gparams1)
    ]
    updates2 = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(net2.params, gparams2)
    ]
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print 'epoch', epoch
        #net1.fprop(test_set_x)
        #net2.fprop(test_set_y)
        h1tmpval = fprop_model1().T
        h2tmpval = fprop_model2().T
        if 1:
            H1 = h1tmpval
            H2 = h2tmpval
            H1bar = H1 -(1.0/8.0)*numpy.dot(H1, numpy.ones((H1.shape[1],H1.shape[1])))
            H2bar = H2 -(1.0/8.0)*numpy.dot(H2, numpy.ones((H1.shape[1],H1.shape[1])))
            SigmaHat12 = (1.0/(8-1))*numpy.dot(H1bar, H2bar.T)
            SigmaHat11 = (1.0/(8-1))*numpy.dot(H1bar, H1bar.T)
            SigmaHat11 = SigmaHat11 + 0.0001*numpy.identity(SigmaHat11.shape[0])
            SigmaHat22 = (1.0/(8-1))*numpy.dot(H2bar, H2bar.T)
            SigmaHat22 = SigmaHat22 + 0.0001*numpy.identity(SigmaHat22.shape[0])
            Tval = numpy.dot(SigmaHat11**(-0.5), numpy.dot(SigmaHat12, SigmaHat22**(-0.5)))
            corr = numpy.trace(numpy.dot(Tval.T, Tval))**(0.5)
        #X_theano = theano.shared(value=X, name='inputs')
        #h1tmp = theano.shared( value=h1tmpval, name='hidden1_rep', dtype=theano.config.floatX , borrow=True)
        h1tmp = theano.shared(numpy.asarray(h1tmpval,dtype=theano.config.floatX),
                                 borrow=True)
        #h2tmp = theano.shared( value=h2tmpval, name='hidden2_rep', dtype=theano.config.floatX , borrow=True)
        h2tmp = theano.shared(numpy.asarray(h2tmpval,dtype=theano.config.floatX),
                                 borrow=True)
        #h1tmp = T.shared( value=net1.output.eval(), name='hidden1_rep' )
        #h2tmp = T.shared( net2.output.eval() )

        train_model1 = theano.function(
            inputs=[],
            outputs=cost1,
            updates=updates1,
            givens={
                x1: test_set_x,
                h2: h2tmp
            }
        )
        train_model2 = theano.function(
            inputs=[],
            outputs=cost2,
            updates=updates2,
            givens={
                x2: test_set_y,
                h1: h1tmp,
            }
        )
       
        minibatch_avg_cost1 = train_model1()
        minibatch_avg_cost2 = train_model2()
        print 'corr1', minibatch_avg_cost1
        print 'corr2', minibatch_avg_cost2

        if epoch > 10:
            break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        #import copy
        data_x, data_y = data_xy
        #daya_y = copy.deepcopy(data_x)
        data_y_new = numpy.zeros((data_y.shape[0], data_y.max()+1))
        for i in range(data_y.shape[0]):
            data_y_new[i, data_y[i]] = 1
        data_y = data_y_new
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    test_dcca()
