import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA_orig import AE


# start-snippet-1
class SdA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            import dA_orig
            dA_layer = dA_orig.AE(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
            
        inp = self.x
        for i in xrange(self.n_layers):
            inp=self.dA_layers[i].get_hidden_values(inp)
        for i in xrange(self.n_layers):
            inp=self.dA_layers[self.n_layers-i-1].get_reconstructed_input(inp)
        self.output = inp
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,##$
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),##$
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        train_set_x=train_set_x.eval()
        train_set_x_lab=train_set_x[:1000,:]
        train_set_y=train_set_y.eval()
        train_set_y_lab=train_set_y[:1000]
    
        import theano
        train_set_x=theano.shared(numpy.asarray(train_set_x_lab,
                                                    dtype=theano.config.floatX),
                                      borrow=True)
        train_set_y=theano.shared(numpy.asarray(train_set_y_lab,
                                                    dtype=theano.config.floatX),
                                      borrow=True)
        train_set_y=T.cast(train_set_y_lab, 'int32')

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_SdA(finetune_lr=0.1, pretraining_epochs=0,
             pretrain_lr=0.05, training_epochs=100,
             dataset='mnist.pkl.gz', batch_size=10):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    train_set_x=train_set_x.eval()
    train_set_y=train_set_y.eval()

    train_set_x_lab=train_set_x[:1000,:]
    train_set_x_unlab=train_set_x[1000:,:]
    train_set_y_lab=train_set_y[:1000]
    train_set_y_unlab=train_set_y[1000:]

    import theano
    train_set_x_lab=theano.shared(numpy.asarray(train_set_x_lab,
                                                dtype=theano.config.floatX),
                                  borrow=True)
    train_set_y_lab=theano.shared(numpy.asarray(train_set_y_lab,
                                                dtype=theano.config.floatX),
                                  borrow=True)
    train_set_y_lab=T.cast(train_set_y_lab, 'int32')
    train_set_x_unlab=theano.shared(numpy.asarray(train_set_x_unlab,
                                                  dtype=theano.config.floatX),
                                    borrow=True)
    train_set_y_unlab=theano.shared(numpy.asarray(train_set_y_unlab,
                                                  dtype=theano.config.floatX),
                                    borrow=True)
    train_set_y_unlab=T.cast(train_set_y_unlab, 'int32')

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_y_lab.eval().shape[0]
    n_train_batches /= batch_size
    n_train_batches_u = train_set_y_unlab.eval().shape[0]
    n_train_batches_u /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    hidden_layer_size = 100
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[100],
        n_outs=10
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x_unlab,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [0.1, 0.2, 0.3]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],##$
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    fprop = theano.function(
            [],
            sda.output,
            givens={
                sda.x: test_set_x
            },
            name='fp'
        )
    Q=fprop()
    print 'rec', ((Q-test_set_x.eval())**2).mean()
    from utils import tile_raster_images,plot_weights
    import PIL.Image as Image
    image = Image.fromarray(
        tile_raster_images(X=sda.dA_layers[0].W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    # end-snippet-4
    ########################
    # FINETUNING THE MODEL FOR REGRESSION #
    if 0 : # pretrain middle layer
        print '... pre-training MIDDLE layer'

        h1 = T.matrix('x')  # the data is presented as rasterized images
        h2 = T.matrix('y')  # the labels are presented as 1D vector of
        log_reg = HiddenLayer(numpy_rng, h1, hidden_layer_size, hidden_layer_size)

        if 1: # for middle layer
            learning_rate = 0.05
            fprop_inp = theano.function(
                [],
                SdA_inp.sigmoid_layers[-1].output,
                givens={
                    SdA_inp.sigmoid_layers[0].input: train_set_x
                },
                name='fprop_inp'
            )
            fprop_out = theano.function(
                [],
                SdA_out.sigmoid_layers[-1].output,
                givens={
                    SdA_out.sigmoid_layers[0].input: train_set_y
                },
                name='fprop_out'
            )
            H11=fprop_inp() 
            H21=fprop_out()
            H1=N1.predict(train_set_x.eval())
            H2=N2.predict(train_set_y.eval())

            H1=theano.shared(H1)
            H2=theano.shared(H2)
            # compute the gradients with respect to the model parameters
            logreg_cost = log_reg.mse(h2)

            gparams = T.grad(logreg_cost, log_reg.params)
    
            # compute list of fine-tuning updates
            updates = [
                (param, param - gparam * learning_rate)
                for param, gparam in zip(log_reg.params, gparams)
            ]

            train_fn_middle = theano.function(
                inputs=[],
                outputs=logreg_cost,
                updates=updates,
                givens={
                    h1: H1,
                    h2: H2
                },
                name='train_middle'
            )
        epoch = 0
        while epoch < 10:
            print epoch, train_fn_middle()
            epoch += 1
            
    
    from mlp import MLP
    net = MLP(numpy_rng, train_set_x_lab, 28*14, hidden_layer_size, 28*14, W1=sda.dA_layers[0].W, b1=sda.dA_layers[0].b, W2=None, b2=None)
    ########################
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_SdA()