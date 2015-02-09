""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import numpy

def load_vc(dataset='../gitlab/voice-conversion/src/test/data/clb_slt_MCEP24_static_span0.data'):
    import sys
    sys.path.append('../gitlab/voice-conversion/src')
    import voice_conversion
    
    import pickle
    f=open(dataset,'r')
    vcdata=pickle.load(f)
    x=vcdata['aligned_data1'][:,:24]
    y=vcdata['aligned_data2'][:,:24]
    num = x.shape[0]
    st_train = 0
    en_train = int(num * (64.0/200.0))
    st_valid = en_train
    en_valid = en_train+int(num * (36.0/200.0))
    st_test = en_valid
    en_test = num
    
    x_mean = x[st_train:en_train,:].mean(axis=0)
    y_mean = y[st_train:en_train,:].mean(axis=0)
    x_std = x[st_train:en_train,:].std(axis=0)
    y_std = y[st_train:en_train,:].std(axis=0)
    x -= x_mean
    y -= y_mean
    x /= x_std
    y /= y_std

    import theano
    train_set_x = theano.shared(numpy.asarray(x[st_train:en_train,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    train_set_y = theano.shared(numpy.asarray(y[st_train:en_train,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_x = theano.shared(numpy.asarray(x[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_y = theano.shared(numpy.asarray(y[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    valid_set_x = theano.shared(numpy.asarray(x[st_valid:en_valid,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    valid_set_y = theano.shared(numpy.asarray(y[st_valid:en_valid,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval, x_mean, y_mean, x_std, y_std
def load_data_half(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # LOAD DATA #
    import os
    import cPickle
    import gzip
    import theano
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

    def shared_dataset(data_xy, train_xy, borrow=True):       
        data_x, data_y = data_xy
        data_x = data_x.reshape((data_x.shape[0], 28,28))
        data_y = data_x[:,:,14:].reshape((data_x.shape[0], 28*14))
        data_x = data_x[:,:,:14].reshape((data_x.shape[0], 28*14))
        t_x, t_y = train_xy
        t_x = t_x.reshape((t_x.shape[0], 28,28))
        t_y = t_x[:,:,14:].reshape((t_x.shape[0], 28*14))
        t_x = t_x[:,:,:14].reshape((t_x.shape[0], 28*14))
        #data_x = data_x - t_x.mean(axis=0)
        #data_y = data_y - t_y.mean(axis=0)

        #for j in range(data_x.shape[1]):
            #data_x[:, j] -= numpy.mean(data_x[:, j])
        #for j in range(data_y.shape[1]):
            #data_y[:, j] -= numpy.mean(data_y[:, j])
        #data_x = data_x[:5000,:]
        #data_y = data_y[:5000,:]

        #data_y = data_y[:]

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        
        return shared_x, shared_y

    
    train_set_x, train_set_y = shared_dataset(train_set, train_set)
    test_set_x, test_set_y = shared_dataset(test_set, train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set, train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval




def plot_weights(w, M=28, N=28, num=10):
    import numpy as np
    try:
        from matplotlib import pyplot as pp
        import matplotlib.cm as cm
    except ImportError:
        print 'matplotlib is could not be imported'

    a=np.zeros((M*num,N*num))
    for i in range(num*num):
        m=i%num
        n=i/num
        a[m*M:(m+1)*M, n*N:(n+1)*N] = w[i,:].reshape((M,N))
    pp.imshow(a,interpolation='none',aspect='auto',cmap=cm.Greys)
    #pp.show()
    
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array