import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import itertools
import plotly.graph_objs as go
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
train.head()

def one_hot_encode(indices, num_classes):
    """
    Converts a list of integer indices into a one-hot encoding matrix.

    Parameters:
    indices (list): A list of integer indices representing the classes.
    num_classes (int): The number of classes.

    Returns:
    one_hot (numpy.ndarray): A 2D numpy array of shape (len(indices), num_classes) containing the one-hot encoding representation.
    """
    one_hot = np.zeros((len(indices), num_classes), dtype=np.float32)
    one_hot[np.arange(len(indices)), indices] = 1
    return one_hot
X = train.iloc[:, 1:].values
y = train.iloc[:, 0].values
X = X.reshape(-1, 28, 28, 1)
X = X / 255
n_classes = train.label.nunique()
y = one_hot_encode(y, n_classes)
X_subset = X[:200]
y_subset = y[:200]
px.imshow(X_subset[20].reshape(28, 28), color_continuous_scale='ice')

def zero_pad(X, padding):
    """
    Pad with zeros all images of the dataset X

    Argument:
    X --  numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad

class Conv2D:
    """
    A 2D Convolutional Layer in a Neural Network.

    Attributes:
        filters (int): The number of filters in the Convolutional layer.
        filter_size (int): The size of the filters.
        input_channels (int, optional): The number of input channels. Default is 3.
        padding (int, optional): The number of zero padding to be added to the input image. Default is 0.
        stride (int, optional): The stride length. Default is 1.
        learning_rate (float, optional): The learning rate to be used during training. Default is 0.001.
        optimizer (object, optional): The optimization method to be used during training. Default is None.
        cache (dict, optional): A dictionary to store intermediate values during forward and backward pass. Default is None.
        initialized (bool, optional): A flag to keep track of whether the layer has been initialized. Default is False.
    """

    def __init__(self, filters, filter_size, input_channels=3, padding=0, stride=1, learning_rate=0.001, optimizer=None):
        """
        Initialize the Conv2D layer with the given parameters.

        Args:
            filters (int): The number of filters in the Convolutional layer.
            filter_size (int): The size of the filters.
            input_channels (int, optional): The number of input channels. Default is 3.
            padding (int, optional): The number of zero padding to be added to the input image. Default is 0.
            stride (int, optional): The stride length. Default is 1.
            learning_rate (float, optional): The learning rate to be used during training. Default is 0.001.
            optimizer (object, optional): The optimization method to be used during training. Default is None.
        """
        self.filters = filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.padding = padding
        self.stride = stride
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.cache = None
        self.initialized = False

    def relu(self, Z):
        """
        Implement the ReLU function.

        Arguments:
        Z -- Output of the linear layer

        Returns:
        A -- Post-activation parameter
        cache -- used for backpropagation
        """
        A = np.maximum(0, Z)
        cache = Z
        return (A, cache)

    def relu_backward(self, dA, activation_cache):
        """
        Implement the backward propagation for a single ReLU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        activation_cache -- "Z" where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        Z = activation_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def conv_single_step(self, a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
        of the previous layer.

        Parameters:
        a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
        Returns:
        A -- result of applying the activation function to Z
        cache -- used for backpropagation
        """
        s = np.multiply(a_slice_prev, W)
        Z = np.sum(s)
        Z = Z + float(b)
        return Z

    def forward(self, A_prev):
        """
        Implements the forward propagation for a convolution function

        Parameters:
        A_prev -- output activations of the previous layer,
            numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
        activation_caches = []
        if self.initialized == False:
            np.random.seed(0)
            self.W = np.random.randn(self.filter_size, self.filter_size, A_prev.shape[-1], self.filters)
            self.b = np.random.randn(1, 1, 1, self.filters)
            self.initialized = True
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = self.W.shape
        n_H = int((n_H_prev - f + 2 * self.padding) / self.stride) + 1
        n_W = int((n_W_prev - f + 2 * self.padding) / self.stride) + 1
        Z = np.zeros((m, n_H, n_W, n_C))
        A_prev_pad = zero_pad(A_prev, self.padding)
        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            for h in range(n_H):
                vert_start = h * self.stride
                vert_end = vert_start + f
                for w in range(n_W):
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + f
                    for c in range(n_C):
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        weights = self.W[:, :, :, c]
                        biases = self.b[:, :, :, c]
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, weights, biases)
                (Z[i], activation_cache) = self.relu(Z[i])
                activation_caches.append(activation_cache)
        self.cache = (A_prev, np.array(activation_caches))
        return Z

    def backward(self, dZ):
        """
        Implement the backward propagation for a convolution function

        Parameters:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()

        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
              numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
              numpy array of shape (1, 1, 1, n_C)
        """
        (A_prev, activation_cache) = self.cache
        (W, b) = (self.W, self.b)
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape
        stride = self.stride
        pad = self.padding
        (m, n_H, n_W, n_C) = dZ.shape
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        self.dW = np.zeros((f, f, n_C_prev, n_C))
        self.db = np.zeros((1, 1, 1, n_C))
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
        for i in range(m):
            dZ[i] = self.relu_backward(dZ[i], activation_cache[i])
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(n_H):
                vert_start = h * stride
                vert_end = vert_start + f
                for w in range(n_W):
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    for c in range(n_C):
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                        self.dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        self.db[:, :, :, c] += dZ[i, h, w, c]
            if pad:
                dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            else:
                dA_prev[i, :, :, :] = dA_prev[i, :, :, :]
        self.update_parameters(self.optimizer)
        return dA_prev

    def Adam(self, beta1=0.9, beta2=0.999):
        """
        Update parameters using Adam

        Parameters:
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        """
        self.epsilon = 1e-08
        self.v_dW = np.zeros(self.W.shape)
        self.v_db = np.zeros(self.b.shape)
        self.s_dW = np.zeros(self.W.shape)
        self.s_db = np.zeros(self.b.shape)
        self.t = 1
        self.v_dW = beta1 * self.v_dW + (1 - beta1) * self.dW
        self.v_db = beta1 * self.v_db + (1 - beta1) * self.db
        self.v_dW_corrected = self.v_dW / (1 - beta1 ** self.t)
        self.v_db_corrected = self.v_db / (1 - beta1 ** self.t)
        self.s_dW = beta2 * self.s_dW + (1 - beta2) * np.square(self.dW)
        self.s_db = beta2 * self.s_db + (1 - beta2) * np.square(self.db)
        self.s_dW_corrected = self.s_dW / (1 - beta2 ** self.t)
        self.s_db_corrected = self.s_db / (1 - beta2 ** self.t)
        self.t += 1
        self.W = self.W - self.learning_rate * (self.v_dW_corrected / (np.sqrt(self.s_dW_corrected) + self.epsilon))
        self.b = self.b - self.learning_rate * (self.v_db_corrected / (np.sqrt(self.s_db_corrected) + self.epsilon))

    def update_parameters(self, optimizer=None):
        """
        Updates parameters 
        Parameters:
        Optimizer -- the optimizer used (default) : None           
        """
        if optimizer == 'adam':
            self.Adam()
        else:
            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db

class Pooling2D:
    """
    2D Pooling layer for down-sampling image data.

    Parameters:
    filter_size (int) -- size of the pooling window
    stride (int) -- the stride of the sliding window
    mode (str, optional) -- the pooling operation to use, either 'max' or 'average' (default is 'max')
    """

    def __init__(self, filter_size, stride, mode='max'):
        """
        Initialize the parameters of the pooling layer.

        Parameters:
        filter_size (int) -- size of the pooling window
        stride (int) -- the stride of the sliding window
        mode (str, optional) -- the pooling operation to use, either 'max' or 'average' (default is 'max')
        """
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

    def forward(self, A_prev):
        """
        Implements the forward pass of the pooling layer

        Parameters:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        f = self.filter_size
        stride = self.stride
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        A = np.zeros((m, n_H, n_W, n_C))
        for i in range(m):
            for h in range(n_H):
                vert_start = h * stride
                vert_end = vert_start + f
                for w in range(n_W):
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    for c in range(n_C):
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        if self.mode == 'max':
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == 'average':
                            A[i, h, w, c] = np.mean(a_prev_slice)
        self.cache = A_prev
        return A

    def create_mask_from_window(self, x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.

        Parameters:
        x -- Array of shape (f, f)

        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        mask = x == x.max()
        return mask

    def distribute_value(self, dz, shape):
        """
        Distributes the input value in the matrix of dimension shape

        Parameters:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)
        a = np.ones((n_H, n_W)) * average
        return a

    def backward(self, dA):
        """
        Implements the backward pass of the pooling layer

        Parameters:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input

        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
        A_prev = self.cache
        stride = self.stride
        f = self.filter_size
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (m, n_H, n_W, n_C) = dA.shape
        dA_prev = np.zeros(A_prev.shape)
        for i in range(m):
            a_prev = A_prev[i, :, :, :]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        if self.mode == 'max':
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
                        elif self.mode == 'average':
                            da = dA[i, h, w, c]
                            shape = (f, f)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += self.distribute_value(da, shape)
        return dA_prev

class Flatten:
    """
    A class for flattening the input tensor in a neural network.
    """

    def __init__(self):
        """
        Initialize the input shape to None.
        """
        self.input_shape = None

    def forward(self, X):
        """Implement the forward pass.

        Parameters:
        X (numpy.ndarray): The input tensor.

        Returns:
        numpy.ndarray: The flattened input tensor.
        """
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        """Implement the backward pass.

        Parameters:
        dout (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
        numpy.ndarray: The reshaped gradient tensor.
        """
        return dout.reshape(self.input_shape)

class Dense:
    """

    A class representing a dense layer in a neural network.
    """

    def __init__(self, units, activation='relu', optimizer=None, learning_rate=0.001):
        """
        Initialize the dense layer with the given number of units and activation function.

        Parameters:
        -----------
        :units: (int), the number of units in the dense layer.
        :activation: (str), the activation function to use, either 'relu' or 'softmax'.
        :optimizer: (str), the optimizer to use for updating the weights.
        : learning_rate: (float), the learning rate to use during training.
        """
        self.units = units
        self.W = None
        self.b = None
        self.activation = activation
        self.input_shape = None
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, A):
        """
        Perform the forward pass of the dense layer.

        Parameters:
        -----------
        :A: (ndarray), the input data of shape (batch_size, input_shape).

        :return: (ndarray), the output of the dense layer, shape (batch_size, units).
        """
        if self.W is None:
            self.initialize_weights(A.shape[1])
        self.A = A
        out = np.dot(A, self.W) + self.b
        if self.activation == 'relu':
            out = np.maximum(0, out)
        elif self.activation == 'softmax':
            out = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)
        return out

    def initialize_weights(self, input_shape):
        """
        Initialize the weights of the dense layer.
        Parameters:
        -----------
            input_shape: (int), the shape of the input data.
        """
        np.random.seed(0)
        self.input_shape = input_shape
        self.W = np.random.randn(input_shape, self.units) * 0.01
        self.b = np.zeros((1, self.units))

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Compute the cross entropy loss between the true labels and the predicted labels.

        Parameters
        ----------
          y_true: (ndarray), the true labels of shape (batch_size, num_classes).
          y_pred: (ndarray), the predicted labels of shape (batch_size, num_classes).

        Returns
        -------
        numpy.ndarray, the cross entropy loss.
        """
        loss = -np.mean(y_true * np.log(y_pred + 1e-07))
        return loss

    def backward(self, dout):
        """
        Perform the backward pass for this dense layer.

        Parameters
        ----------
        dout : numpy.ndarray
            Gradients of the loss with respect to the output of this layer.
            Shape: (batch_size, units)

        Returns
        -------
        numpy.ndarray
            Gradients of the loss with respect to the input of this layer.
            Shape: (batch_size, input_shape)
        """
        dA = np.dot(dout, self.W.T)
        self.dW = np.dot(self.A.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        self.update_parameters(self.optimizer)
        return dA

    def update_parameters(self, optimizer=None):
        """
         Updates parameters using choosen optimizer
         Parameters:
         Optimizer -- the optimizer used (default) : None
         """
        if self.optimizer == 'adam':
            self.Adam()
        else:
            self.W -= self.learning_rate * self.dW
            self.b -= self.learning_rate * self.db

    def Adam(self, beta1=0.9, beta2=0.999):
        """
        Update parameters using Adam
    
        Parameters:
        beta1 -- Exponential decay hyperparameter for the first moment estimates
        beta2 -- Exponential decay hyperparameter for the second moment estimates
        """
        self.v_dW = np.zeros(self.W.shape)
        self.v_db = np.zeros(self.b.shape)
        self.s_dW = np.zeros(self.W.shape)
        self.s_db = np.zeros(self.b.shape)
        self.v_dW = beta1 * self.v_dW + (1 - beta1) * self.dW
        self.v_db = beta1 * self.v_db + (1 - beta1) * self.db
        self.v_dW = self.v_dW / (1 - beta1 ** 2)
        self.v_db = self.v_db / (1 - beta1 ** 2)
        self.s_dW = beta2 * self.s_dW + (1 - beta2) * np.square(self.dW)
        self.s_db = beta2 * self.s_db + (1 - beta2) * np.square(self.db)
        self.s_dW = self.s_dW / (1 - beta2 ** 2)
        self.s_db = self.s_db / (1 - beta2 ** 2)
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.db

class CNN:
    """
    A class representing a Convolutional Neural Network.
    
    Parameters:
    -----------
    layers : list
        A list of instances of the layer classes in this network.
    learning_rate : float, optional (default=0.001)
        The learning rate for the network.
    optimizer : object, optional
        An instance of an optimization algorithm.
    
    Attributes:
    -----------
    layers : list
        A list of instances of the layer classes in this network.
    learning_rate : float
        The learning rate for the network.
    optimizer : str
        Optimziation algorithm to be used
    """

    def __init__(self, layers, learning_rate=0.001, optimizer=None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.initialize_network()

    def initialize_network(self):
        """
        A method to initialize the network. It sets the learning rate and optimizer 
        of the layers in the network to the network's learning rate and optimizer.
        """
        for layer in self.layers:
            if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                layer.learning_rate = self.learning_rate
                layer.optimizer = self.optimizer

    def forward(self, inputs):
        """
        A method to perform the forward pass of the network.
        
        Parameters:
        -----------
        inputs : numpy array
            The input to the network.
        
        Returns:
        --------
        outputs : numpy array
            The output of the network.
        """
        inputs = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, inputs):
        """
        A method to perform the backward pass of the network.
        
        Parameters:
        -----------
        inputs : numpy array
            The input to the network's backward pass.
        
        Returns:
            None
        """
        inputs = self.layers[-1].backward(inputs)
        for layer in reversed(self.layers[:-1]):
            inputs = layer.backward(inputs)

    def compute_cost(self, y_true, y_pred):
        """
        A method to compute the cost of the network's predictions.
        
        Parameters:
        -----------
        y_true : numpy array
            The true labels.
        y_pred : numpy array
            The predicted labels.
        
        Returns:
        --------
        cost : float
            The cost of the network's predictions.
        """
        if isinstance(self.layers[-1], Dense):
            cost = self.layers[-1].cross_entropy_loss(y_true, y_pred)
            return cost
        else:
            raise ValueError('The last layer in the layers list should be a Dense layer.')

    def step_decay(self, epoch, lr, decay_rate=0.1, decay_step=10, lowest_learning_rate=1e-05):
        """
        A function that implements step decay for the learning rate.

        Parameters:
        -----------
        epoch : int
            The current epoch number.
        lr : float
            The current learning rate.
        decay_rate : float, optional (default=0.1)
            The decay rate of the learning rate.
        decay_step : int, optional (default=10)
            The number of epochs after which the learning rate will be decayed.

        Returns:
        --------
        new_lr : float
            The updated learning rate after decay.
        """
        if lr > lowest_learning_rate:
            new_lr = lr * decay_rate ** (epoch // decay_step)
        else:
            new_lr = lr
        return new_lr

    def fit(self, X, y, epochs=10, decay_rate=0.2, print_cost=True, plot_cost=False):
        """Trains the CNN model on the input data (X) and target data (y)

        Parameters:
            X (np.ndarray): Input data with shape (number_of_examples, height, width,  num_channels)
            y (np.ndarray): Target data with shape (number_of_examples, num_classes)
            epochs (int): Number of iterations to train the model. Default is 10.
            print_cost (bool): If True, print cost value for each iteration. Default is True.
            plot_cost (bool): If True, plot cost value for each iteration. Default is False.

        Returns:
            None
        """
        costs = []
        for i in range(epochs):
            self.learning_rate = self.step_decay(i, self.learning_rate, decay_rate)
            predictions = self.forward(X)
            cost = self.compute_cost(y, predictions)
            accuracy = (np.argmax(predictions, axis=1) == np.argmax(y, axis=1)).mean()
            dout = predictions - y
            gradients = self.backward(dout)
            costs.append(cost)
            if print_cost:
                print(f'the cost for iteration {i} = {cost}, accuracy = {str(accuracy * 100)}%')
        if plot_cost:
            fig = px.line(y=np.squeeze(costs), title='Cost', template='plotly_dark')
            fig.update_layout(title_font_color='#f6abb6', xaxis=dict(color='#f6abb6'), yaxis=dict(color='#f6abb6'))
            fig.show()

    def predict(self, X):
        """Make predictions on the input data (X) using the trained CNN model.

        Parameters:
            X (np.ndarray): Input data with shape (number_of_examples, height, width,  num_channels)

        Returns:
            np.ndarray: Predictions with shape (number_of_examples, num_classes)
        """
        predictions = self.forward(X)
        return predictions

    def evaluate(self, X, y):
        """Evaluate the performance of the CNN model on the input data (X) and true labels (y).

        Parameters:
            X (np.ndarray): Input data with shape (number_of_examples, height, width,  num_channels)
            y (np.ndarray): True labels with shape (number_of_examples, num_classes)

        Returns:
            float: Loss calculated using the model's loss function
            float: Accuracy score calculated as (number of correct predictions) / (number of examples)
        """
        y_pred = self.predict(X)
        loss = self.compute_cost(y, y_pred)
        accuracy = (np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)).mean() * 100
        return f'{accuracy}%'
conv2d = Conv2D(8, 3, padding=0, learning_rate=0.001)
maxpool = Pooling2D(2, 2, 'max')
flatten = Flatten()
dense_relu = Dense(128, activation='relu')
dense = Dense(10, activation='softmax')
layers = [conv2d, maxpool, flatten, dense_relu, dense]
cnn = CNN(layers, learning_rate=0.001, optimizer='adam')