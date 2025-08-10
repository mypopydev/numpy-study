"""
Simple Neural Network Implementation

This module implements a basic 3-layer neural network with sigmoid activation functions
and identity function for the output layer. It demonstrates forward propagation
through a neural network with predefined weights and biases.
"""

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid activation function.
    
    The sigmoid function maps any real-valued number to a value between 0 and 1,
    making it useful for binary classification problems.
    
    Args:
        x (numpy.ndarray): Input array of any shape
        
    Returns:
        numpy.ndarray: Output array with the same shape as x, with values between 0 and 1
        
    Examples:
        >>> sigmoid(np.array([0, 1, -1]))
        array([0.5, 0.73105858, 0.26894142])
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    Compute the Rectified Linear Unit (ReLU) activation function.
    
    ReLU returns the input if it's positive, otherwise returns 0.
    This function helps introduce non-linearity and addresses the vanishing gradient problem.
    
    Args:
        x (numpy.ndarray): Input array of any shape
        
    Returns:
        numpy.ndarray: Output array with the same shape as x, with values >= 0
        
    Examples:
        >>> relu(np.array([-1, 0, 1]))
        array([0, 0, 1])
    """
    return np.maximum(0, x)


def identity_function(x):
    """
    Compute the identity function (linear activation).
    
    The identity function simply returns the input unchanged.
    This is commonly used as the activation function for the output layer
    in regression problems.
    
    Args:
        x (numpy.ndarray): Input array of any shape
        
    Returns:
        numpy.ndarray: Output array identical to the input
        
    Examples:
        >>> identity_function(np.array([1, 2, 3]))
        array([1, 2, 3])
    """
    return x


def init_network():
    """
    Initialize a 3-layer neural network with predefined weights and biases.
    
    Creates a dictionary containing the weights and biases for a neural network
    with the following architecture:
    - Input layer: 2 neurons
    - Hidden layer 1: 3 neurons (with sigmoid activation)
    - Hidden layer 2: 2 neurons (with sigmoid activation) 
    - Output layer: 2 neurons (with identity activation)
    
    Returns:
        dict: Dictionary containing weights and biases for each layer
            - 'W1': Weight matrix for layer 1 (2x3)
            - 'b1': Bias vector for layer 1 (3,)
            - 'W2': Weight matrix for layer 2 (3x2)
            - 'b2': Bias vector for layer 2 (2,)
            - 'W3': Weight matrix for layer 3 (2x2)
            - 'b3': Bias vector for layer 3 (2,)
            
    Examples:
        >>> net = init_network()
        >>> print(net['W1'].shape)
        (2, 3)
    """
    network = {}

    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    """
    Perform forward propagation through the neural network.
    
    Computes the output of the neural network by propagating the input
    through all layers using the specified weights and biases.
    
    Args:
        network (dict): Dictionary containing weights and biases for each layer
            Must contain keys: 'W1', 'W2', 'W3', 'b1', 'b2', 'b3'
        x (numpy.ndarray): Input array with shape (n_features,)
            where n_features should match the number of input neurons (2)
            
    Returns:
        numpy.ndarray: Output array with shape (n_outputs,)
            where n_outputs is the number of output neurons (2)
            
    Raises:
        KeyError: If required network parameters are missing
        ValueError: If input dimensions don't match network architecture
        
    Examples:
        >>> net = init_network()
        >>> x = np.array([1.0, 0.5])
        >>> y = forward(net, x)
        >>> print(y.shape)
        (2,)
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # Forward propagation through layer 1
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    # Forward propagation through layer 2
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # Forward propagation through output layer
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


# Example usage
if __name__ == "__main__":
    # Initialize the neural network
    net = init_network()
    
    # Define input data
    x = np.array([1.0, 0.5])
    
    # Perform forward propagation
    y = forward(net, x)
    
    # Print the result
    print("Input:", x)
    print("Output:", y)