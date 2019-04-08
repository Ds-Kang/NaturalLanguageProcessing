from __future__ import print_function

import torch

class TwoLayerNet(object):
  """
  two-layer-perceptron.
  Input dimension : N
  Hidden layer dimension : H
  Output dimension : C

  Train a network with softmax loss function.
  Use ReLU as activation function of Hidden layer.

  The structure of the network as follows

  input - linear layer - ReLU - linear layer - output
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize a model, weights are small random values and biases are zero.
    Weights and biases are stored in a dictionary, "self.params".

    W1: Weight of the first layer; (D, H)
    b1: Bias of the first layer; (H,)
    W2: Weight of the second layer; (H, C)
    b2: Bias of the second layer; (C,)

    Inputs:
    - input_size: Dimension of input data
    - hidden_size: The number of neurons in hidden layer
    - output_size: Output dimension
    """
    self.params = {}
    self.params['W1'] = std * torch.randn(input_size, hidden_size)
    self.params['b1'] = torch.zeros(hidden_size)
    self.params['W2'] = std * torch.randn(hidden_size, output_size)
    self.params['b2'] = torch.zeros(output_size)

  def loss(self, X, y=None):
    """
    Calculate loss and gradient of neural network.

    Inputs:
    - X: Input data. shape (N, D). Each X[i] is one of training samples, and total N number of samples are given as input.
    - y: Training label vector. y[i] is the integer label of X[i].
      Given y, return loss and gradient. Otherwise, return only output

    Returns:
    Not given y, return a score matrix of shape (N, C)
    scores[i, c] is the score of class c about input X[i]

    Given y, return (loss, grads) tuple
    loss: loss of training batch (scalar)
    grads: Dictionary of {Name of a parameter : gradient} (Name of a parameter is same as the key of self.params)
    """
    # Load weight and bias from Dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.size()

    # Forward path
    scores = None
    #############################################################################
    #   TODO: Execute forward path, store results in 'scores' (shape : (N, C))  #
    #           input - linear layer - ReLU - linear layer - output             #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # Return score if given the ground truth(target)
    if y is None:
      return scores

    # Loss
    loss = None
    e = torch.exp(scores)
    softmax = e / torch.sum(e, dim=1, keepdim=True)
    #############################################################################
    #     TODO: Calculate loss with output and store it in 'loss'(scalar)       #
    #                loss function : negative log likelihood                    #
    #                    Use softmax probability in 'softmax'                   #
    #                     'y' indicates the index of target                     #
    #       Apply -log to the probability of target and get mean of it          #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward path(Gradient)
    grads = {}
    ####################################################################################
    # TODO: Calculate gradient of weight and bias and store them in 'grads'(Dictionary)#
    #             Set keys of dictionary same to them of self.params                   #
    #          grads['W1'] has the same shape of self.params['W1']                     #
    #                Calculate gradient step-by-step from softmax                      #
    ####################################################################################
    pass
    ####################################################################################
    #                              END OF YOUR CODE                                    #
    ####################################################################################

    return loss, grads

  def train(self, X, y,
            learning_rate=1e-3, learning_rate_decay=0.95,
            num_iters=100,
            batch_size=200, verbose=False):
    """
    Train a neural network using SGD optimizer

    Inputs:
    - X: shape (N, D) numpy array (training data)
    - y: shape (N,) numpy array(training labels; y[i] = c
                                  c is the label of X[i], 0 <= c < C)
    - learning_rate: Scalar learning rate
    - num_iters: Number of steps
    - batch_size: Number of training examples in a mini-batch.
    - verbose: Print progress if true
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # SGD optimization
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      loss, grads = self.loss(X, y=y)
      loss_history.append(loss)

      #########################################################################
      # TODO: Load gradient from 'grads' dictionary and conduct SGD update    #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))


      if it % iterations_per_epoch == 0:
        # Accuracy
        train_acc = (self.predict(X) == y).float().mean()
        train_acc_history.append(train_acc)

        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    return torch.argmax(self.loss(X),1)


