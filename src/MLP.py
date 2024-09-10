
import torch.nn as nn
 
class MLP(nn.Module):
    def __init__(self,
                 input_layer_size=2,
                 hidden_layer_sizes=[16, 8],
                 output_layer_size=1,
                 activation='relu'):
        super(MLP, self).__init__()
        """
        Args:
            hidden_layer_sizes (list): The ith element represents the number of neurons in the ith hidden layer.
            activation (str, torch.nn.Module): Activation function for the hidden layers.
        """
        self._input_layer_size = input_layer_size
        self._hidden_layer_sizes = hidden_layer_sizes
        self._output_layer_size = output_layer_size
        self._activation = activation
 
        self._layers = nn.ModuleList()
 
        # Create hidden layers with specified activations
        input_size = input_layer_size
        for size in hidden_layer_sizes:
            self._layers.append(nn.Linear(input_size, size))
            self._layers.append(self.__activation(activation))
            input_size = size
 
        # Add the output layer
        self._layers.append(nn.Linear(input_size, output_layer_size))
 
        self._mlp = nn.Sequential(*self._layers)
 
    def __activation(self, activation):
        """
        Handles creation of activation function instances.
        Args:
            activation (str, torch.nn.Module): Activation function for the hidden layer.
 
        Returns:
            activation function (nn.Module)
        """
        act = None
 
        if isinstance(activation, str):
            if activation == "relu":
                act = nn.ReLU(True)
            elif activation == "tanh":
                act = nn.Tanh()
            elif activation == "sigmoid":
                act = nn.Sigmoid()
            else:
                raise ValueError(f"Unknown activation function: {activation}")
 
        elif isinstance(activation, nn.Module):
            act = activation
        else:
            raise TypeError(f"Activation must be either a string or a nn.Module, but got {type(activation)}")
 
        return act
 
    def forward(self, x):
        return self._mlp(x)
 
    @property
    def input_layer_size(self):
        return self._input_layer_size
 
    @property
    def hidden_layer_sizes(self):
        return self._hidden_layer_sizes
 
    @property
    def output_layer_size(self):
        return self._output_layer_size
 
    @property
    def activation(self):
        return self._activation