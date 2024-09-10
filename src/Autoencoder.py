from torch import nn

class Autoencoder(nn.Module):
    def __init__(self,
                 input_layer_size=2,
                 hidden_layer_sizes=[16,8],
                 bottleneck_size=2,
                 activation='relu',):
        super(Autoencoder,self).__init__()
        """
        Args:
            hidden_layer_sizes (list): The ith element represents the number of neurons in the ith hidden layer.
            activation (str, torch.nn.Module): Activation function for the hidden layer.
        """
        self._input_layer_size = input_layer_size
        self._hidden_layer_sizes = hidden_layer_sizes
        self._bottleneck_size = bottleneck_size
        self._activation = activation
 
        self._encoder = nn.ModuleList()
        self._decoder = nn.ModuleList()
        
                
        #======================================================================
        # Create Encoder
        #======================================================================
        # encoder includes bottleneck
        encoder_layer_sizes = [input_layer_size] + hidden_layer_sizes + [bottleneck_size]
        encoder_activations = [activation]*len(hidden_layer_sizes) + [None]

        input_size = encoder_layer_sizes[0]
        for size, activa in zip(encoder_layer_sizes[1:], encoder_activations):
            self._encoder.append(nn.Linear(input_size,size))
            
            if activa is not None:
                self._encoder.append(self.__activation(activa))

            input_size = size

        self._encoder = nn.Sequential(*self._encoder)
        
        #======================================================================
        # Create Decoder
        #======================================================================
        # decoder excludes bottleneck
        decoder_layer_sizes = hidden_layer_sizes[::-1] + [input_layer_size]
        decoder_activations = encoder_activations               

        for size, activa in zip(decoder_layer_sizes, decoder_activations):
            self._decoder.append(nn.Linear(input_size,size))
            
            if activa is not None:
                self._decoder.append(self.__activation(activa))

            input_size = size

        self._decoder = nn.Sequential(*self._decoder)
        
        
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
        x = self._encoder(x)
        x = self._decoder(x)
        
        return x
    
    
    def encode(self, x):
        x = self._encoder(x)

        return x

    
    def decode(self, x):
        x = self._decoder(x)
        
        return x
        

    @property
    def input_layer_size(self):
        return self._input_layer_size

    
    @property
    def hidden_layer_sizes(self):
        return self._hidden_layer_sizes

    
    @property
    def bottleneck_size(self):
        return self._bottleneck_size

    
    @property
    def activation(self):
        return self._activation    