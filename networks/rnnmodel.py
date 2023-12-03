import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type="LSTM", token_num=125, embed=45, layer_num=1, dropout=0.8, tie_weights=False):
        super(RNNModel, self).__init__()
        self.token_num = token_num
        self.embed = embed
        self.layer_num = layer_num
        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        #self.encoder = nn.Embedding(token_num, embed)
        if self.rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(input_size=self.embed, hidden_size=self.token_num, num_layers=self.layer_num, bias=True, batch_first=True, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size=self.embed, hidden_size=self.token_num, num_layers=self.layer_num, nonlinearity=nonlinearity, bias=True, batch_first=True, dropout=dropout)
        self.decoder=None
        #self.decoder = nn.Linear(128, self.token_num)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
            # if self.embed != embed:
            #     raise ValueError('When using the tied flag, nhid must be equal to emsize')
            # self.decoder.weight = self.encoder.weight

    def _init_weights(self):
        initrange = 0.0001
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def _init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.layer_num, batch_size, self.token_num).zero_()),
                    Variable(weight.new(self.layer_num, batch_size, self.token_num).zero_()))
        else:
            return Variable(weight.new(self.layer_num, batch_size, self.token_num).zero_())

    def forward(self, inputs):
        #inputs = torch.as_tensor(inputs,dtype=torch.float64)
        #inputs = torch.transpose(inputs, 2,1)
        # embedded = self.drop(inputs)
        #print(embedded.shape)
        batch_size = (np.shape(inputs))[0]
        #print(batch_size)
        #if batch_size != 128:
        #    raise ValueError(f'batch size is not 128, but {batch_size}')
        hidden = self._init_hidden(batch_size)
        #print(np.shape(inputs))
        output, hidden = self.rnn(inputs, hidden)
        #print(np.shape(output), np.shape(hidden))
        output = self.drop(output)
        #print(output.shape)
        #print(self.decoder)
        output = output.reshape(output.size(0), -1) #data copy problem of reshape (view)
        #print(output.shape)
        decoded = self.decoder(output)
        #print(decoded.shape)
        #x = decoded.view(output.size(0), output.size(1), decoded.size(1), hidden)
        return decoded

    # OUT_FEATURES updated to NUMCLASS
    def Incremental_learning(self, numclass,device):
        #self.numclass = numclass
        if self.decoder is None:
            in_feature = self.token_num * self.token_num
            self.decoder = nn.Linear(in_feature, numclass, bias=True)
            self.decoder.to(device)
            self._init_weights()
        else:
            weight = self.decoder.weight.data
            bias = self.decoder.bias.data
            in_feature = self.decoder.in_features
            out_feature = self.decoder.out_features
            del self.decoder 
            self.decoder = nn.Linear(in_feature, numclass, bias=True)
            self.decoder.weight.data[:out_feature] = weight
            self.decoder.bias.data[:out_feature] = bias
            self.decoder.to(device)
        self.features_dim = self.decoder.in_features
        print("features_dim : ", self.decoder.in_features)

        
    