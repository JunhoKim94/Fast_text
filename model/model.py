import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear, sigmoid
from preprocess import *
import pickle
import torch

class Fasttext:
    def __init__(self, input_size, embed_size, hidden, output, padding_idx):
        self.embed = Embedding(input_size, embed_size, padding_idx)
        self.hidden = Linear(embed_size, hidden)
        self.output_layer = Linear(hidden, output)

        self.layer = [self.embed, self.hidden, self.output_layer]

        self.params =[]
        self.grads =[]

        for layer in self.layer:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self, x):
        '''
        x = list of vocab(index) = (batch,S)
        '''
        if len(x.shape) != 0:
            length = len(x)
        else:
            x = [x]
            length = 1

        output = self.embed.forward(x)
        #Average of words
        output = np.sum(output, axis = 0, keepdims = True)/length
        output = self.hidden.forward(output)
        output = self.output_layer.forward(output)
        return output

    def backward(self, dev, lr):
        '''
        dev = (Batch, class)
        '''
        dout = self.output_layer.backward(dev,lr)
        dout = self.hidden.backward(dout, lr)
        self.embed.backward(dout, lr)

class Fasttext_torch(torch.nn.Module):
    def __init__(self, input_size, embed_size, hidden, output, padding_idx):
        super(Fasttext_torch, self).__init__()

        self.embed = torch.nn.Embedding(input_size, embed_size, padding_idx = padding_idx)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(embed_size,hidden),
            torch.nn.Linear(hidden,output)
        )

    def forward(self, x):
        '''
        x = (batch , S)
        '''

        output = self.embed(x)

        output = torch.sum(output,dim = 1).squeeze(1)/x.shape[1]

        output = self.linear(output)

        return output