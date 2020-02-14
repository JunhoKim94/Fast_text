import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear, sigmoid
from preprocess import *
import pickle

class Fasttext:
    def __init__(self, input_size, embed_size, hidden, output):
        self.embed = Embedding(input_size, embed_size)
        self.hidden = Linear(embed_size, hidden)
        self.output_layer = Linear(hidden, output)

        self.layer = [self.embed, self.hidden, self.output_layer]

    def forward(self, x):
        '''
        x = list of vocab(index) = (batch,S)
        '''

        output = self.embed.forward(x)

        output = np.sum(output, axis = 1)

        output = self.hidden.forward(output)
        output = self.output_layer.forward(output)

        return output

    def backward(self, dev):
        '''
        dev = (Batch, class)
        '''
        dout = self.output_layer.backward(dev)
        dout = self.hidden.backward(dout)
        self.embed.backward(dout)