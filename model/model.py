import numpy as np
from model.layers import Embedding, Sigmoid, Softmax, Linear, sigmoid
from preprocess import *
import pickle

class BCELossWithSigmoid:
    def __init__(self):
        self.params = None
        self.grads = None
        self.eps = 1e-7

        self.y_pred , self.target = None, None
        self.loss = None

    def forward(self, y_pred, target):

        self.target = target
        self.y_pred = sigmoid(y_pred)

        number = target.shape[0]

        self.loss = -self.target * np.log(self.y_pred + self.eps) - (1 - self.target) * np.log(1 - self.y_pred + self.eps)

        self.loss = np.sum(self.loss) / number

        return self.loss

    def backward(self):
        dx = self.y_pred - self.target
        return dx

class Negative_Sampling:
    def __init__(self, vocab_size, sub_vocab, projection):
        self.Embedding = Embedding(sub_vocab, projection)
        self.N_Embdding = Embedding(vocab_size , projection)

        self.layers = [self.Embedding, self.N_Embdding]
        self.params = []
        self.grads = []

        for layer in self.layers:
            self.params.append(layer.params)
            self.grads.append(layer.grads)

    def forward(self, x, sampled):
        '''
        x = (1, D) D = subwords 개수
        sampled = (1, sampled)
        '''
        
        #D x projection
        out = self.Embedding.forward(x)
        #1 x projection
        out = np.sum(out, axis = 0, keepdims= True)

        #sampled x projection
        vec = self.N_Embdding(sampled)

        output = np.sum(out * vec, axis = 1)
        return output

    def backward(self, dout):
        
