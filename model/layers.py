import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

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

class Ce_losswithsoftmax:
    def __init__(self):
        self.params = None
        self.grads = None
        self.eps = 1e-7

        self.y_pred, self.target = None, None
        self.loss = 0

    def forward(self, y_pred, target):
        '''
        y_pred = (Batch, class num)
        target = (Batch, 1)
        '''
        #if len(target.shape) == 0:
        #    target = np.expand_dims(target, 0)
        #batch = target.shape[0]
        self.y_pred = softmax(y_pred)
        self.target = target


        self.loss = -np.log(self.y_pred[0,self.target] + self.eps)

        self.loss = np.sum(self.loss)

        return self.loss
    def backward(self):
        self.y_pred[0,self.target] -= 1

        return self.y_pred

class Embedding:
    def __init__(self, input_size, output_size, padding_idx):

        W = np.random.uniform(low = -0.5/ output_size, high = 0.5/ output_size, size = (input_size, output_size))

        #self.W = np.random.uniform(size = (self.input_size, self.output_size))
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        '''
        if padding_idx:
            self.padding_idx = padding_idx
        else:
            self.padding_idx = None
        '''
    def forward(self, x):
        '''
        x = list or array
        
        if self.padding_idx is not None:
            self.idx = []
            for i in x:
                if (i != self.padding_idx):
                    self.idx.append(i)
        else:
            self.idx = x
        '''
        self.idx = x
        W, = self.params
        output = W[self.idx]

        return output

    def backward(self, dout, lr):
        '''
        idx 해당 하는 w 만 grad = 1 * dout
        나머지 0
        '''
        #dW, = self.grads
        W,  = self.params
        #dW[self.idx] += dout
        W[self.idx] -= dout * lr
        #np.add.at(dW, self.idx, dout)

    def _zero_grad(self):
        dW, = self.grads
        dW[...] = 0


class Linear:
    def __init__(self, input_size, output_size):

        W = np.random.uniform(low = -1, high = 1, size = (input_size, output_size))
        b = np.random.uniform(low = -1, high = 1, size = (1, output_size))
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.params = [W, b]

    def forward(self, x):
        '''
        W = (D,H)
        x = (N,D)

        out : (N,D)
        '''

        W,b = self.params
        self.x = x
        output = np.matmul(self.x,W) + b

        return output

    def backward(self, dout, lr):
        '''
        input: d_out (N,H)
        self.x : (N,D)
        output: dW : (D,H)
                db : (1,H)
        '''
        W,b = self.params

        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        W -= lr * dW
        b -= lr * db
        #self.grads[0][...] = dW
        #self.grads[1][...] = db

        return dx

def softmax(z):
    #numerically stable softmax
    z = z - np.max(z, axis = 1 , keepdims= True)
    _exp = np.exp(z)
    _sum = np.sum(_exp,axis = 1, keepdims= True)
    sm = _exp / _sum

    return sm

def cross_entropy_loss(z, target):
    if z.ndim == 1:
        target = target.reshape(1, target.size)
        z = z.reshape(1, z.size)

    if z.size == target.size:
        target = target.argmax(axis = 1)

    batch_size = z.shape[0]

    return -np.sum(np.log(z[np.arange(batch_size), t] + 1e-7)) / batch_size


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        '''
        input: d_out (N,H)
        self.x : (N,H)
        output: 
        '''
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

