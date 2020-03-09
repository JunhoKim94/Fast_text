import numpy as np
import matplotlib.pyplot as plt
from preprocess import *


def plot(acc_stack, loss_stack, epochs):
    a = [i for i in range(epochs + 1)]
    
    #plt.figure(figsize = (10,8))
    fig , ax1 = plt.subplots()
    ax2 = ax1.twinx()
    acc = ax1.plot(a, acc_stack, 'r', label = 'Accuracy')
    loss = ax2.plot(a, loss_stack, 'b', label = 'loss')
    plt.legend()
    ax1.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax1.set_ylabel("accuracy")

    ax = acc + loss
    labels = [l.get_label() for l in ax]
    plt.legend(ax, labels, loc =2)

    plt.show()
    
def evaluate(test, label, model, word2idx, n_grams):
    total = len(test)
    score = 0
    to = 0

    for i in range(total):
        dev = word_to_id([test[i]], [label[i]], word2idx, n_grams)
        length = dev[0, -2]
        x = dev[0, :length]
        y = dev[0, -1] - 1

        y_val = np.argmax(model.forward(x), axis = 1) 
        score += len(y_val[y_val == y])
        to += length
    return score / total